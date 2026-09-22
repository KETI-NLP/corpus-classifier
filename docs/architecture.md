# 실행 구조와 재현성

## 모델 계약

v0.2.0의 HF 모델 디렉터리 루트에는 `config.json`, `model-*.safetensors`, `model.safetensors.index.json`, tokenizer와 custom model Python 파일이 있습니다. `PretrainedConfig`/`PreTrainedModel`을 구현하고 `register_for_auto_class`로 `AutoConfig`와 `AutoModelForSequenceClassification`을 등록했습니다. 표준 `save_pretrained`로 전체 state dict를 저장했으며 로딩 중 외부 베이스 모델을 가져오지 않습니다.

Hub에서 AutoClass를 직접 쓰면 `trust_remote_code=True`로 저장소의 모델 코드를 로딩합니다. GitHub 실행기는 설치된 동일 클래스와 `local_files_only=True`를 사용하며 Hub Python 파일은 실행하지 않습니다. 설치 코드와 모델에 동봉한 3개 Python 파일의 SHA256이 다르면 실행을 거절합니다. 이전 v0.1.0의 `model/base` + `model/adapter` 형식도 계속 읽을 수 있습니다.

원본 베이스와 학습된 LoRA·score head를 병합하지 않고 보존합니다. 학습된 49개 FP32 텐서는 그대로 저장하며 CUDA BF16 추론에서 변환합니다. 기존 runner와 동일하게 rotary frequency를 포함한 부동소수점 버퍼도 BF16으로 맞춥니다. 비활성 PEFT `original_module` head만 재현 가능한 0으로 저장하며 추론에는 사용하지 않습니다. tokenizer, pooling, 임계값과 최종 디코딩은 기존과 같습니다.

출력 순서는 root 33, exact 1,029, auxiliary primary 1,029, state 9입니다. state는 integrity 4, topic structure 3, labelability 2입니다. 상위 분야 점수 0.5 이상으로 gate한 뒤 세부 코드 0.95 이상을 선택하고 같은 계층의 조상/자손 중복을 제거합니다.

## 데이터 흐름

```text
정규화할 원본 파일
  → prepare: 전체 ID 중복 검사 + 샤드 저장 + immutable manifest
  → run: 모델·코드 검증 + 작업/출력 binding
      → GPU 0: 모델 1회 로딩 → 담당 샤드의 배치 추론
      → GPU 1: 모델 1회 로딩 → 담당 샤드의 배치 추론
      → ...
  → 샤드별 predictions.jsonl + receipt.json 원자적 게시
  → export: 완료 샤드 검증 → 정규화 Parquet 테이블
```

## 무결성과 재개 범위

- 작업 manifest는 상대 입력 경로를 사용하므로 작업 폴더를 통째로 이동할 수 있습니다.
- run binding은 입력 manifest SHA256, 모델 manifest SHA256, 설치된 Python 소스 파일 집합 SHA256, 배치 크기를 기록합니다.
- launcher는 모델 payload를 해시 검증합니다. 내부 worker는 launcher가 검증한 읽기 전용 모델을 사용하고 manifest 동일성을 확인합니다. 실행 중 모델 파일 변경을 허용하는 저장소로 사용하지 마세요.
- worker는 담당 입력을 실행 전 검사하고, 실제 처리 후 다시 해시 검사합니다. 완료 출력은 매 재개에 해시 검증합니다.
- `fsync` 후 디렉터리 rename으로 샤드를 게시합니다. 완료 전 죽으면 그 샤드는 다시 계산할 수 있고, 완료 후 죽으면 재사용합니다.
- 잠금은 Linux `flock`입니다. 로컬/공유 파일시스템이 `flock`, 원자적 rename, `fsync`를 올바르게 지원해야 합니다.
- SHA256은 파일 동일성 검증이며 배포자 서명이 아닙니다. 다운로드는 신뢰하는 조직의 고정 Hub commit을 사용하세요.

완료 판정은 샤드 단위입니다. GPU가 마지막 forward까지 실행했더라도 receipt 게시 전 종료했다면 해당 샤드가 다시 계산됩니다. 정확히 한 번의 GPU 계산을 보장하는 것은 아니며, 완료 결과가 중복 게시되지 않도록 합니다.

## 메모리와 처리량

입력 준비는 SQLite로 전체 문서 ID를 검사하고, 추론은 배치 단위로 문서를 읽습니다. worker의 ID 중복 검사 set은 한 샤드 크기에 한정됩니다. 원문 길이와 Parquet reader 배치 크기에 따라 CPU 메모리가 달라집니다. GPU 메모리는 배치·최대 토큰·모델 크기에 좌우됩니다.

모델은 worker가 담당하는 모든 샤드 동안 유지됩니다. 여러 작은 작업을 동시에 새 프로세스로 실행하면 로딩 비용이 증가하므로 충분한 크기의 작업/샤드로 구성합니다. 추출과 추론의 비동기 연결, 동적 GPU 작업 훔치기, 다중 호스트 스케줄링은 이 릴리스에 포함하지 않습니다.

## 학습 시 근거 보존 정책과의 관계

학습 데이터 일부는 두 판정 모델의 합의·범위 검증 후, 실제 입력에 근거 토큰이 보존된 양성만 사용했습니다. 완전 판정 분야·전체 입력·양성 근거가 보존돼야 그 분야의 나머지를 음성으로 사용했고 미확인은 손실에서 제외했습니다. 학습 직전 토큰 해시도 재검증했습니다.

운영 모델은 그 데이터로 학습한 분류기입니다. 추론마다 두 판정 모델을 다시 실행하거나 실제 근거 인용을 생성하지 않습니다. 예측을 신규 학습 정답으로 쓰려면 별도의 검증과 마스크 생성이 필요합니다. 기존 학습 데이터 전체가 이 엄격한 정책으로 재생성된 것은 아닙니다.
