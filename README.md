# Corpus Classifier

학습된 **Qwen3.5-2B 기반 멀티라벨 분류기**로 대량 문서를 로컬 GPU에서 분류합니다. 하나의 입력을 한 번 인코딩해 33개 상위 분야, 1,029개 세부 코드와 문서 상태를 예측합니다.

코드 저장소: `KETI-NLP/corpus-classifier`  
모델 저장소: `KETI-NLP/Qwen3.5-2B-CorpusClassifier`

공개 모델과 코드를 별도로 내려받아 사용합니다. 로컬 체크아웃과 모델 디렉터리로도 모든 분류 기능을 사용할 수 있습니다.

## 제공 기능

- GPU마다 모델을 한 번 로딩하고 여러 샤드를 연속 처리합니다.
- GPU당 기본 batch 32, BF16, 최대 768토큰, last non-pad pooling을 사용합니다.
- JSONL, gzip JSONL, zstd JSONL, Parquet 입력을 스트리밍으로 준비합니다.
- 입력 ID 중복을 전체 작업에서 검사하고, 모델·입력·코드 해시를 고정합니다.
- 완료 샤드를 원자적으로 저장합니다. 재실행하면 완료 샤드는 재계산하지 않습니다.
- JSONL 예측과 Parquet의 `documents`, `exact_membership`, `root_membership`, `review_queue`를 제공합니다.
- 모델 다운로드는 명시적 별도 명령입니다. 준비·분류·내보내기에서는 네트워크 연결을 차단합니다.

이 저장소에는 모델 가중치, 학습 데이터, 원문 코퍼스, API 키가 없습니다. 모델은 Hugging Face 저장소에서 별도로 받습니다.

## 1. 설치

Linux, Python 3.11, BF16을 지원하는 CUDA GPU를 준비하세요. 실제 검증 환경은 NVIDIA B200, Python 3.11.14, PyTorch 2.11.0+cu130입니다. CPU 추론, Windows 파일 잠금, 양자화, 다른 추론 엔진은 검증 범위에 없습니다.

```bash
git clone https://github.com/KETI-NLP/corpus-classifier.git
cd corpus-classifier
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 드라이버에 맞는 PyTorch CUDA 배포본을 먼저 설치하세요.
python -m pip install '.[hub,data]'
corpus-classifier --help
```

로컬 배포본을 받았다면 그 디렉터리에서 `python -m pip install '.[hub,data]'`부터 실행합니다. 기본 의존성은 검증 버전으로 고정했습니다. `hub`는 다운로드, `data`는 zstd·Parquet 입출력용입니다. CUDA 빌드는 환경에 맞게 준비하되 다른 버전으로 변경했다면 동일성 검증을 다시 수행하세요.

## 2. 모델 받기

아래 명령은 Transformers 호환 버전 `v0.2.0`의 **40자리 commit hash**에 고정되어 있습니다. 다른 버전을 쓸 때는 Hugging Face의 Files and versions에서 해당 commit을 확인하세요.

```bash
corpus-classifier download \
  --repo-id KETI-NLP/Qwen3.5-2B-CorpusClassifier \
  --revision 9310cfb804fc51782a7475def4bba5d9db817445 \
  --output models/corpus-classifier

corpus-classifier verify --model models/corpus-classifier
```

`main` 대신 고정 commit을 요구해 재현 가능한 실행을 만듭니다. 공개 저장소 다운로드에 쓰기 토큰은 필요하지 않습니다. 이미 모델 파일을 받았다면 다운로드를 생략합니다. 모델 전체는 약 4.6 GB이며 베이스와 병합하지 않은 LoRA·분류 헤드를 루트의 표준 sharded safetensors에 함께 저장합니다. 추가 베이스 모델 다운로드는 필요하지 않습니다.

v0.2.0 모델은 루트 `config.json`, tokenizer, safetensors와 등록된 custom Transformers 클래스를 제공합니다. `AutoModelForSequenceClassification.from_pretrained`로 직접 로딩할 수 있으며, 그 경로에서는 `trust_remote_code=True`가 필요합니다. 아래 GitHub CLI는 설치된 패키지의 클래스를 사용하고 다운로드한 Python 코드를 실행하지 않습니다. 모델에 포함된 코드와 설치 코드의 SHA256도 비교합니다.

출력 2,100개에는 여러 종류의 head가 함께 있으므로 범용 `pipeline()`의 단일 softmax/sigmoid 결과를 최종 분야 라벨로 쓰면 안 됩니다. [Transformers 사용 예제](docs/transformers.md)의 `prepare_inputs` → forward → `decode_logits`를 사용하세요. 기존 v0.1.0 모델의 로딩도 유지하지만 새 모델은 코드 v0.2.0 이상이 필요합니다.

## 3. 입력 준비

UTF-8 JSONL의 한 줄은 한 문서입니다.

```json
{"sample_id":"doc-0001","text_window":"토양 수분과 관개가 작물 생장에 미치는 영향","metadata_compact":{"language":"ko","title":"관개 연구"},"source_family":"example","source_locator":{"file":"documents.jsonl","record":1},"raw_windowed":false}
```

| 필드 | 필수 | 의미 |
|---|---|---|
| `sample_id` | 예 | 작업 전체에서 유일한 비어 있지 않은 문자열 |
| `text_window` | 예 | 분류에 제공할 본문 구간 |
| `metadata_compact` | 아니요 | title, language, date, publisher, format_family만 모델 입력에 사용 |
| `source_family` | 아니요 | 출처별 운영 집계용. 기본 `unspecified` |
| `source_locator` | 아니요 | 후속 원문 조회용 JSON 값. 모델 입력에서는 제외 |
| `raw_windowed` | 아니요 | 상류 추출에서 원문을 축약했는지. boolean 또는 null |

```bash
corpus-classifier prepare \
  --input documents.jsonl \
  --output jobs/run-001 \
  --shard-rows 4096
```

기존 데이터가 `id`, `text` 필드라면 `--id-field id --text-field text`를 추가합니다. `.jsonl.gz`, `.jsonl.zst`, `.jsonl.zstd`, `.parquet`도 같은 필드를 사용합니다. ID 중복, 빈 본문, 잘못된 JSON은 조용히 버리지 않고 준비를 실패시킵니다.

준비는 전체 파일을 읽어 입력을 정규화하고 샤드로 저장하며, SQLite로 전체 ID 중복을 검사합니다. 원본은 시작·종료에 SHA256을 비교합니다. 실패 시 최종 manifest가 생성되지 않습니다. 같은 준비 명령을 다시 실행하면 이 명령이 소유한 `.preparing` 임시 디렉터리부터 재생성합니다. 준비 자체의 레코드 단위 재개는 제공하지 않으므로 매우 큰 코퍼스는 원본 파일 단위의 제한된 작업으로 나누세요.

기본 `--max-record-chars 8000000`은 JSONL 한 줄/본문의 비정상적인 크기를 제한합니다. Parquet는 128행 배치로 읽으므로 최대 원본 행 크기에 따라 메모리 사용량이 달라집니다. 전체 코퍼스를 RAM에 적재하지는 않습니다.

### 입력 범위 주의

분류 입력은 정제한 메타데이터와 본문을 합친 앞 **768토큰**입니다. 이 도구는 임의의 원문에 기존 연구용 head/tail 추출을 자동 적용하거나 여러 구간을 합치지 않습니다. 원래 검증한 window 정책을 재현하려면 상류에서 동일한 구간을 만들어 넣으세요. 긴 원문을 `text`로 직접 넣는 경우에는 prefix 분류입니다.

## 4. GPU에서 대량 분류

한 GPU:

```bash
corpus-classifier run \
  --manifest jobs/run-001/manifest.json \
  --model models/corpus-classifier \
  --output outputs/run-001 \
  --devices 0
```

8 GPU:

```bash
corpus-classifier run \
  --manifest jobs/run-001/manifest.json \
  --model models/corpus-classifier \
  --output outputs/run-001 \
  --devices 0,1,2,3,4,5,6,7 \
  --batch-size 32
```

GPU마다 독립 프로세스와 모델 복사본을 실행합니다. 모델 병렬화/DDP 학습은 사용하지 않습니다. `--devices`는 작업에 할당된 CUDA ID 또는 GPU UUID이며, 스케줄러 환경에서는 할당받은 GPU만 지정하세요. 샤드를 GPU에 순환 배정하므로 샤드가 GPU보다 충분히 많아야 병렬성이 나옵니다. 하나의 샤드는 한 GPU가 처리합니다.

기본 batch 32가 기존 운영 검증값입니다. VRAM 부족 시 새 출력 경로에서 작은 배치를 선택할 수 있지만, 배치 변경으로 수치·처리량이 달라질 수 있습니다. 모델·임계값·pooling·768토큰 제한은 고정입니다.

### 중단 후 재개

**같은 명령을 같은 출력 경로로 다시 실행**하세요. 완료한 샤드는 입력/출력 해시를 확인하고 건너뜁니다. 저장 도중 중단된 샤드만 다시 처리합니다. 모두 완료되어 있으면 모델을 로딩하지 않고 `already_complete`, `model_forwards: 0`을 반환합니다.

입력 manifest, 모델 manifest, Python 실행 코드, 배치 크기가 바뀌면 같은 출력 경로로 이어 쓰지 않습니다. 새 작업/출력 경로를 만드세요. 모델과 입력은 실행 중 읽기 전용으로 유지해야 합니다. 프로세스·샤드 잠금과 atomic rename을 지원하는 파일시스템이 필요합니다. 오류/OOM 발생 시 완료 결과는 유지하고 다른 실행 중 worker는 종료합니다.

추론 중 GPU 수를 바꾸어 재개할 수 있으나 실행기를 중복 실행하지 마세요. 이 버전은 한 호스트의 여러 GPU를 위한 실행기입니다. 여러 서버를 가로지르는 동적 작업 큐는 포함하지 않습니다.

## 5. 출력과 Parquet 변환

```text
outputs/run-001/
  run_binding.json
  shard-000000/
    predictions.jsonl
    receipt.json
  shard-000001/
    predictions.jsonl
    receipt.json
  session-....json
```

각 예측에는 `sample_id`, `roots`, `exact`, 선택한 라벨의 `root_probabilities`/`exact_probabilities`, `auxiliary_primary`, `document_state`, 토큰 수·잘림·미분류, 원문 참조와 입력 해시가 있습니다. `probabilities`라는 기존 필드 이름은 유지했지만 **교정된 정답 확률은 아닙니다**.

```bash
corpus-classifier export \
  --manifest jobs/run-001/manifest.json \
  --predictions outputs/run-001 \
  --output outputs/run-001-parquet
```

| 테이블 | 행의 단위 |
|---|---|
| `documents` | 문서 1개. 라벨 목록, 상태, 원문 참조, 입력 해시 |
| `exact_membership` | 문서와 세부 코드의 연결 1개 및 점수 |
| `root_membership` | 문서와 상위 코드의 연결 1개 및 점수 |
| `review_queue` | 미분류·토큰 잘림·상류 window 축약 사유가 있는 문서 |

내보내기는 전체 샤드가 완료되고 해시가 맞을 때만 최종 디렉터리를 게시합니다. 테이블별 Parquet 파일이 생성되므로 Arrow/DuckDB/Spark 등에서 dataset으로 읽을 수 있습니다. 출력 manifest에 파일 해시와 행 수를 남깁니다.

멀티라벨 문서는 여러 코드에 연결됩니다. 문서를 물리적으로 복제할 필요는 없습니다. 후속 중복 검사는 `(min(document_id), max(document_id))`로 문서 쌍을 중복 제거하고, 다른 라벨·미분류 문서 사이의 후보도 고려하세요. 이 프로그램은 LLM 중복 검사를 실행하지 않습니다.

## Python API

```python
from corpus_classifier import Classifier

model = Classifier("models/corpus-classifier", device="cuda:0")
results = model.predict([
    {"sample_id": "demo-1", "text_window": "토양 수분과 관개 연구",
     "metadata_compact": {"language": "ko"}}
])
print(results[0]["exact"])
```

`predict`에 배치를 전달하고 모델 인스턴스를 재사용하세요. 문서마다 `Classifier(...)`를 다시 생성하지 않습니다. API도 모델 로딩 시 파일 해시를 검사하고 원격 로딩을 사용하지 않습니다.

## 품질·속도의 검증 범위

기존 동일 가중치 운영 시험에서 B200 8장으로 준비된 입력 100,000문서를 100.73초(약 993문서/초)에 처리했습니다. 모델 로딩·읽기·추론·저장이 포함되며 원시 코퍼스 전체 추출은 포함하지 않습니다. 이 수치는 기존 실행기의 실측이며, 새 refactoring의 대규모 처리량 보장으로 제시하지 않습니다. 새 코드의 로더 동일성·GPU smoke·재개 검증은 `docs/validation.json`에 별도로 기록합니다.

독립 확인의 완전 합의 17문서에서 2B 모델의 precision/recall/micro-F1은 45.16%/58.33%/50.91%였습니다. 최소 20문서에 미달한 작은 모델 합의 표본으로 사람 gold나 전체 코퍼스 정확도가 아닙니다. 9B 교체의 이득이 입증되지 않아 기존 2B를 유지했습니다.

1,029개 코드가 모두 같은 수준으로 검증된 것은 아닙니다. 선택되지 않은 라벨을 검증된 음성으로, 미분류를 top1 확정 라벨로 바꾸지 마세요. `auxiliary_primary`는 보조 출력입니다. `truncated=false`도 원문 전체를 읽었다는 보장이 아닙니다. 단일 인코딩 모델은 라벨별 근거 인용이나 완전 판정을 생성하지 않습니다.

## 개발과 테스트

```bash
python -m pip install -e '.[data]'
python -m unittest discover -s tests -v
```

테스트는 합성 입력과 CPU mock 모델로 중복 ID, 잘못된 입력, 여러 파일 형식, 샤드 분할, 중단 전후 재개, 해시 불일치, 잠금, Parquet 관계를 검사합니다. 모델 정확도 평가나 실제 GPU 실행은 이 단위 테스트에 포함되지 않습니다.

아키텍처·재개 범위는 [architecture.md](docs/architecture.md), 조직 인증·업로드 절차는 [publishing.md](docs/publishing.md)를 참고하세요.

## 라이선스와 출처

- Python 코드·테스트·빌드 설정: **Apache-2.0**, [LICENSE](LICENSE).
- README와 프로젝트 설명 문서: **CC BY 4.0**, [문서 라이선스](docs/LICENSE-CC-BY-4.0.txt). 출처: KETI-NLP, Corpus Classifier.
- 모델 저장소의 추가 학습 LoRA·분류 헤드: **CC BY 4.0**.
- 번들에 포함된 원본 Qwen3.5-2B 베이스: 원래 **Apache-2.0**을 유지합니다.

Qwen3.5-2B 기반의 KETI-NLP 파생 분류기이며 Qwen 공식 모델 배포가 아닙니다. 베이스의 생성·멀티모달 예제를 이 분류기의 사용법으로 적용하지 마세요.
