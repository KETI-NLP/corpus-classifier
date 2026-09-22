# Transformers로 직접 사용하기

v0.2.0은 Hugging Face의 custom model / AutoClass 규약을 따릅니다. 저장소 루트의 config, tokenizer, 3개 safetensors shard와 index로 로딩합니다. 별도의 Qwen 베이스 다운로드나 GitHub 패키지 설치는 필요하지 않습니다. 검증 버전의 `torch==2.11.0`, `transformers==5.11.0`, `peft==0.19.0`을 설치하고 CUDA BF16 환경을 준비하세요.

이 모델 클래스는 Transformers 내장 클래스가 아니므로 직접 AutoClass 로딩에는 `trust_remote_code=True`가 필요합니다. 고정 commit의 Python 코드를 확인하고 사용하세요. 다운로드한 모델 코드를 실행하고 싶지 않다면 README의 GitHub CLI를 사용하세요. CLI는 설치된 패키지의 동일 클래스를 사용합니다.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

repo = "KETI-NLP/Qwen3.5-2B-CorpusClassifier"
revision = "9310cfb804fc51782a7475def4bba5d9db817445"  # immutable release commit

tokenizer = AutoTokenizer.from_pretrained(
    repo, revision=revision, trust_remote_code=True,
)
model = AutoModelForSequenceClassification.from_pretrained(
    repo, revision=revision, trust_remote_code=True,
    dtype=torch.bfloat16,
).to("cuda").eval()

batch = model.prepare_inputs(
    tokenizer,
    ["토양 수분과 관개가 작물 생장에 미치는 영향"],
    metadata=[{"language": "ko"}],
).to("cuda")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    logits = model(**batch).logits.float().cpu()  # [batch, 2100]
results = model.decode_logits(logits, sample_ids=["example-1"])
print(results[0]["roots"], results[0]["exact"])

```

`prepare_inputs`는 원래 metadata/body 포맷과 768토큰 제한을 적용합니다. `decode_logits`는 root/exact gate와 계층 중복 제거, auxiliary primary와 state 해석을 수행합니다. 범용 text-classification pipeline의 기본 softmax/sigmoid는 이 joint head의 최종 디코딩과 다릅니다.

직접 API는 라벨·점수를 반환합니다. 원문 참조, 토큰 해시, 잘림 여부, 재개 가능한 샤드와 Parquet 결과까지 필요하면 `corpus_classifier.Classifier` 또는 CLI를 사용하세요. 긴 원문을 자동 분할해 전체 주제를 통합하는 기능은 없습니다.

로컬 snapshot을 쓰려면 `repo`를 디렉터리 경로로 바꾸고 `revision`을 제거하며 `local_files_only=True`를 추가합니다. `model.save_pretrained(path)`도 지원합니다. 로딩 후 BF16으로 다시 저장하면 학습된 FP32 원본 텐서는 BF16으로 저장되므로 공식 원본 보존용 재배포에는 사용하지 마세요. 또한 이 메서드는 taxonomy/manifest 등 CLI 부속 파일을 자동 복사하지 않습니다. 공식 배포는 원본 FP32 학습 텐서를 전체 state dict에 넣어 저장했습니다.

참고: [Hugging Face custom models](https://huggingface.co/docs/transformers/custom_models).
