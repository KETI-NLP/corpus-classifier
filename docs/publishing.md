# KETI-NLP 공개 배포 설정

## 이름과 공개 범위

권장 대상은 Hugging Face 모델 `KETI-NLP/Qwen3.5-2B-CorpusClassifier`, GitHub 코드 `KETI-NLP/corpus-classifier`이며 둘 다 public입니다. `KETI-NLP`는 조직/소유자 이름이고 슬래시 뒤가 개별 저장소 이름입니다.

공식 베이스 이름 `Qwen3.5-2B`에 용도 `CorpusClassifier`를 붙인 파생 모델명입니다. 공식 Qwen 배포로 오인되지 않도록 조직과 설명에 KETI-NLP를 표시합니다. 버전은 `v0.1.0` 같은 release/tag와 Hub commit hash로 관리합니다. 검증된 최대 입력이 768토큰이라는 점은 모델 카드에 표시하며, 이름만으로 베이스의 전체 기능을 주장하지 않습니다.

GitHub 저장소명은 특정 모델 크기에 종속되지 않아 향후 호환 모델을 추가할 수 있습니다. 현재 다른 모델의 호환성은 검증하지 않았습니다.

## 계정과 권한

1. 개인 계정으로 두 서비스에 로그인하고 `KETI-NLP` 조직에 속해 있는지 확인합니다.
2. GitHub는 조직 정책상 저장소 생성 권한과 대상 저장소 쓰기 권한이 필요합니다. 조직이 SSO를 사용하면 해당 인증도 완료합니다.
3. Hugging Face는 조직의 저장소 생성/쓰기 권한이 필요합니다. 개인 User Access Token에도 대상 조직/저장소 쓰기 범위를 부여합니다. 조직 권한과 토큰 권한을 모두 충족해야 합니다.
4. 인증은 CLI의 대화형 로그인이나 서비스가 지원하는 비밀 저장소를 이용합니다. 토큰을 README, Python 파일, Git 설정 URL, 채팅에 넣지 않습니다.

```bash
gh auth login --hostname github.com --git-protocol https --web
hf auth login
```

GitHub CLI가 없는 서버는 [공식 설치 안내](https://github.com/cli/cli#installation)에 따라 설치하거나 GitHub 웹에서 빈 저장소를 만들고 Git으로 push할 수 있습니다. Hugging Face CLI는 `python -m pip install huggingface-hub==1.15.0`으로 설치할 수 있습니다.

## 업로드 전에 확인할 배포 디렉터리

작업 트리 전체가 아니라 분리한 `github/`와 `huggingface/`만 각각 업로드합니다.

- `github/`: 소스, 테스트, pyproject, README, 설명 문서, 합성 예제. 가중치·원문 데이터·비밀 파일 제외.
- `huggingface/`: 루트 Transformers config·safetensors·tokenizer, custom model Python 3개, taxonomy, 모델 카드, 라이선스, 집계 평가, 파일 manifest. 학습 corpus·문서별 판정·내부 경로 제외.
- 코드 Apache-2.0, 추가 가중치·문서 CC BY 4.0, 베이스 Apache-2.0 보존.

검증 결과는 `github/docs/validation.json`에 있습니다. 로컬 staging에 원격 Git 저장소가 설정돼 있지 않다면 아래 최초 생성 명령을 사용합니다. 동일 이름의 기존 저장소가 있으면 그 내용을 먼저 확인하고 덮어쓰거나 force-push하지 마세요.

## GitHub 최초 공개

**분리한 github 디렉터리 안에서만** 실행합니다. 상위 연구 저장소를 초기화하거나 push하지 않습니다.

```bash
cd github
git init -b main
git add .
git diff --cached --stat
git commit -m "Release standalone corpus classifier"
gh repo create KETI-NLP/corpus-classifier \
  --public --source=. --remote=origin --push \
  --description "Fast offline multi-label document classification with resumable multi-GPU inference"
git tag v0.1.0
git push origin v0.1.0
```

GitHub 웹을 쓴다면 조직 아래 `corpus-classifier`를 Public으로 만들고, 초기 README/LICENSE/.gitignore 자동 생성을 선택하지 않습니다. 위 로컬 commit을 만든 뒤 웹에 표시되는 remote URL을 `git remote add origin ...`으로 등록하고 `git push -u origin main`을 실행합니다. 라이선스와 README는 이 배포본에 이미 있습니다.

## Hugging Face 최초 공개

아래는 `github/`, `huggingface/`를 포함하는 배포 상위 디렉터리에서 실행합니다.

```bash
hf repos create KETI-NLP/Qwen3.5-2B-CorpusClassifier --repo-type model --public
hf upload KETI-NLP/Qwen3.5-2B-CorpusClassifier ./huggingface . \
  --repo-type model --commit-message "Release frozen 2B corpus classifier"
```

모델을 텍스트 생성 앱/Space로 만들 필요는 없습니다. `Model` 저장소에 파일을 업로드하고 README 상단 YAML에 분류 task와 base model을 표시합니다. custom AutoClass 로딩을 지원하며 직접 사용 시 `trust_remote_code=True`가 필요합니다. joint head 전용 디코딩이 필요하므로 자동 inference widget 동작을 주장하지 않습니다.

업로드한 뒤 Files and versions의 commit hash를 기록하고, 소비자는 `corpus-classifier download --revision <40자리 hash>`로 받도록 안내합니다. 공개 저장소 상태·원격 파일 목록을 확인하고 새 폴더로 다시 내려받아 `corpus-classifier verify`를 실행하세요. `MODEL_MANIFEST.json`은 runtime payload를 검증하며 README/라이선스 등 문서 편집으로 가중치 식별자가 바뀌지 않도록 문서를 제외합니다.

## 공식 참고

- [GitHub 조직 저장소 생성 CLI](https://cli.github.com/manual/gh_repo_create)
- [Hugging Face 조직 접근 권한](https://huggingface.co/docs/hub/organizations-security)
- [Hugging Face 토큰](https://huggingface.co/docs/hub/security-tokens)
- [Hugging Face 업로드](https://huggingface.co/docs/huggingface_hub/guides/upload)
- [모델 카드 메타데이터](https://huggingface.co/docs/hub/model-cards)

이 절차 문서를 작성하거나 로컬 패키지를 검증하는 것만으로 원격 저장소가 생성·업로드되지는 않습니다. 실제 게시 후 원격 URL과 commit을 별도로 기록합니다.

## v0.2.0 업데이트

기존 공개 저장소의 main에 새 commit을 추가하고 `v0.2.0` 태그를 만듭니다. HF의 새 main에서는 중첩된 `model/base`, `model/adapter`를 루트 Transformers payload로 교체합니다. 기존 `v0.1.0` 태그와 commit은 보존하며 force-push하지 않습니다. 원격 HEAD를 확인하고 그 commit을 parent로 지정해 동시 변경을 덮어쓰지 않습니다.

HF 게시 후 새 commit과 manifest SHA256을 `docs/release.json` 및 사용 예제에 기록한 GitHub 코드를 게시합니다. 게시본을 새 폴더로 다운로드하고 표준 AutoClass 로딩, GitHub 로더, 원본 합성 기준 출력 동일성을 검증합니다. 코드 release에는 wheel, source archive, SHA256SUMS를 첨부합니다.
