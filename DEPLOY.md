# 배포 안내

KB 심사역이 브라우저로 접속해 바로 쓸 수 있도록 웹에 올리는 절차입니다.
저장소는 이미 배포 가능한 상태로 준비돼 있습니다 — **새로 clone 한 상태에서 API 키
없이도 5개 화면이 모두 정상 동작하는 것을 실측 확인**했습니다.

---

## 1. Streamlit Community Cloud (권장 · 무료)

가장 빠릅니다. GitHub 계정으로 로그인하면 3분 안에 URL이 나옵니다.

### 1-1. 배포

아래 주소를 열면 저장소·브랜치·진입점이 미리 채워진 배포 화면이 뜹니다.

```
https://share.streamlit.io/deploy?repository=SEUNGMINSHIN7277%2FfranSCORE&branch=main&mainModule=src%2Fapp.py
```

화면에서 확인할 값:

| 항목 | 값 |
|---|---|
| Repository | `SEUNGMINSHIN7277/franSCORE` |
| Branch | `main` |
| Main file path | `src/app.py` |
| Python version | `3.12` 또는 `3.13` (Advanced settings) |
| App URL | 원하는 주소 (예: `franscore`) |

`Deploy!` 를 누르면 의존성 설치에 3~5분 걸립니다.
완료되면 `https://<정한이름>.streamlit.app` 으로 누구나 접속할 수 있습니다.

### 1-2. API 키 등록 (선택)

**키가 하나도 없어도 앱은 정상 동작합니다.** 새로 clone 한 상태에서 키 없이 5개
화면이 전부 뜨는 것을 실측 확인했습니다. 이미 수집해 둔 산출물(점수·진단 소견·
로고 269건·검색수요 292건)이 저장소에 들어 있기 때문입니다.

키는 **다시 수집하거나 AI 상담 답변을 생성할 때만** 필요합니다.

| 키 | 없을 때 화면 | 발급처 |
|---|---|---|
| `GEMINI_API_KEY` | AI 상담이 답변을 만들지 않고 **수집된 사실만 정리**해 보여줍니다 | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — `AIza…` 형식 |
| `NCP_API_KEY_ID`<br>`NCP_API_KEY` | 이미 수집된 292개 브랜드의 검색수요는 **그대로 보입니다**. 갱신만 안 됩니다 | NCP 콘솔 → NAVER API HUB → Application |
| `DART_API_KEY` | 이미 수집된 본부 재무는 그대로 보입니다. 갱신만 안 됩니다 | [opendart.fss.or.kr](https://opendart.fss.or.kr) |

등록: 배포된 앱 → 우측 하단 `Manage app` → `Settings` → `Secrets` 에 붙여넣습니다.

```toml
GEMINI_API_KEY = "여기에-키"
NCP_API_KEY_ID = "여기에-아이디"
NCP_API_KEY = "여기에-시크릿"
DART_API_KEY = "여기에-키"
```

저장하면 앱이 자동 재시작합니다. `src/common.load_secrets()` 가 이 값을 환경변수로
올리므로 코드 수정은 필요 없습니다.

> ⚠️ **키를 저장소에 커밋하지 마십시오.** `.env` 는 `.gitignore` 에 있고, 배포는
> 플랫폼 Secrets 로만 전달합니다. 이 대화에 붙여넣은 키들은 제출 전 재발급을 권합니다.

### 1-3. 자원 한도

| 항목 | 실측값 | Streamlit Cloud 한도 |
|---|---|---|
| 저장소 크기 | 251MB (clone 기준) | 제한 없음 (파일당 100MB) |
| 최대 파일 | `rag_index.joblib` 64MB | 100MB |
| RAG 색인 상주 메모리 | 236MB | — |
| 앱 전체 상주 (추정) | 약 500MB | 약 2.7GB |

첫 배포는 저장소 clone + 의존성 설치로 5~8분 걸립니다. 이후 재시작은 빠릅니다.

RAG 색인은 배포를 감안해 `float32` + 문자 n-gram 상한 40,000 으로 줄였습니다
(91MB → 67MB, 상주 263MB → 236MB). 줄이기 전후 검색 결과를 실측 비교해
**상위 5건 겹침 95% · 1위 일치 11/12** 로 품질이 유지되는 것을 확인했습니다
(`python tools/check_rag_quality.py --compare`).

---

## 2. Hugging Face Spaces (대안)

메모리 여유가 더 필요하거나 저장소를 비공개로 두고 싶을 때 씁니다.

1. https://huggingface.co/new-space → SDK `Streamlit`, Hardware `CPU basic (무료)`
2. 생성된 Space 저장소에 이 저장소 내용을 push
3. `app_file: src/app.py` 를 `README.md` 프런트매터에 지정
4. Settings → `Variables and secrets` 에 위 키들을 등록

67MB 색인 파일은 Git LFS 로 올리는 편이 안정적입니다.

```bash
git lfs install
git lfs track "outputs/rag_index.joblib"
```

---

## 3. 사내 서버 (KB 내부망 가정)

```bash
pip install -r requirements.txt
streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0
```

내부망 배포 시에는 `config.yaml` 의 `portfolio.exposure_source` 에 실제 여신 CSV
경로를 넣으십시오. 그러면 화면의 익스포저가 **추정값이 아니라 실제 여신 잔액**으로
바뀌고, 고지 문구도 자동으로 그에 맞게 바뀝니다 (필요 컬럼: `brand_id` 또는
`brand_name`, `exposure_mkrw`).

---

## 4. 데이터 갱신

공정거래위원회 공시는 매년 갱신됩니다. 새 연도 공시가 나오면:

```bash
python run_pipeline.py --step collect     # 공시 원본 수집 (DATA_GO_KR_KEY 필요)
python run_pipeline.py --step panel
python run_pipeline.py --step dart        # 가맹본부 재무 (DART_API_KEY 필요)
python run_pipeline.py --step features
python run_pipeline.py --step labels
python run_pipeline.py --step model
python run_pipeline.py --step score
python run_pipeline.py --step demand      # 검색수요 + 브랜드 로고 (NCP 키 필요, 선택)
python run_pipeline.py --step news        # 뉴스 (선택)
python run_pipeline.py --step diagnose    # 브랜드별 진단 소견
python -m src.rag                         # RAG 색인 재구축
```

또는 한 번에 `python run_pipeline.py --step all`.

갱신된 산출물을 커밋해 push 하면 배포된 앱이 자동으로 다시 뜹니다.

---

## 5. 배포 전 점검

```bash
python -m ruff check .
python tests/test_sanity.py
python tests/test_diagnosis.py
python tests/test_llm_paths.py
python tools/check_doc_numbers.py
```

전부 통과해야 합니다. 마지막 확인 시점 기준: ruff 통과 · sanity 6/6 ·
diagnosis 8/8 · LLM/RAG 4/4 · 문서 수치 88건 일치.

**새 clone 재현 확인** — 배포 환경과 같은 조건(키 없음)을 흉내 냅니다.

```bash
git clone https://github.com/SEUNGMINSHIN7277/franSCORE.git /tmp/franscore-check
cd /tmp/franscore-check && pip install -r requirements.txt
streamlit run src/app.py --server.port 8788
```

5개 화면이 모두 뜨고 오류가 없어야 합니다.

---

## 6. 네이버 API 자격증명 얻는 곳 (혼동하기 쉬움)

네이버가 개발자센터 오픈API를 **NAVER API HUB**(네이버 클라우드)로 이관했습니다.
둘은 자격증명도 엔드포인트도 다릅니다.

| | 구 개발자센터 | **NAVER API HUB (현재 사용)** |
|---|---|---|
| 콘솔 | developers.naver.com | ncloud.com → NAVER API HUB |
| 자격증명 | Client ID / Secret | Client ID / Secret (`X-NCP-APIGW-…`) |
| 검색 API | 가능 | 가능 |
| **검색어트렌드** | **불가** (Scope Status Invalid) | **가능** |

Client ID 는 **10자 안팎**, Client Secret 은 **40자**입니다. 둘의 길이가 크게
다르므로 헷갈리면 길이로 구분하십시오.

경로: NCP 콘솔 → `NAVER API HUB` → `Application` → 앱 선택 → `인증 정보`
