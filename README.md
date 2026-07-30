# FranSCORE

**프랜차이즈 브랜드 여신 포트폴리오 집중·상관 리스크 관리 — KB 제8회 AI Challenge 프로토타입**

> 같은 프랜차이즈 브랜드에 묶인 여신은 '브랜드'라는 공통 요인으로 상관된다.
> FranSCORE는 공정위 공개 공시로 **브랜드의 구조악화 전환을 예측(LightGBM)·설명(SHAP)** 하고,
> 이를 은행의 **브랜드별 여신 집중도와 결합해 상관 손실 위험을 원화로 정량화**하는 2선 리스크 관리 도구다.

## 핵심 결과 (실데이터 백테스트)

- **데이터:** 공정거래위원회 가맹정보 오픈API 실데이터 — 브랜드×연도 76,776행 (실적연도 2016~2024, 23,220개 브랜드)
- **표본:** 외식업 · 가맹점 30개+ · 3년 연속 관측 → 1,820개 브랜드 / 11,940행, 라벨 표본 5,734행 (양성률 14.5%)
- **검증:** 완전 시간분할 (train ≤2021 / valid 2022 / test 2023), seed 고정, 라벨 규칙 학습 전 git 커밋으로 동결

| 모형 (test 2023, n=932, base rate 10.6%) | Lift@10% | Precision@10% | ROC-AUC | Brier |
|---|---|---|---|---|
| ① 전년 상태 유지 (persistence) | 1.01 | 0.108 | 0.597 | 0.106 |
| ② 계약종료율 단일변수 | 1.62 | 0.172 | 0.634 | 0.209 |
| ③ 로지스틱 회귀 | 1.92 | 0.204 | **0.741** | 0.149 |
| **LightGBM (주모형)** | **2.43** | **0.258** | 0.705 | **0.097** |

→ 위험 상위 10% 지목 시 무작위 대비 **2.4배**, 최강 기준모형(로지스틱) 대비 **+26%** 정밀도.
(정직성 노트: ROC-AUC는 로지스틱이 소폭 우세 — 업무 지표인 Lift/Precision@10%와 확률 정확도(Brier)에서 LightGBM이 우위. 지표 전부를 `outputs/metrics.csv`에 공개한다.)

## 빠른 시작 (재현 절차)

```bash
git clone <repo> franscore && cd franscore
pip install -r requirements.txt          # 버전 고정 (Python 3.13 기준)

# 전체 파이프라인 (수집→패널→피처→라벨→학습→평가→포트폴리오→뉴스)
python run_pipeline.py --step all

# 대시보드
streamlit run src/app.py

# sanity 테스트 (라벨규칙·시점누출·시간분할·스키마·포트폴리오)
python -m tests.test_sanity
```

- `data/raw/` 스냅샷이 저장소에 포함되어 있어 **API 키 없이도 수집 단계를 건너뛰고 전체 지표가 재현**된다
  (수집 재실행 시에만 `DATA_GO_KR_KEY` 필요 — data.go.kr 개발계정 자동승인).
- LLM 기능(뉴스 신호 구조화·심사메모)은 `ANTHROPIC_API_KEY` 설정 시 Claude(claude-opus-5)로 동작하고,
  없으면 규칙기반 폴백으로 자동 전환된다(화면에 `llm_used=false` 명시). 뉴스 신호는 **모델 점수에 미투입**.
- 합성 데이터로 배선만 점검하려면: `python run_pipeline.py --step all --demo` (산출물 `outputs/_smoke/` 격리)

## 디렉토리

```
franscore/
├─ config.yaml          # 모든 파라미터 (라벨 임계값·분할·모델·시나리오) — 학습 전 동결
├─ run_pipeline.py      # 파이프라인 러너 (--step collect|panel|features|labels|model|evaluate|portfolio|news|all)
├─ data/raw/            # 공정위 API 원본 스냅샷 (재현성 — 수정 금지)
├─ data/processed/      # 패널·피처·라벨·예측 parquet
├─ outputs/             # 지표표·보정곡선·SHAP·포트폴리오·뉴스신호 (제출 근거)
├─ src/
│  ├─ collect.py entity.py panel.py       # M1 수집·정합·패널
│  ├─ features.py labels.py               # M2 피처·라벨 (시점누출 자동검증)
│  ├─ model.py evaluate.py                # M3 학습·기준모형 3종·지표·보정·SHAP
│  ├─ news_llm.py portfolio.py memo_llm.py # M4 뉴스신호·원화손실·심사메모
│  └─ app.py                              # M5 Streamlit 대시보드 (2화면)
├─ tests/test_sanity.py # sanity 스위트 (5종)
└─ docs/                # INTERFACES(모듈 계약) · IMPLEMENTATION(명세→코드 매핑) · AI_USAGE
```

## 데이터 출처 (공개·무료·개인정보 0)

| 소스 | 내용 | 용도 |
|---|---|---|
| 공정위 15110241 (FftcBrandFrcsStatsService) | 브랜드×연도 가맹점수·신규등록·계약종료/해지·명의변경·평균매출 | 패널 본체 |
| 공정위 15109828 (FftcBrandBrandStatsService) | 가맹사업개시일·임직원수 | 업력·본부규모 피처 |
| 공정위 15157660 (FftcIndutyFrcsCntOpclStatsService) | 업종×연도 개폐점률 | 업종 기준선 |
| Google News RSS | 브랜드 뉴스 (제목·출처·발행일) | LLM 실시간 신호 (점수 미투입) |

주의: 공시 기준연도(yr)의 수치는 직전 회계연도 실적이므로 **패널 연도 = yr − 1**로 정렬했다.

## 정직성·재현성 장치

1. **시점 누출 금지:** 연도 t 피처는 t 이하 데이터만 사용. 윈저라이즈·업종 분위수도 연도 내 횡단면으로만 계산.
   `tests/test_sanity.py`가 t+1 전체 교란 후 피처 불변(bit-exact)을 자동 검증.
2. **라벨 동결:** 악화 전환 라벨 규칙·임계값은 `config.yaml`에 학습 전 고정, 최초 git 커밋으로 증빙.
3. **보정:** isotonic 보정기는 valid에서만 적합 → test에 적용 (`outputs/calibration.png`).
4. **합성 명시:** 여신 익스포저는 방법론 실증용 합성 예시 — 모든 화면·산출물에 상시 표기. 실배포는 은행 실여신 데이터로 대체.
5. **한계 명시:** 브랜드 악화→개별 차주 부도 연계는 본 프로토타입 범위 밖 (KB 내부데이터 PoC 항목).

## 라이선스·이용 고지

공정위 오픈API 데이터는 공공데이터로 자유 이용 가능. Google News RSS는 개발 단계 보조 신호로만 사용했으며
상용 배포 시 뉴스 소스는 계약 기반 피드(예: 네이버 오픈API)로 대체를 권장한다.
