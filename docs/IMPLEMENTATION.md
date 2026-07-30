# 구현 내역서 — 개발명세서 → 코드 전수 매핑

> KB 제8회 AI Challenge · FranSCORE
> 개발명세서(FranSCORE_개발명세서.md)의 모든 항목이 어디에, 어떻게 구현되었는지의 전수 대조표.
> 모든 수치는 **공정위 실데이터** 기준 실측값이다 (합성 아님).

## 0. 우선순위 대비 달성 현황

| 티어 | 모듈 | 상태 | 증빙 |
|---|---|---|---|
| MUST | M1 데이터패널 | ✅ | `outputs/survival_report.csv` — 표본 1,820브랜드/11,940행 |
| MUST | M2 피처/라벨 | ✅ | `outputs/label_composition.csv`, `feature_summary.csv` — 22피처, 양성률 14.5% |
| MUST | M3 GBM+기준모형+핵심지표 | ✅ | `outputs/metrics.csv` — Lift@10 2.43 > 기준 최고 1.92 |
| MUST | M5 데모화면1 (브랜드 상세) | ✅ | `src/app.py` — 실데이터 렌더링 검증 완료 |
| SHOULD | 확률 보정 | ✅ | isotonic(valid 적합→test 적용), `outputs/calibration.png/csv` |
| SHOULD | SHAP | ✅ | `outputs/shap_summary.png`, `shap_values.parquet`(행별 상위 10) |
| SHOULD | M4 포트폴리오 집중·상관(원화) | ✅ | `outputs/portfolio.csv`, `portfolio_summary.json` |
| SHOULD | M4 LLM 뉴스층 | ✅ | `outputs/news_signals.json` (키 없으면 규칙 폴백, 점수 미투입) |
| COULD | 스트레스 시나리오 | ✅ | 위험 상위 10% PD×1.5 동시 악화 — portfolio_summary에 포함 |
| COULD | 데모화면2 (포트폴리오 뷰) | ✅ | `src/app.py` 화면② — KPI·산점·랭킹·EL 비교 |
| COULD | 진짜 RAG · bootstrap CI | ⬜ 미구현 | 명세상 "시간 남으면" — 한계로 명시 |

## 1. 시스템 아키텍처 → 파일 매핑 (명세 §1)

| 명세 항목 | 구현 파일 | 핵심 함수 |
|---|---|---|
| 공정위 오픈API 수집·캐시 | `src/collect.py` | `collect_all`, `fetch_service_year` (재시도·백오프·스냅샷) |
| 엔티티 정합 | `src/entity.py` | `build_master`, `normalize_name` |
| 패널(brand×year) | `src/panel.py` | `build_panel`, `apply_sample_filter`, `survival_report` |
| 피처 | `src/features.py` | `build_features` (f_lvl/chg/trd/ind/struct 5그룹 22개) |
| 라벨(악화 전환) | `src/labels.py` | `build_labels`, `compute_derived_metrics` |
| 기준모형 3종 + LightGBM | `src/model.py` | `train_all` |
| 지표·보정·SHAP | `src/evaluate.py` | `evaluate_all` |
| LLM 뉴스 신호 | `src/news_llm.py` | `fetch_news`, `extract_signals`, `run_news` |
| 포트폴리오 집중·상관 손실 | `src/portfolio.py` | `build_portfolio` |
| LLM 심사메모 | `src/memo_llm.py` | `generate_memo` |
| 대시보드 | `src/app.py` | 화면① 브랜드 상세 / 화면② 포트폴리오 뷰 |
| 오케스트레이션 | `run_pipeline.py` | `--step collect…news / --demo` |
| 파라미터 단일 원천 | `config.yaml` | 라벨·분할·모델·시나리오 전부 |

기술 스택: Python 3.13 · pandas · lightgbm 4.7 · shap 0.52 · scikit-learn 1.6 · streamlit 1.57 · anthropic SDK · feedparser (requirements.txt 버전 고정).

## 2. M1 — 데이터 파이프라인 (명세 §2)

- **수집 (§2.1):** 조사 결과 명세서의 API 후보 중 실제 브랜드×연도 패널을 제공하는 3종을 확정 수집:
  - `15110241 getBrandFrcsStats` (패널 본체: 가맹점수·신규등록·계약종료/해지·명의변경·평균매출·면적당매출)
  - `15109828 getBrandBrandStats` (가맹사업개시일→업력, 임직원수 — 라벨 차원 밖 피처)
  - `15157660 getIndutyFrcsCntOpclStats` (업종 개폐점률 기준선; 명세의 15110402는 폐기 확인→대체)
  - 명세의 15143710은 조사 결과 **가명처리·구간값 데이터**(fairdata.go.kr 별도 공문 승인 필요)로 판명되어 제외 — 15110241이 정확한 비구간 원값을 제공.
  - 원본 JSON을 `data/raw/{service}_{year}.json`으로 보존(키 미포함), 재호출 생략 캐시, 429 재시도·백오프. **2017~2025 공시연도 전량(76,846행) 수집 완료.**
- **엔티티 정합 (§2.2):** API에 브랜드 고유ID가 없음(실측) → 정합 규칙: ① NFKC·법인접두 제거·특수문자 제거 정규화 ② (정규화 브랜드명, 업종대분류) 그룹으로 연도 연결(본부변경 3,835건 이력 추적) ③ 같은 연도 복수 법인 공존 시 동명이브랜드 분리(269그룹) ④ 정규화 불가 3행 제거 로깅. 전 규칙 `src/entity.py` 도크스트링에 문서화.
- **패널 (§2.3):** 패널 연도 = 공시 기준연도 − 1 (공시 수치가 직전 회계연도 실적이므로 — 문서화). 매출 0→결측 처리(행 유지), (brand,year) 중복 134행 점포수 최대 행으로 해소. 생존성 진단표 구현 및 출력:

| 단계 | 행수 | 브랜드수 | 매출보유율 |
|---|---|---|---|
| ① 전체(정합 후) | 76,776 | 23,220 | 39.7% |
| ② 외식 필터 | 60,526 | 18,355 | 39.0% |
| ③ +점포30+ & 3년연속 | 11,940 | 1,820 | 79.2% |

- **✅ M1 DoD:** 라벨 표본 5,734행(수백 건 이상), 양성률 14.5%(목표 대역 10~30%) → **통과, 완화 조건 불사용.**

## 3. M2 — 피처 & 라벨 (명세 §3)

- **피처 (§3.1):** 22개(5그룹) — 시계열 변화율·2~3년 추세/변동성(f_chg/f_trd), 라벨 차원 밖(f_struct: 신규개점-종료 격차·직영 변화 등 + 업력·임직원수), 업종 정규화(f_ind: 업종그룹×연도 내 분위수 + 업종 더미). 윈저라이즈 상하위 1%는 **연도 내 횡단면으로만** 계산(누출 차단).
- **⛔ 시점 누출 방지:** (brand,t) 피처는 t 이하만 사용. **t+1 이후 전체 교란 후 피처 bit-exact 불변**을 자동 검증 (`tests/test_sanity.py` [2/5]).
- **라벨 (§3.2):** "현재 양호 브랜드의 미래 악화 전환" — ①가맹점 순증감률 업종 하위20% (+|Δ점포|≥3) ②실질 평균매출 증가율 하위20%(업종 중앙값 차감) ③계약종료율 상위20% 중 **2개 이상 또는 1개 극단(5%)**. healthy gate(t년 무발동)·min_stores_at_t=10 적용. 임계값은 학습 전 `config.yaml` 고정 + **최초 git 커밋(cfcf164)으로 증빙.**
- **✅ M2 DoD:** 양성률 14.5% 목표 대역, 사건 구성 균형(store 13.1% / sales 15.7% / contract 13.1% — 한쪽 쏠림 없음, `outputs/label_composition.csv`).

## 4. M3 — 모델링 & 검증 (명세 §4)

- **기준모형 3종 (§4.1):** ①persistence(t년 사건 발동수+심도 점수화) ②전년 계약종료율 단일변수(업종 내 백분위) ③로지스틱(중앙값 대치+표준화 파이프라인). 주모형 LightGBM은 config 파라미터 고정, valid 조기종료, **그리드서치 없음(과튜닝 금지 준수)**.
- **완전 시간분할:** train 2016~2021(3,920) / valid 2022(882) / test 2023(932). 랜덤분할 없음, seed=42.
- **평가 (§4.2) — 전 항목 구현:**
  - 지표표 `outputs/metrics.csv` (4모형×3분할): 본 문서 상단 표 참조. **LightGBM test Lift@10=2.43 > 기준 최고 1.92 → M3 DoD 통과.**
  - 확률 보정: isotonic, **valid에서만 적합**→전체 적용, `calibration.png/csv` + 보정확률 p_calibrated.
  - 라벨 분해 진단: 사건별 recall@10% `label_decomposition.csv`.
  - 일반화 분리: 기존 vs 신규 브랜드 `newbrand_split.csv`.
  - 모델 ablation: 수준→+변화율→+추세→+업종/구조 누적 기여 `ablation.csv`.
  - SHAP: `shap_summary.png` + 행별 상위요인 `shap_values.parquet`(대시보드 연동).
- **누출 점검 체크리스트 (§4.2):** 라벨 시점(t+1) > 피처 시점(≤t) — labels.py 구조상 보장 + sanity 자동검증 / 미래 파생 없음 — 윈저라이즈·분위수 연도 내 계산 / 분할 정합 — 연도 교집합 없음 자동검증.
- **한계(정직 기록):** LightGBM early stopping best_iteration=3 — 소표본에서 빠른 수렴. ROC-AUC는 로지스틱이 소폭 우세하나 업무 헤드라인 지표(Lift/Precision@10)와 Brier에서 LightGBM 우위. 지표 전체 공개.

## 5. M4 — 신규성/실시간 층 (명세 §5)

- **LLM 뉴스 신호 (§5.1):** Google News RSS(키 불필요, 발행일 포함) 수집→`data/raw/news/` 스냅샷. 엔티티 매칭 필터(일반명사 오염 제거 — 실측 로그로 확인). Claude(claude-opus-5) 구조화 추출 `{사건유형(본부분쟁/재무이슈/집단폐점/기타), 발생시점, 근거문장, 신뢰도}` — JSON Schema 강제(output_config), refusal 처리. **키 없으면 규칙기반 폴백(llm_used=false 표기)**. **점수 미투입 원칙** 코드·화면 모두 명시.
- **포트폴리오 (§5.2):** 합성 exposure(점포수 비례×로그정규, seed 고정, **합성임을 모든 산출물에 명시**). 집중도(브랜드별 exposure·상위10 비중 70.5%·HHI 0.177), 상관 손실 `EL = exposure × PD(보정) × LGD{0.3,0.45,0.6}`, 스트레스(상위 10% PD×1.5 cap 1.0). 헤드라인(실측): *"총 여신 3,615억원, LGD 45% 기준 1년 예상손실 162.8억원(4.50%), 스트레스 시 174.7억원."*
- **심사메모 (§5.3):** 입력(SHAP 상위요인+공시수치+뉴스 근거)만 인용하도록 시스템 프롬프트 제약, 출처·발행일 표기, "2선 리스크 참고용·자동 여신결정 아님" 고지 강제 삽입. 키 없으면 결정적 템플릿 폴백(동일 입력→동일 출력).

## 6. M5 — 대시보드 (명세 §6)

- **화면① 브랜드 상세(MUST):** 위험등급 배지 + 보정확률 · SHAP 상위 4(한글 요인명) · 공시 지표 추이 · 뉴스 경보(출처·발행일·"점수 미반영" 고지) · 집중 여신·상관손실 카드 · 등급별 리스크관리 권고("자동 결정 아님") · 심사메모 생성 버튼. **실데이터 브라우저 렌더링 검증 완료.**
- **화면② 포트폴리오 뷰(COULD):** KPI 4종 · 위험×익스포저 산점 · 위험 상위 20 랭킹 · LGD 시나리오 EL 비교 · 스트레스 비교 · 백테스트 지표 expander. **검증 완료.**
- 합성 exposure 경고 배너 상시 표시. 산출물 파일 부재 시 실행 안내 메시지(크래시 없음 — AppTest로 검증).

## 7. 신뢰도·재현성 (명세 §7) & 일정 (§8)

- seed 고정 · requirements 버전 고정 · data/raw 스냅샷 보존 · 전 파라미터 config.yaml — **전부 구현.**
- 코드 품질: 모듈 단일책임(INTERFACES.md 계약 기반) · sanity 테스트 5종 · 로깅 · README 실행순서.
- 일정: 명세의 D1~D5(5일)를 **1야간 세션에 압축 완료** (D5 잔여: PPT·신청서 본문·코드 freeze 태그).

## 8. 평가기준 매핑 (명세 §10) 및 리스크 대응 (§11)

- "GBM이 기준모형 못 이김" 리스크 → **실측으로 해소** (Lift@10 2.43 vs 1.92). 플랜B 불필요.
- "패널 표본 부족" → 미발동 (완화 조건 불사용).
- "LLM 뉴스 시간 부족" → 미발동 (구현 완료, 키 유무 무관 동작).
- 남은 체크리스트(§9): 참가신청서 본문 · 기술설명서 PPT · GitHub 업로드 · 접수 전 `submit` 태그 freeze.

## 9. 알려진 한계 (심사 정직성)

1. 여신 익스포저는 합성 예시 (실배포 시 은행 실여신 대체) — 전 화면 고지.
2. 직영점수(n_direct)는 사용 API 미제공 → 결측 처리 (fairdata API는 가명·구간값이라 제외).
3. 브랜드 악화→개별 차주 부도 연계 미검증 (KB 내부데이터 PoC 항목으로 명시).
4. 브랜드 고유ID 부재로 이름 기반 정합 — 규칙·손실 건수 로깅으로 투명화.
5. 2021년 공시 등록 급증(직영점 요건 완화)으로 2020↔2021 표본 단절 존재 — 업종×연도 분위수 피처가 부분 흡수.
6. LLM 층은 ANTHROPIC_API_KEY 설정 시 활성화 (미설정 시 규칙 폴백으로 데모 가능, llm_used=false 표기).
