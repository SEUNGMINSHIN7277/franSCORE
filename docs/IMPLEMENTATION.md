# 구현 내역서 — 개발명세서 → 코드 전수 매핑

> KB 제8회 AI Challenge · FranSCORE
> 개발명세서(FranSCORE_개발명세서.md) 및 기획서 최종본의 **모든 항목**이 어디에, 어떻게
> 구현되었는지의 전수 대조표. 모든 수치는 **공정위 실데이터** 실측값이다(합성 아님).
> 미구현·불가 항목은 근거와 함께 명시한다(§9 한계).

## 0. 우선순위 티어 달성 현황

| 티어 | 모듈 | 상태 | 증빙 |
|---|---|---|---|
| MUST | M1 데이터패널 | ✅ | `outputs/survival_report.csv` — 1,810브랜드/7,556 자격행 |
| MUST | M2 피처/라벨 | ✅ | `outputs/feature_summary.csv`(36피처, 죽은 피처 0), `label_composition.csv`(양성률 9.2%) |
| MUST | M3 GBM+기준모형+지표 | ✅ | `outputs/metrics.csv` — Lift@10 **2.336 > 기준 최고 2.170** |
| MUST | M5 데모화면1 | ✅ | `src/app.py` 화면① — 실데이터 브라우저 렌더링 검증 |
| SHOULD | 확률 보정 | ✅ | isotonic(valid 적합→test 적용) + 동률 tie-break, `calibration.png/csv`, Brier 0.076 |
| SHOULD | SHAP | ✅ | `shap_summary.png`, `shap_values.parquet`(행별 상위10) |
| SHOULD | M4 포트폴리오 집중·상관(원화) | ✅ | `portfolio.csv`, `portfolio_summary.json` |
| SHOULD | M4 LLM 뉴스층 | ✅ | `news_signals.json` + **모의 API 실경로 검증**(`tests/test_llm_paths.py`) |
| COULD | 진짜 RAG | ✅ | `src/rag.py` — 2만 문서 코퍼스·TF-IDF 색인·검색 인용 |
| COULD | 스트레스 시나리오 | ✅ | 위험 상위 10% PD×1.5 동시 악화 |
| COULD | bootstrap | ✅ | `metrics_bootstrap_ci.csv`, `lift_delta_bootstrap.csv`, `walkforward_delta_ci.csv` |
| COULD | 데모화면2 | ✅ | `src/app.py` 화면② |
| **추가** | 워크포워드 확장창 백테스트 | ✅ | `src/backtest.py` — 명세 요구 이상의 검정력 보강 |
| **추가** | 두 표본 트랙 동시 공개 | ✅ | `run_both_tracks.py` → `track_comparison.csv` (체리피킹 방지) |

## 1. 아키텍처 → 파일 매핑 (명세 §1)

| 명세 항목 | 구현 파일 | 핵심 함수 |
|---|---|---|
| 공정위 오픈API 수집·캐시 | `src/collect.py` | `collect_all`, `fetch_service_year` (재시도·백오프·gzip 스냅샷·키 마스킹) |
| 엔티티 정합 | `src/entity.py` | `build_crosswalk`, `build_master` (관리번호 2홉 정합) |
| 패널(brand×year) | `src/panel.py` | `build_panel`, `_merge_region_direct`, `_merge_cancel`, `apply_sample_filter`, `survival_report` |
| 피처 | `src/features.py` | `build_features` (36개 / 5그룹) |
| 라벨(악화 전환) | `src/labels.py` | `build_labels`, `compute_derived_metrics`, `_event_flags` |
| 기준모형 3종 + LightGBM | `src/model.py` | `train_all`, `_baseline_scores`, `fit_lgbm_valid_selected` |
| 지표·보정·SHAP·ablation | `src/evaluate.py` | `evaluate_all`, `_bootstrap_ci`, `_paired_bootstrap` |
| 워크포워드 백테스트 | `src/backtest.py` | `run_walkforward`, `_paired_delta_ci` |
| LLM 뉴스 신호 | `src/news_llm.py` | `fetch_news`, `extract_signals`, `run_news` |
| 포트폴리오 집중·상관 손실 | `src/portfolio.py` | `build_portfolio` |
| LLM 심사메모 | `src/memo_llm.py` | `generate_memo` |
| RAG (COULD) | `src/rag.py` | `build_corpus`, `build_index`, `retrieve_evidence` |
| 대시보드 | `src/app.py` | 화면① 브랜드 상세 / 화면② 포트폴리오 뷰 |
| 오케스트레이션 | `run_pipeline.py`, `run_both_tracks.py` | `--step / --scope / --demo` |
| 파라미터 단일 원천 | `config.yaml` | 라벨·분할·모델 복잡도 후보·시나리오·LLM 전부 |

기술 스택: Python 3.13 · pandas · lightgbm 4.7 · shap 0.52 · scikit-learn 1.6 · streamlit 1.57 ·
anthropic SDK · feedparser · joblib (requirements.txt 버전 고정). 명세는 3.11 기준이나 3.13에서 검증.

## 2. M1 — 데이터 파이프라인 (명세 §2)

### 2.1 수집 — 명세 대상 API 전수 대조

| 명세 지정 | 구현 | 비고 |
|---|---|---|
| 15110241 가맹점 현황 | ✅ 수집 | 패널 본체(가맹점수·신규·계약종료/해지·명의변경·평균매출·면적당매출) |
| 15109828 브랜드 개요 | ✅ 수집 | 업력·연차·임직원수 |
| 15110402 / 15157660 업종 개폐점률 | ✅ 15157660 수집 | 15110402는 **폐기 확인**(data.go.kr 404) → 후속 데이터셋으로 대체 |
| 15143710 브랜드 평균매출·점포수 | ⛔ 제외 | fairdata.go.kr **가명처리·구간값**(브랜드ID 가명, 매출 구간 하/상한) + 공문 승인 필요. 15110241이 **비구간 실값** 제공 → 상위 대체 |
| 15143709 지역별 | ⛔ 대체 | 동일 fairdata 계열. **15125490**(비가명·실값·브랜드×지역)으로 대체 달성 |
| 15125569 정보공개서 목록 | ⛔ 미수집 | **공정위 사이트 별도 키** 필요(data.go.kr 키로 7가지 변형 모두 `INVALID_KEY` 실측). 한계 명시 |
| **추가 15125467** | ✅ 수집 | 브랜드관리번호 — **명세 §2.2 "연결 키" 요건 충족용** |
| **추가 15125490** | ✅ 수집 | 브랜드×연도×지역 가맹점수·직영점수 — **명세 §3.1 직영비중·지역분산 피처 원천** |
| **추가 15125518** | ✅ 수집 | 브랜드 등록취소(자진/직권) — 하드 실패 신호 |

구현 세부: `serviceKey` env 우선(`DATA_GO_KR_KEY`), 원본 JSON을 `data/raw/`에 그대로 보존
(대용량은 gzip — 152만행이 6.7MB), 429/실패 재시도·지수 백오프, 호출 로그, **예외 메시지의 키 마스킹**.

### 2.2 엔티티 정합 — 명세 "브랜드 관리번호 우선" 충족

문제: 실적 API(15110241)에 관리번호가 **없다**(전 필드 실측 확인).
해결: **2홉 정합** — ① 명칭 크로스워크로 관리번호 획득 → ② 관리번호로 연도 간 연결.

| 규칙 | 내용 | 실측 |
|---|---|---|
| R1 정규화 | NFKC → 법인접두 제거 → 특수문자 제거 → 소문자 | — |
| R2 크로스워크 | (브랜드명, 법인명) 완전일치 → 브랜드명 단독 폴백. **마스터 전 연도 합집합** 사용 | 완전일치 98.5% + 폴백 0.6% |
| R3 유일키 | brandMnno 단독은 연도 내 비유일 → **연도 내 법인 충돌 ID만** 본부번호로 분리 | 충돌 ID 분리, 나머지는 연도 연속성 유지 |
| R4 모호 처리 | 후보 2개+ → 관리번호 미부여, 명칭 기반 ID로 **남기고** 로깅(버리면 생존편향) | 1.4% 무효화 |
| R5 제거 | 정규화 후 명칭 공백 → 제거·로깅 | 3행 |

**결과: 관리번호 확보율 97.8%**, 고유 브랜드 20,590개(관리번호 기반 19,662 / 명칭 기반 928),
본부 변경 이력 4,136 브랜드 추적. 리포트: `outputs/entity_resolution_report.csv`.
정합 개선 효과: 명칭 기반 23,220개 → 관리번호 기반 20,590개 (리브랜딩이 올바르게 **연결**됨).

### 2.3 패널

- 패널 연도 = 공시 기준연도 − 1 (수치가 직전 회계연도 실적, `acntgYr`로 교차 확인).
- 매출 0/미기재 → 결측(행 유지). 상하위 1% winsorize(연도별 횡단면).
- **0점포 아티팩트 처리**: 직전연도 10개+ 브랜드가 당해 0으로 보고된 558행은 공시 미기재·재등록
  아티팩트로 판정해 결측 처리(적대적 리뷰 확정 결함 수정 — 412→0→395 같은 원복 사례 다수).
- **직영점수·지역분산 병합**: 15125490의 `areaNm='전체'` 행 → 직영점수(병합률 84.6%),
  17개 시도 행 → 진출지역수·지역HHI·최다지역비중(56.7%, 자격 표본에서는 97.7%).
  ⚠️ '전체'와 지역 행을 합산하면 정확히 2배 중복 → 분리 사용. **독립 API 간 가맹점수 일치율 98.9%**.
- **등록취소 병합**: 390행 매칭(자진 331 / 직권 59).
- **실시간 표본 자격**(적대적 리뷰 수정): `eligible_t` = [t까지 누적 최대 점포수 ≥ 30] AND
  [t에서 끝나는 연속 관측 ≥ 3년] — 과거 정보만 사용.

**✅ M1 DoD:** 생존성 진단표 출력, 라벨 표본 3,641행(수백 건 이상), 양성률 9.2% (목표 10~30%
대역 하단 근접 — 완화 조건 미사용, 확장 트랙은 별도 공개).

## 3. M2 — 피처 & 라벨 (명세 §3)

### 3.1 피처 — 명세 요구 항목 전수 대조 (36개)

| 명세 요구 | 구현 피처 | 상태 |
|---|---|---|
| 가맹점수 변화율·추세·기울기·변동성 | `f_chg_store_growth`, `f_trd_store_growth_{mean,std,slope}` | ✅ |
| 평균매출 변화율·추세 | `f_lvl_avg_sales_log`, `f_chg_sales_growth`, `f_trd_sales_growth_{mean,std,slope}` | ✅ |
| **면적당매출** 변화율·추세 | `f_lvl_sales_per_area_log`, `f_chg_sales_per_area_growth`, `f_trd_sales_per_area_{mean,std,slope}` | ✅ **신규 추가** |
| **직영비중** 및 변화 | `f_lvl_direct_ratio`, `f_struct_direct_ratio_chg`, `f_struct_direct_growth` | ✅ **부활**(과거 100% 결측 → 2.1%) |
| **지역 분산(집중도)** | `f_struct_{n_regions,region_hhi,top_region_share}` + 각 `_chg` + `f_struct_stores_per_region` | ✅ **신규 추가**(과거 완전 누락) |
| 업종 대비 상대위치(분위수) | `f_ind_{store_growth,real_sales_growth,contract_end}_pct` | ✅ |
| 업종 정규화(업종 더미) | `f_ind_major_*` | ✅ |
| 기타 라벨 밖 신호 | `f_chg_new_open_rate`, `f_chg_name_change_rate`, `f_struct_open_close_gap`, `f_lvl_biz_age`, `f_lvl_emp_cnt_log`, `f_lvl_brand_age` | ✅ |
| 뉴스 신호(피처) | ⛔ 의도적 미투입 | 명세 §5.1 "점수에 미투입"이 우선(누출·재현성) |

**죽은 피처 0개** (학습 표본 결측률: 직영비중 2.1%, 지역분산 2.3%, 면적당매출 6.7~19.8%).
⛔ 시점 누출 방지: (brand, t) 피처는 t 이하만 사용 — 자동 검증(t+1 이후 전체 교란 후 bit-exact 불변).

### 3.2 라벨

- 타깃 = "t년 양호 브랜드의 t+1년 악화 전환". 3사건: ①점포 순증감률 하위20%(+|Δ점포|≥3)
  ②실질 평균매출 증가율 하위20% ③계약종료율 상위20% → **2개 이상 또는 1개 극단(5%)**.
- **검증 가능한 양호 요건**(적대적 리뷰 수정): t년 지표 3종이 전부 결측이면 "양호"가 아니라
  "판정 불가"로 표본 제외. 과거에는 27%가 미검증 상태로 포함되어 헤드라인을 왜곡했다.
- 분위수 풀 = 업종그룹×연도 내 **비교 가능 규모 동료군**(전년 점포 ≥10) — 영세 브랜드의
  초변동성이 임계값을 왜곡하지 않도록. 판정 시점 이전 정보만 사용해 실시간 안전.
- 임계값·규칙은 `config.yaml`에 **학습 전 고정** + 최초 git 커밋(`cfcf164`) 증빙.
- 라벨 구성 로깅: `label_composition.csv`(사건별 비율·중복 분포), `label_rate_by_year.csv`.

**✅ M2 DoD:** 양성률 9.2%, 사건 구성 균형(점포 12.2% / 매출 14.9% / 계약종료 10.8% — 쏠림 없음).

## 4. M3 — 모델링 & 검증 (명세 §4)

### 4.1 학습
- **기준모형 3종**: ①persistence(t년 사건 발동수+심도) ②전년 계약종료율 단일변수(업종 백분위)
  ③로지스틱(median impute + 표준화 파이프라인, **train에서만 fit**).
- **주모형 LightGBM**: config 하이퍼파라미터, lr 0.03, `is_unbalance`.
  조기종료는 **valid AUC 기준**(`first_metric_only=True` — 리뷰가 잡은 logloss 조기종료 결함 수정).
  복잡도는 **3개 사전 선언 후보 중 valid AUC로만** 선택(test 미사용) → 그리드서치 아님, 과튜닝 금지 준수.
- **완전 시간분할**: train 2018~2021 / valid 2022 / test 2023, 랜덤분할 없음, seed 42.

### 4.2 평가 — 명세 요구 전항목
| 명세 요구 | 산출물 |
|---|---|
| Lift@10% · Precision@10% | `metrics.csv` |
| PR-AUC · ROC-AUC · Brier | `metrics.csv` |
| 확률 보정 + calibration curve | `calibration.png`, `calibration.csv`, `calibrator.joblib`, 보정 점수 지표 행 추가 |
| 라벨 분해 진단(사건별 recall) | `label_decomposition.csv` |
| 신규 vs 기존 브랜드 분리 | `newbrand_split.csv` |
| 모델 ablation(수준→+변화율→+추세→+업종) | `ablation.csv` |
| 누출 점검 체크리스트 | **`docs/LEAKAGE_CHECKLIST.md`** (전수 점검표, 각 항목 자동테스트/코드 위치 증빙) |
| (추가) bootstrap CI | `metrics_bootstrap_ci.csv`, `lift_delta_bootstrap.csv` |
| (추가) 워크포워드 확장창 | `walkforward_metrics.csv`, `walkforward_predictions.parquet`, `walkforward_delta_ci.csv` |
| (추가) 두 트랙 비교 | `track_comparison.csv` |

**✅ M3 DoD:** `metrics.csv`에 기준모형 3종 vs LightGBM 비교표 생성, LightGBM Lift@10%
**2.336 > 기준모형 최고 2.170** → 통과. 플랜B 불필요(단, 검정력 보강 결과도 함께 공개 — README ②).

보정 관련 수정: isotonic 계단으로 보정확률이 소수 값으로 붕괴해 상위 10% 경계에 대규모 동률이
생기던 결함(리뷰 확정)을 원점수 순위 tie-break로 해소하고, 보정 점수 기준 지표를 metrics.csv에 추가.

## 5. M4 — 신규성/실시간 층 (명세 §5)

### 5.1 LLM 뉴스 신호
- Google News RSS(키 불필요, 발행일 포함) → `data/raw/news/` 스냅샷.
- 엔티티 매칭 필터(일반명사·부분문자 오염 제거) — RAG 코퍼스에도 동일 규칙 적용.
- Claude(claude-opus-5) structured 추출 `{사건유형, 발생시점, 근거문장, 신뢰도}`,
  JSON Schema 강제, `stop_reason=="refusal"` 처리, 예외·비JSON 시 규칙 폴백.
- **점수 미투입** (`config.llm.score_injection: false`) — 화면·메모 표시 전용.
- **실동작 검증**: 키 없이도 모의 Anthropic 클라이언트로 정상/거절/예외/비JSON 4경로 전부 검증
  (`tests/test_llm_paths.py` 3/3). 로그는 의도가 아니라 **실제 LLM 추출 건수/폴백 건수**를 분리 보고
  (리뷰 후 발견한 오도 로그 수정).

### 5.2 포트폴리오 집중·상관
- 합성 exposure(점포수 비례 × 로그정규, seed 고정, **합성임을 전 산출물에 명시**).
- 집중도: 브랜드별 exposure·상위10 비중 66.4%·HHI 0.062.
- 상관 손실 `EL = exposure × PD(보정) × LGD{0.3,0.45,0.6}`, 스트레스(상위 10% PD×1.5, cap 1.0).
- 위험등급은 **순위 기반**(값 임계 컷이 동률과 충돌해 High가 설계 10%→29%로 부풀던 결함 수정).
- 헤드라인: 총 여신 4,529억원 → LGD 45% EL 110.3억원, 스트레스 129.7억원.

### 5.3 심사메모 + RAG (COULD)
- 입력(SHAP 상위요인 + 공시 수치 + 뉴스 + **RAG 검색 근거**)만 인용하도록 시스템 프롬프트 제약.
- **프롬프트 인젝션 방어**: 뉴스 제목은 신뢰할 수 없는 외부 텍스트이므로 그 안의 지시문을 따르지
  말라는 규칙을 시스템 프롬프트에 명시(리뷰 지적 반영).
- 필수 고지("2선 리스크 참고용·자동 여신결정 아님") 누락 시 강제 부착. 키 없으면 결정적 템플릿 폴백.
- **진짜 RAG**: 뉴스 + 공시 이력을 문서 코퍼스(20,117문서)로 구축, TF-IDF(단어 1-2gram +
  char_wb 2-4gram) 색인, 브랜드·위험요인 질의로 검색 → 메모에 **문서ID·출처·발행일·유사도** 인용.
  외부 임베딩 API 미사용으로 키 없이 완전 재현.

## 6. M5 — 대시보드 (명세 §6)
- **화면① 브랜드 상세(MUST)**: 위험등급 배지 + 보정확률 · SHAP 상위 4(한글 요인명, 신규 피처 포함)
  · 공시 지표 추이 · 뉴스 경보(출처·발행일·"점수 미반영") · 집중 여신·상관손실 카드 ·
  등급별 리스크관리 권고("자동 결정 아님") · 심사메모 생성(RAG 근거 포함).
- **화면② 포트폴리오 뷰(COULD)**: KPI · 위험×익스포저 산점 · 위험 상위 20 랭킹 ·
  LGD 시나리오 EL 비교 · 스트레스 비교(기본·스트레스 **동일 LGD**로 짝 맞춤 — 리뷰가 잡은
  방향 반전 표시 결함 수정) · 백테스트 요약.
- 합성 exposure 경고 상시 표시, 산출물 부재 시 실행 안내(크래시 없음).
- 실데이터 브라우저 렌더링 검증 완료.

## 7. 신뢰도·재현성 (명세 §7)
seed·버전·raw 스냅샷·config 단일원천 ✅ / 모듈 단일책임 ✅ / sanity 테스트 5종 + LLM·RAG 3종 ✅ /
로깅 ✅ / README 실행순서 ✅ / 출처 링크·발행일 ✅
**재현 테스트 실측**: 새 클론에서 API 없이 스냅샷만으로 라벨 3,641행·양성률 9.2%·Lift 2.336 동일 재현.

## 8. 일정 (명세 §8)
명세 D1~D5(5일)를 압축 수행. 잔여: PPT·참가신청서 본문·GitHub 업로드·`submit` 태그 freeze.

## 9. 알려진 한계 (심사 정직성)

1. **여신 익스포저는 합성 예시** — 실배포 시 은행 실여신 대체. 전 화면·메모 고지.
2. **브랜드 악화 → 개별 차주 부도 연계 미검증** — KB 내부데이터 PoC 항목(기획서와 동일 범위).
3. **정보공개서 원문(15125569) 미수집** — 공정위 사이트 별도 키 필요(실측 확인). 로드맵.
4. **관리번호 미확보 2.2%** — 명칭 기반 ID 사용, `id_source` 컬럼으로 출처 표기.
5. **LightGBM 우위 범위** — persistence 대비 유의(+0.788, CI [0.413,1.195]), 강한 기준모형
   2종과는 통계적 동등. "압도적 우위"를 주장하지 않으며 부트스트랩 CI를 함께 공개.
6. **확장 표본에서는 LightGBM 열위** — 업종 혼합이 트리 모형에 불리. 숨기지 않고 공개.
7. **2021년 표본 단절** — 공시 등록 급증(직영점 요건 완화). 업종×연도 분위수가 부분 흡수.
8. **LLM 라이브 호출 미실행** — 개발 환경에 API 키 없음. 실제 코드 경로는 모의 API로 검증했고,
   키 설정 시 즉시 동작. 로그·화면에 실제 LLM 사용 건수를 정직 표기.
9. **뉴스 소스 라이선스** — Google News RSS는 개발 보조용. 상용 시 계약 피드 대체 권장.
