# FranSCORE 모듈 인터페이스 계약 (구현 규약)

> 모든 모듈은 이 계약을 따른다. 계약 변경 시 이 문서를 먼저 수정한다.
> 원칙: 단일책임 · 시점누출 금지 · 모든 파라미터는 config.yaml · seed 고정.

## 0. 공통 (src/common.py — 이미 구현됨, 수정 금지)

```python
from src.common import load_config, get_logger, PATHS, set_seed, make_synthetic_panel
cfg = load_config()          # config.yaml → dict. cfg["_root"]에 프로젝트 루트 Path
log = get_logger("module")   # 표준 로거 (stdout + outputs/pipeline.log)
set_seed(cfg["seed"])        # random/numpy 시드 고정
panel = make_synthetic_panel(cfg)  # ⚠️ 스모크테스트 전용 합성 패널 (아래 §1 스키마 준수)
```

- 실행은 항상 **프로젝트 루트에서** `python -m src.모듈` 또는 `python run_pipeline.py`.
- 금액 단위: `avg_sales`·`avg_sales_per_area`는 **천원**(원천 그대로), 포트폴리오 exposure는 **백만원**, 화면 표시는 **억원**(=백만원/100).

## 1. 패널 스키마 (data/processed/panel.parquet) — 모든 모듈의 공용 입력

brand_id × year 롱포맷. 1행 = 한 브랜드의 한 해.

| 컬럼 | 타입 | 의미 |
|---|---|---|
| brand_id | str | 브랜드 관리번호(FTC). 없으면 정규화 명칭 키 |
| brand_name | str | 브랜드명 |
| company_name | str | 가맹본부(법인)명 |
| industry_major | str | 업종 대분류 (외식/서비스/도소매) |
| industry_mid | str | 업종 중분류 (치킨/커피/한식/…) |
| year | int | 실적 기준 연도 |
| n_stores | float | 가맹점 수 (연말) |
| n_direct | float | 직영점 수 |
| n_new | float | 신규개점(등록) 수 (해당 연도 중) |
| n_contract_end | float | 계약종료 수 |
| n_contract_cancel | float | 계약해지 수 |
| n_name_change | float | 명의변경 수 |
| avg_sales | float | 가맹점 평균매출액 (천원, 결측 NaN) |
| avg_sales_per_area | float | 3.3㎡당 평균매출액 (천원, 결측 NaN) |

정렬: (brand_id, year). 중복 (brand_id, year) 없음 보장.

### 파생 지표 정의 (여러 모듈 공용 — 반드시 이 식 사용)
- `store_growth_rate(t)` = (n_stores_t − n_stores_{t−1}) / n_stores_{t−1}
- `contract_end_rate(t)` = (n_contract_end_t + n_contract_cancel_t) / n_stores_{t−1}
- `sales_growth(t)` = avg_sales_t / avg_sales_{t−1} − 1
- `real_sales_growth(t)` = sales_growth(t) − (업종그룹 내 sales_growth(t) 중앙값)
- `direct_ratio(t)` = n_direct_t / (n_direct_t + n_stores_t)
- 업종그룹 = industry_mid (해당 연도 그룹 크기 ≥ 30) else industry_major

## 2. M2 — features.py / labels.py

### labels.build_labels(panel: DataFrame, cfg) -> DataFrame
반환: `[brand_id, year, label, healthy_at_t, ev_store, ev_sales, ev_contract, n_events, extreme_fired]`
- **의미: `label=1` ⇔ t년에 양호했던 브랜드가 t+1년에 '악화 전환'.** 행의 year는 **t** (피처 시점).
- 악화 사건 3종(t+1 지표, 업종그룹×연도 내 분위수, config `label.events`):
  ① store_growth_rate(t+1) 하위 20% **그리고** |Δ점포| ≥ `min_abs_store_change`
  ② real_sales_growth(t+1) 하위 20%
  ③ contract_end_rate(t+1) 상위 20%
- 발동: **2개 이상** or **1개가 극단**(하위/상위 5% = `extreme_quantile`).
- **healthy_at_t**: t년(즉 t−1→t 지표)에 위 사건이 하나도 발동 안 함. `healthy_gate_at_t=true`면 label 표본은 healthy_at_t=True인 행만 (전환 예측).
- 게이트: n_stores_t ≥ `min_stores_at_t`. t+1 데이터 없는 행은 label=NaN(반환에서 제외).
- 분위수 임계값은 사건 지표의 **t+1년 실측 분포**로 계산(라벨 정의이므로 누출 아님 — 피처가 아님).
- 로깅: 사건별 발동 비율, 중복 분포(1개/2개/3개), 연도별 양성률 → `outputs/label_composition.csv`.

### features.build_features(panel: DataFrame, cfg) -> DataFrame
반환: `[brand_id, year] + f_* 컬럼들`. 행 (brand_id, t)의 피처는 **t 이하 연도 값만** 사용(⛔ t+1 절대 금지).
필수 피처 그룹(접두어):
- `f_lvl_*` 수준: n_stores, avg_sales(log1p), direct_ratio, brand_age(관측연차)
- `f_chg_*` 변화율: store_growth_rate(t), sales_growth(t), contract_end_rate(t), 신규개점률(n_new/n_stores_{t-1}), 명의변경률
- `f_trd_*` 추세: 최근 2~3년 store_growth·sales_growth 평균/기울기/표준편차 (관측 부족 시 NaN 허용 — LightGBM은 NaN 처리 가능, 로지스틱용은 model.py에서 median impute)
- `f_ind_*` 업종 상대위치: store_growth_rate(t)·real_sales_growth(t)·contract_end_rate(t)의 업종그룹×연도 내 분위수(rank pct), 업종 더미(major)
- `f_struct_*` 라벨 차원 밖: direct_ratio 변화, 직영점수 변화율, 신규개점-종료 격차
윈저라이즈: 연속 피처 상하위 `sample.winsorize_pct` 클립. 피처 목록·결측률 → `outputs/feature_summary.csv`.

## 3. M3 — model.py / evaluate.py

### model.train_all(features, labels, cfg) -> dict
- 표본 = features ⨝ labels (inner, on brand_id+year).
- **완전 시간분할**(`split.mode=auto_last_two`): 라벨 최대 연도 T → test={T}, valid={T−1}, train={≤T−2}. 결정된 연도를 `outputs/split_years.json`에 기록.
- 기준모형 3종 (예측확률 컬럼명 고정):
  - `p_persistence`: 전년(t−1→t) 사건 발동 여부(비양호도)를 그대로 위험 점수화 — healthy gate 표본에선 t년 사건 지표 3종 중 발동 개수/심도 기반 점수 (0~1 스케일)
  - `p_single`: 전년 계약종료율 contract_end_rate(t) 단일변수 (업종그룹 내 rank pct)
  - `p_logistic`: 로지스틱 회귀 (median impute + 표준화 파이프라인)
- 주모형 `p_lgbm`: LightGBM(config 하이퍼파라미터, valid로 early stopping, **과튜닝 금지** — 그리드서치 안 함).
- 반환/저장: `data/processed/predictions.parquet` `[brand_id, year, split, y_true, p_lgbm, p_persistence, p_single, p_logistic, is_new_brand]` (is_new_brand: train 연도에 등장 안 한 brand_id). 모델은 `outputs/model_lgbm.txt`(booster), 로지스틱은 joblib.

### evaluate.evaluate_all(cfg) -> None (predictions.parquet 읽음)
산출물 (전부 `outputs/`):
- `metrics.csv`: (model × split) 별 Lift@10%·Precision@10%·PR-AUC·ROC-AUC·Brier·n·base_rate. Lift@10% = Precision@10% / base_rate.
- `calibration.png` + `calibration.csv`: valid로 isotonic(표본<300이면 sigmoid) 학습→test 곡선. 보정기는 joblib 저장, 보정확률 `p_calibrated`를 predictions.parquet에 추가.
- `shap_summary.png` + `shap_values.parquet`(test 행별 상위 SHAP, 컬럼: brand_id, year, feature, shap_value, feature_value) — 대시보드가 읽음.
- `label_decomposition.csv`: 사건유형별 recall@10% (통합모델이 쉬운 사건만 잡는지).
- `newbrand_split.csv`: 기존 vs 신규 브랜드 성능 분리.
- `ablation.csv`: 피처그룹 누적 추가(수준→+변화율→+추세→+업종) 별 test Lift@10%.
- 콘솔에 기준모형 3종 vs LightGBM 비교표 출력.

## 4. M4 — portfolio.py / news_llm.py / memo_llm.py

### portfolio.build_portfolio(cfg) -> None
- 입력: predictions.parquet(test 연도, p_calibrated 우선 else p_lgbm), panel(점포수).
- **예시 여신 포트폴리오 합성**(seed 고정): test 브랜드 중 `n_brands_sampled`개, exposure_i = n_stores × `exposure_scale_per_store_mkrw` × LogNormal(0, σ) [백만원]. **합성임을 모든 산출물에 명시.**
- 집중도: 브랜드별 exposure, 상위10 비중, HHI.
- 예상손실: `EL_i = exposure_i × PD_i × LGD` (LGD 시나리오 3종). 스트레스: 위험 상위 `stress_top_pct` 브랜드 PD×`stress_pd_multiplier`(cap 1.0).
- 위험등급: p 분위수 컷(config `portfolio.risk_grades`) → High/Medium/Low.
- 산출: `outputs/portfolio.csv`(브랜드별), `outputs/portfolio_summary.json`(총 exposure·HHI·상위10집중·시나리오별 EL·스트레스 EL·헤드라인 문장·가정 명세).

### news_llm.py
- `fetch_news(brand_names: list[str], cfg) -> dict[str, list]`: Google News RSS(`https://news.google.com/rss/search?q={quote(query)}&hl=ko&gl=KR&ceid=KR:ko`, feedparser)로 브랜드별 기사 수집(제목·링크·발행일·출처), **원본을 data/raw/news/{safe_name}.json 스냅샷 저장**. 브랜드명+query_suffixes 조합. 실패 무시(빈 리스트).
- `extract_signals(articles, cfg) -> list[dict]`: Anthropic API(`ANTHROPIC_API_KEY`)로 구조화 추출 `{brand, event_type: 본부분쟁|재무이슈|집단폐점|기타|무관, date, evidence_sentence, confidence: 상|중|하, source_url, published}`. **키 없으면 호출 생략하고 규칙기반(키워드) 라이트 신호로 폴백 + `llm_used: false` 표기.** 산출: `outputs/news_signals.json`.
- ⚠️ 뉴스 신호는 **모델 점수에 절대 미투입**. 화면 표시 전용.

### memo_llm.generate_memo(context: dict, cfg) -> str
- context = {brand_name, grade, prob, shap_top(list), panel_metrics(dict), news(list), portfolio(dict)}.
- LLM 프롬프트 원칙: **입력 근거만 인용, 입력 밖 주장 금지**, 출처·발행일 표기, "자동 여신결정 아님·2선 리스크 참고" 문구 포함. 마크다운 반환.
- 키 없으면 결정적 템플릿 폴백(동일 입력→동일 출력, `llm_used: false` 각주).

## 5. M5 — app.py (Streamlit)

- 실행: `streamlit run src/app.py` (루트에서). 데이터는 위 산출물 파일들만 읽음(재계산 없음).
- 화면1 브랜드 상세: 브랜드 선택 → 위험등급 배지 + (보정 안정 시) 확률/점수 · SHAP 상위 4 요인(막대) · 공시 핵심지표 미니 시계열 · 뉴스 경보 카드(출처·발행일·"점수 미반영·사실관계 별도확인") · 집중 여신·상관손실 시나리오(LGD 3종+스트레스, 억원) · 리스크관리 권고(등급별 문구, "자동 결정 아님") · 심사메모 생성 버튼.
- 화면2 포트폴리오 뷰: 총 exposure·HHI·상위10 집중 KPI, 위험등급×exposure 산점/버블, 위험 상위 브랜드 랭킹 표, 시나리오 EL 비교.
- 합성 exposure 경고 배너 상시 표시.

## 6. run_pipeline.py (루트)

`python run_pipeline.py --step all|collect|panel|features|labels|model|evaluate|portfolio|news [--demo]`
- `--demo`: make_synthetic_panel로 전체 파이프라인 스모크 실행, 산출물은 `outputs/_smoke/`로 격리(제출 지표 오염 금지).
- 각 스텝은 독립 실행 가능(중간 산출물 파일 기반).

## 7. tests/ (pytest 불요 — 표준 assert 스크립트)

`python -m tests.test_sanity` 로 실행. 검증 항목:
1. 라벨 규칙: 수제 미니 패널로 사건 발동·극단·게이트·healthy gate 케이스별 기대 라벨 일치
2. 시점 누출: features의 (brand,t) 행이 t+1 패널값 변화에 불변함 (t+1 값 교란 후 재계산 비교)
3. 시간분할: train/valid/test 연도 교집합 없음, test가 최댓값
4. predictions 스키마·확률 범위 [0,1]
5. 포트폴리오: EL ≤ exposure, 시나리오 단조성(LGD↑→EL↑)
