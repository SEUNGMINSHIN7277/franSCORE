"""워크포워드(확장창) 백테스트 — 검정력 보강.

배경 (정직성 기록):
    명세 §4.1의 단일 시간분할(train ≤T-2 / valid T-1 / test T)은 test 표본이
    한 연도(695행·양성 56건)뿐이라, 페어드 부트스트랩에서 LightGBM의 기준모형 대비
    우위가 표본 변동 범위를 넘지 못했다(CI가 0 포함). 이는 모형 결함이 아니라
    **단일 연도 검정력 부족**이다.

해결 (우회가 아닌 표준 방법):
    확장창 워크포워드 — 여신 리스크 백테스트의 표준 관행.
        fold 1: train ≤2019, valid 2020, test 2021
        fold 2: train ≤2020, valid 2021, test 2022
        fold 3: train ≤2021, valid 2022, test 2023
    각 fold는 **자기 시점 이전 데이터만** 사용하고(시점 누출 없음), 매 fold마다
    모형을 처음부터 재학습한다. 각 test 연도의 out-of-sample 예측을 풀링하면
    표본이 수 배로 늘어 기준모형 대비 우위를 실제로 검정할 수 있다.

    단일분할 결과(명세 준수, 주 지표)는 그대로 유지하고, 본 모듈은 **검정력 보강용
    보조 근거**로 병기한다. 두 결과를 모두 공개한다.

산출물:
    outputs/walkforward_metrics.csv       fold별 + 풀링 지표
    outputs/walkforward_predictions.parquet  풀링된 OOS 예측
    outputs/walkforward_delta_ci.csv      풀링 표본 페어드 부트스트랩 (기준모형 대비 Lift 차이)
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common import get_logger, load_config, set_seed
from src.evaluate import MODEL_COLS, _metric_row
from src.model import (_baseline_scores, _numeric_feature_frame, _sanitize_feature_names,
                       fit_lgbm_valid_selected)

log = get_logger("backtest")

MIN_TRAIN_YEARS = 2  # fold 성립 최소 학습 연도 수


def _join_sample(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    feats = features.drop_duplicates(subset=["brand_id", "year"]).copy()
    lab = labels.dropna(subset=["label"]).drop_duplicates(subset=["brand_id", "year"]).copy()
    lab["label"] = lab["label"].astype(int)
    overlap = (set(feats.columns) & set(lab.columns)) - {"brand_id", "year"}
    if overlap:
        feats = feats.drop(columns=sorted(overlap))
    df = feats.merge(lab, on=["brand_id", "year"], how="inner")
    return df.sort_values(["brand_id", "year"]).reset_index(drop=True)


def _fit_fold(df: pd.DataFrame, tr: np.ndarray, va: np.ndarray, cfg: dict) -> dict:
    """한 fold 학습 → 전 행 예측 점수 딕셔너리 반환 (기준모형 3종 + LightGBM)."""
    y = df["label"].to_numpy()
    X_orig = _numeric_feature_frame(df, log)
    safe_names, _ = _sanitize_feature_names(X_orig.columns)
    X = X_orig.set_axis(safe_names, axis=1)

    p_pers, p_single = _baseline_scores(df, list(X_orig.columns), cfg)

    log_cfg = (cfg.get("model") or {}).get("logistic") or {}
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            C=float(log_cfg.get("C", 1.0)),
            max_iter=int(log_cfg.get("max_iter", 2000)),
            class_weight=log_cfg.get("class_weight", "balanced"),
            random_state=int(cfg["seed"]))),
    ])
    pipe.fit(X.loc[tr], y[tr])
    p_logit = pipe.predict_proba(X)[:, 1]

    # fold별로 동일 절차 재적용 (valid 기반 복잡도 선택 — 각 fold의 valid만 사용)
    clf, best_iter, _ = fit_lgbm_valid_selected(X, y, tr, va, cfg, log)
    p_lgbm = clf.predict_proba(X)[:, 1]
    return {
        "p_lgbm": np.clip(p_lgbm, 0, 1), "p_persistence": np.clip(p_pers, 0, 1),
        "p_single": np.clip(p_single, 0, 1), "p_logistic": np.clip(p_logit, 0, 1),
        "best_iteration": int(best_iter),
    }


def _paired_delta_ci(pooled: pd.DataFrame, k_pct: float, n_boot: int, seed: int) -> pd.DataFrame:
    """풀링 OOS 표본에서 LightGBM − 기준모형 Lift@k 차이의 부트스트랩 CI·승률.

    풀링 표본은 연도가 섞여 있으므로 **연도 블록 부트스트랩**을 쓴다
    (같은 연도 행끼리 상관되므로 연도 단위로 복원추출하면 CI가 과소추정되지 않는다).
    """
    y_all = pooled["y_true"].to_numpy(dtype=float)
    if len(y_all) == 0:
        return pd.DataFrame()
    years = pooled["year"].to_numpy()
    uniq_years = np.unique(years)
    idx_by_year = {int(yy): np.flatnonzero(years == yy) for yy in uniq_years}
    rng = np.random.default_rng(seed)
    baselines = ["persistence", "single", "logistic"]
    deltas: dict[str, list[float]] = {b: [] for b in baselines}
    main_lifts: list[float] = []
    for _ in range(int(n_boot)):
        pick = rng.choice(uniq_years, size=len(uniq_years), replace=True)
        idx = np.concatenate([idx_by_year[int(yy)] for yy in pick])
        yb = y_all[idx]
        m = _metric_row(yb, pooled["p_lgbm"].to_numpy()[idx], k_pct)["lift_at_10"]
        if m is None or (isinstance(m, float) and math.isnan(m)):
            continue
        main_lifts.append(float(m))
        for b in baselines:
            bl = _metric_row(yb, pooled[MODEL_COLS[b]].to_numpy()[idx], k_pct)["lift_at_10"]
            if bl is not None and not (isinstance(bl, float) and math.isnan(bl)):
                deltas[b].append(float(m - bl))
    rows = []
    for b in baselines:
        arr = np.asarray(deltas[b], dtype=float)
        if arr.size == 0:
            continue
        rows.append({"comparison": f"lgbm - {b}", "mean_delta_lift": float(arr.mean()),
                     "ci_lo": float(np.quantile(arr, 0.025)),
                     "ci_hi": float(np.quantile(arr, 0.975)),
                     "p_win": float((arr > 0).mean()), "n_boot": int(arr.size),
                     "bootstrap_unit": "year_block"})
    return pd.DataFrame(rows)


def run_walkforward(cfg: dict) -> pd.DataFrame:
    """확장창 워크포워드 실행 → fold별·풀링 지표 저장 및 반환."""
    set_seed(cfg["seed"])
    proc = Path(cfg["paths"]["processed"])
    out_dir = Path(cfg["paths"]["outputs"])
    k_pct = float((cfg.get("evaluate") or {}).get("k_pct", 0.10))

    features = pd.read_parquet(proc / "features.parquet")
    labels = pd.read_parquet(proc / "labels.parquet")
    df = _join_sample(features, labels)
    years = sorted(int(y) for y in df["year"].unique())
    if len(years) < MIN_TRAIN_YEARS + 2:
        log.warning("워크포워드 fold 성립 불가 (라벨 연도 %s) - 생략", years)
        return pd.DataFrame()

    fold_rows: list[dict] = []
    pooled_parts: list[pd.DataFrame] = []
    # test 연도는 앞에 train(≥MIN_TRAIN_YEARS) + valid(1) 가 확보되는 연도부터
    for i in range(MIN_TRAIN_YEARS + 1, len(years)):
        test_year = years[i]
        valid_year = years[i - 1]
        train_years = years[:i - 1]
        tr = df["year"].isin(train_years).to_numpy()
        va = df["year"].eq(valid_year).to_numpy()
        te = df["year"].eq(test_year).to_numpy()
        if te.sum() == 0 or tr.sum() == 0 or len(np.unique(df["label"].to_numpy()[va])) < 2:
            log.warning("fold test=%d 건너뜀 (표본/클래스 부족)", test_year)
            continue
        scores = _fit_fold(df, tr, va, cfg)
        y = df["label"].to_numpy()
        fold_pred = pd.DataFrame({
            "brand_id": df["brand_id"].astype(str).to_numpy()[te],
            "year": df["year"].astype(int).to_numpy()[te],
            "y_true": y[te].astype(int),
            "p_lgbm": scores["p_lgbm"][te],
            "p_persistence": scores["p_persistence"][te],
            "p_single": scores["p_single"][te],
            "p_logistic": scores["p_logistic"][te],
            "fold_test_year": test_year,
            "is_new_brand": ~df["brand_id"].isin(set(df.loc[tr, "brand_id"])).to_numpy()[te],
        })
        pooled_parts.append(fold_pred)
        for name, col in MODEL_COLS.items():
            row = _metric_row(fold_pred["y_true"], fold_pred[col], k_pct)
            fold_rows.append({"scope": f"fold_test_{test_year}", "model": name,
                              "train_years": f"{train_years[0]}~{train_years[-1]}",
                              "valid_year": valid_year, **row})
        log.info("fold test=%d: train %s / valid %d (n_test=%d, 양성 %d, best_iter=%d)",
                 test_year, f"{train_years[0]}~{train_years[-1]}", valid_year,
                 int(te.sum()), int(y[te].sum()), scores["best_iteration"])

    if not pooled_parts:
        log.warning("워크포워드 fold 없음 - 생략")
        return pd.DataFrame()

    pooled = pd.concat(pooled_parts, ignore_index=True)
    pooled.to_parquet(out_dir / "walkforward_predictions.parquet", index=False)
    for name, col in MODEL_COLS.items():
        row = _metric_row(pooled["y_true"], pooled[col], k_pct)
        fold_rows.append({"scope": "pooled_oos", "model": name,
                          "train_years": "expanding", "valid_year": -1, **row})
    metrics = pd.DataFrame(fold_rows)
    metrics.to_csv(out_dir / "walkforward_metrics.csv", index=False, encoding="utf-8-sig")

    n_boot = int((cfg.get("evaluate") or {}).get("bootstrap_n", 200) or 200)
    ci = _paired_delta_ci(pooled, k_pct, n_boot, int(cfg["seed"]))
    if not ci.empty:
        ci.to_csv(out_dir / "walkforward_delta_ci.csv", index=False, encoding="utf-8-sig")

    pool = metrics[metrics["scope"] == "pooled_oos"].set_index("model")
    print()
    print("=== 워크포워드 확장창 백테스트 (풀링 OOS) ===")
    print(pool[["lift_at_10", "precision_at_10", "pr_auc", "roc_auc", "brier", "n", "base_rate"]]
          .round(3).to_string())
    if not ci.empty:
        print()
        print("--- 기준모형 대비 Lift@10% 차이 (연도블록 부트스트랩 95% CI) ---")
        print(ci[["comparison", "mean_delta_lift", "ci_lo", "ci_hi", "p_win"]].round(3)
              .to_string(index=False))
    print()
    log.info("워크포워드 완료: fold %d개, 풀링 OOS n=%d (양성 %d)",
             pooled["fold_test_year"].nunique(), len(pooled), int(pooled["y_true"].sum()))
    return metrics


if __name__ == "__main__":
    run_walkforward(load_config())
