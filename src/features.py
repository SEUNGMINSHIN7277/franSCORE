# -*- coding: utf-8 -*-
"""M2 피처 생성 (INTERFACES.md §2).

행 (brand_id, t)의 피처는 **t 이하 연도 값만** 사용한다 (⛔ t+1 절대 금지).
- 브랜드 자기 이력: 그룹 내 shift/rolling (과거 방향만).
- 횡단면 통계(업종 중앙값·rank pct·윈저라이즈 경계)는 **해당 연도(t) 행만** 사용
  — 연도별 계산이므로 t+1 값이 바뀌어도 (brand, t) 행은 불변 (누출 테스트 대상).

피처 그룹 (계약 §2):
- f_lvl_*    수준: n_stores, avg_sales(log1p), direct_ratio, brand_age(관측연차)
- f_chg_*    변화율: store_growth_rate, sales_growth, contract_end_rate,
             신규개점률(n_new/n_stores_{t-1}), 명의변경률(n_name_change/n_stores_{t-1})
- f_trd_*    추세: 최근 2~3년 store_growth·sales_growth 평균/기울기/표준편차
             (rolling window=3, 유효 관측 <2면 NaN 허용)
- f_ind_*    업종 상대위치: store_growth·real_sales_growth·contract_end_rate의
             업종그룹×연도 내 분위수(rank pct) + 업종 대분류 더미
- f_struct_* 라벨 차원 밖: direct_ratio 변화, 직영점수 변화율, 신규개점-종료 격차

구현상 해석 (계약 모호점 — 명시):
1. 윈저라이즈는 **연도별 횡단면** 분위수로 클립 (t+1 분포 미사용 — 시점누출 방지).
   더미(f_ind_major_*)를 제외한 모든 f_* 연속 피처에 적용.
2. 직영점수 변화율 분모는 (n_direct_{t-1} + 1) — 직영 0점포 브랜드의 0나눗셈 방지
   (스무딩 가정, 로그에 기록).
3. 신규개점-종료 격차는 (n_new − 종료 − 해지) / n_stores_{t-1} 로 정규화.
4. 파생 지표 정의는 labels.compute_derived_metrics 공용 (단일 정의 원칙).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common import get_logger, load_config, make_synthetic_panel, set_seed
from src.labels import compute_derived_metrics

log = get_logger("features")

_TRD_WINDOW = 3   # 최근 2~3년 추세 창
_TRD_MIN_OBS = 2  # 유효 관측 최소 개수 (미만이면 NaN)


def _slope(a: np.ndarray) -> float:
    """rolling 창 내 선형추세 기울기 (NaN 제외, 유효 관측 <2면 NaN)."""
    m = ~np.isnan(a)
    if m.sum() < 2:
        return np.nan
    x = np.arange(len(a), dtype=float)[m]
    return float(np.polyfit(x, a[m], 1)[0])


def build_features(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """계약 §2: [brand_id, year] + f_* 컬럼 반환. (brand, t) 행은 t 이하 연도만 사용."""
    df = compute_derived_metrics(panel)
    gb = df.groupby("brand_id", sort=False)
    feat = df[["brand_id", "year"]].copy()

    # --- f_lvl_ 수준 -------------------------------------------------------
    feat["f_lvl_n_stores"] = df["n_stores"]
    feat["f_lvl_avg_sales_log"] = np.log1p(df["avg_sales"])
    feat["f_lvl_direct_ratio"] = df["direct_ratio"]
    feat["f_lvl_brand_age"] = (gb.cumcount() + 1).astype(float)  # 관측연차
    # 명세 3.1 "면적당매출" 수준 (3.3㎡당 평균매출액)
    feat["f_lvl_sales_per_area_log"] = np.log1p(df["avg_sales_per_area"])
    # 명세 3.1 업력 — 가맹사업개시연도 기준 실제 업력 (관측연차와 별개 신호)
    if "biz_start_year" in df.columns:
        feat["f_lvl_biz_age"] = (df["year"] - df["biz_start_year"]).astype(float)
    # 본부 규모 (라벨 차원 밖)
    if "emp_cnt" in df.columns:
        feat["f_lvl_emp_cnt_log"] = np.log1p(pd.to_numeric(df["emp_cnt"], errors="coerce"))

    # --- f_chg_ 변화율 -----------------------------------------------------
    feat["f_chg_store_growth"] = df["store_growth_rate"]
    feat["f_chg_sales_growth"] = df["sales_growth"]
    feat["f_chg_contract_end_rate"] = df["contract_end_rate"]
    feat["f_chg_new_open_rate"] = df["n_new"] / df["_prev_stores"]
    feat["f_chg_name_change_rate"] = df["n_name_change"] / df["_prev_stores"]
    # 명세 3.1 면적당매출 전년比 변화율
    feat["f_chg_sales_per_area_growth"] = df["sales_per_area_growth"]

    # --- f_trd_ 추세 (브랜드 자기 이력, 과거 방향 rolling만) ---------------
    for metric, tag in [("store_growth_rate", "store_growth"),
                        ("sales_growth", "sales_growth"),
                        ("sales_per_area_growth", "sales_per_area")]:
        s = gb[metric]
        feat[f"f_trd_{tag}_mean"] = s.transform(
            lambda x: x.rolling(_TRD_WINDOW, min_periods=_TRD_MIN_OBS).mean())
        feat[f"f_trd_{tag}_std"] = s.transform(
            lambda x: x.rolling(_TRD_WINDOW, min_periods=_TRD_MIN_OBS).std())
        feat[f"f_trd_{tag}_slope"] = s.transform(
            lambda x: x.rolling(_TRD_WINDOW, min_periods=1).apply(_slope, raw=True))

    # --- f_ind_ 업종 상대위치 (업종그룹×연도 내 rank pct — 해당 연도 행만) --
    grp = df.groupby(["industry_group", "year"])
    feat["f_ind_store_growth_pct"] = grp["store_growth_rate"].rank(pct=True)
    feat["f_ind_real_sales_growth_pct"] = grp["real_sales_growth"].rank(pct=True)
    feat["f_ind_contract_end_pct"] = grp["contract_end_rate"].rank(pct=True)
    dummies = pd.get_dummies(df["industry_major"], prefix="f_ind_major", dtype=float)
    feat = pd.concat([feat, dummies], axis=1)

    # --- f_struct_ 라벨 차원 밖 구조 신호 ----------------------------------
    # 명세 3.1 핵심: "GBM이 persistence를 이길 근거" — 라벨(점포/매출/계약종료)과
    # 다른 차원의 정보. 직영/가맹 비중, 지역 분산, 개폐점 격차.
    feat["f_struct_direct_ratio_chg"] = df["direct_ratio"] - df["_prev_direct_ratio"]
    feat["f_struct_direct_growth"] = (df["n_direct"] - df["_prev_direct"]) / (df["_prev_direct"] + 1.0)
    feat["f_struct_open_close_gap"] = (
        df["n_new"] - df["n_contract_end"] - df["n_contract_cancel"]) / df["_prev_stores"]

    # 명세 3.1 "지역 분산(집중도)" — 브랜드가 몇 개 시도에 퍼져 있고 얼마나 몰려 있는가.
    # (지역 집중이 심한 브랜드는 지역 경기·상권 충격에 취약 → 라벨 차원 밖 구조 위험)
    g2 = df.groupby("brand_id", sort=False)
    for src, tag in [("n_regions", "n_regions"), ("region_hhi", "region_hhi"),
                     ("top_region_share", "top_region_share")]:
        if src in df.columns:
            feat[f"f_struct_{tag}"] = pd.to_numeric(df[src], errors="coerce")
            prev = g2[src].shift(1).where(df["_consec"]) if "_consec" in df.columns \
                else g2[src].shift(1)
            feat[f"f_struct_{tag}_chg"] = feat[f"f_struct_{tag}"] - pd.to_numeric(prev, errors="coerce")
    # 지역당 평균 점포수 (밀도) — 같은 점포수라도 소수 지역 집중 여부를 구분
    if "n_regions" in df.columns:
        nreg = pd.to_numeric(df["n_regions"], errors="coerce")
        feat["f_struct_stores_per_region"] = df["n_stores"] / nreg.where(nreg > 0)

    # --- 윈저라이즈: 연도별 횡단면 분위수 클립 (t+1 미사용 — 누출 방지) ----
    fcols = [c for c in feat.columns if c.startswith("f_")]
    cont_cols = [c for c in fcols if not c.startswith("f_ind_major_")]
    pct = float(cfg["sample"]["winsorize_pct"])
    for c in cont_cols:
        lo = feat.groupby("year")[c].transform(lambda s: s.quantile(pct))
        hi = feat.groupby("year")[c].transform(lambda s: s.quantile(1.0 - pct))
        feat[c] = feat[c].clip(lower=lo, upper=hi)

    feat = feat.sort_values(["brand_id", "year"]).reset_index(drop=True)

    log.info(
        "features: %d rows x %d features (winsorize per-year pct=%.3f on %d continuous cols)",
        len(feat), len(fcols), pct, len(cont_cols),
    )
    log.info(
        "features: assumptions — direct_growth denominator=(prev_direct+1) smoothing; "
        "open_close_gap normalized by prev_stores; year-gap rows get NaN change metrics."
    )
    _write_summary(feat, fcols, cfg)
    return feat


def _write_summary(feat: pd.DataFrame, fcols: list[str], cfg: dict) -> None:
    """피처 목록·결측률·분포 요약 → outputs/feature_summary.csv."""
    rows = []
    for c in fcols:
        s = feat[c]
        rows.append({
            "feature": c,
            "group": c.split("_")[1],  # lvl/chg/trd/ind/struct
            "n": int(s.size),
            "n_missing": int(s.isna().sum()),
            "missing_rate": float(s.isna().mean()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()) if s.notna().any() else np.nan,
            "p50": float(s.quantile(0.5)) if s.notna().any() else np.nan,
            "max": float(s.max()) if s.notna().any() else np.nan,
        })
    summary = pd.DataFrame(rows)
    out_dir = Path(cfg["paths"]["outputs"])  # ⚠️ 호출 시점에 해석 (demo 격리 대응)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "feature_summary.csv"
    summary.to_csv(path, index=False, encoding="utf-8")
    log.info("features: summary -> %s", path)


# ---------------------------------------------------------------------------
# 단독 실행 (합성 패널 스모크 — 산출물은 outputs/_smoke 격리)
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    smoke_cfg = {**cfg, "paths": {**cfg["paths"], "outputs": cfg["paths"]["demo_outputs"]}}
    panel = make_synthetic_panel(cfg)
    feat = build_features(panel, smoke_cfg)
    fcols = [c for c in feat.columns if c.startswith("f_")]
    groups = pd.Series([c.split("_")[1] for c in fcols]).value_counts().to_dict()
    print("=== features smoke (synthetic panel; outputs isolated to _smoke) ===")
    print(f"rows={len(feat)}  n_features={len(fcols)}  by_group={groups}")
    miss = feat[fcols].isna().mean().sort_values(ascending=False)
    print("top-5 missing rates:")
    print(miss.head(5).to_string())


if __name__ == "__main__":
    main()
