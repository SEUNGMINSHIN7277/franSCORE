"""M4 포트폴리오 — 예시(합성) 여신 포트폴리오의 집중도·예상손실 산출.

INTERFACES.md §4 `portfolio.build_portfolio(cfg)` 구현.

⚠️ 정직성 원칙: 본 모듈의 exposure(여신액)는 실여신 데이터가 아니라
   seed 고정 난수로 합성한 **예시값**이다(방법론 실증용).
   그 사실을 portfolio.csv 컬럼과 portfolio_summary.json의 'assumptions'에 명시한다.

입력 (호출 시점에 cfg['paths']에서 해석 — 데모 러너가 경로를 교체할 수 있음):
  - data/processed/predictions.parquet : test split, p_calibrated 우선 else p_lgbm
  - data/processed/panel.parquet       : n_stores (exposure 스케일링)
    (panel 부재 시 features_matrix.parquet의 f_lvl_n_stores로 폴백, 둘 다 없으면 명확한 에러)

산출:
  - outputs/portfolio.csv           : 브랜드별 exposure·등급·시나리오별 EL
  - outputs/portfolio_summary.json  : 총 exposure·HHI·상위10집중·시나리오별 EL·
                                      스트레스 EL·헤드라인 문장·가정 명세

단위: exposure·EL은 백만원(MKRW), 표시용 변환은 억원(= 백만원/100).
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import get_logger, load_config, set_seed

log = get_logger("portfolio")

MKRW_PER_EKW = 100.0  # 1억원 = 100백만원


def _to_ekw(mkrw: float) -> float:
    """백만원 → 억원 변환 (표시용)."""
    return float(mkrw) / MKRW_PER_EKW


def _pick_pd_column(test: pd.DataFrame) -> tuple[pd.Series, str]:
    """PD 대용 확률 컬럼 선택: p_calibrated(값 존재 시) 우선, 결측은 p_lgbm으로 보충."""
    if "p_calibrated" in test.columns and test["p_calibrated"].notna().any():
        p = test["p_calibrated"].astype(float)
        if "p_lgbm" in test.columns:
            p = p.fillna(test["p_lgbm"].astype(float))
        return p.clip(0.0, 1.0), "p_calibrated"
    if "p_lgbm" in test.columns:
        return test["p_lgbm"].astype(float).clip(0.0, 1.0), "p_lgbm"
    raise KeyError("predictions.parquet에 p_calibrated/p_lgbm 컬럼이 없습니다.")


def _load_store_counts(processed: Path) -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    """점포수 소스 로드.

    반환: (stores[brand_id, year, n_stores], meta[브랜드 메타 or None], 소스 파일명)
    panel.parquet 우선, 없으면 features_matrix.parquet의 f_lvl_n_stores 폴백.
    """
    panel_path = processed / "panel.parquet"
    if panel_path.exists():
        panel = pd.read_parquet(panel_path)
        stores = panel[["brand_id", "year", "n_stores"]].copy()
        meta_cols = [c for c in ["brand_id", "brand_name", "industry_major", "industry_mid"]
                     if c in panel.columns]
        meta = (panel.sort_values(["brand_id", "year"])
                     .groupby("brand_id", as_index=False).tail(1)[meta_cols])
        return stores, meta, "panel.parquet"

    fm_path = processed / "features_matrix.parquet"
    if fm_path.exists():
        fm = pd.read_parquet(fm_path)
        if "f_lvl_n_stores" not in fm.columns:
            raise KeyError(
                f"{fm_path}에 f_lvl_n_stores 컬럼이 없어 exposure 스케일링 불가.")
        stores = (fm[["brand_id", "year", "f_lvl_n_stores"]]
                  .rename(columns={"f_lvl_n_stores": "n_stores"}))
        return stores, None, "features_matrix.parquet"

    raise FileNotFoundError(
        f"점포수 소스를 찾지 못했습니다: {panel_path} 또는 {fm_path} 필요 "
        "(panel 또는 features 스텝을 먼저 실행하세요).")


def _attach_store_counts(test: pd.DataFrame, stores: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """(brand_id, year) 정확 매칭 → 실패 시 브랜드별 최신 관측치로 폴백."""
    stores = stores.dropna(subset=["n_stores"])
    merged = test.merge(stores, on=["brand_id", "year"], how="left")
    n_fallback = 0
    missing = merged["n_stores"].isna()
    if missing.any():
        last = (stores.sort_values(["brand_id", "year"])
                      .groupby("brand_id")["n_stores"].last())
        merged.loc[missing, "n_stores"] = merged.loc[missing, "brand_id"].map(last)
        n_fallback = int(missing.sum() - merged["n_stores"].isna().sum())
    return merged, n_fallback


def build_portfolio(cfg: dict) -> None:
    """예시 여신 포트폴리오 합성 → 집중도·시나리오별 예상손실 산출 (INTERFACES.md §4)."""
    set_seed(cfg["seed"])
    pcfg = cfg["portfolio"]
    # 경로는 반드시 호출 시점에 해석 (데모 러너가 cfg['paths']를 교체함)
    processed = Path(cfg["paths"]["processed"])
    outputs = Path(cfg["paths"]["outputs"])
    outputs.mkdir(parents=True, exist_ok=True)

    # ---------------- 1) 예측확률 (test split) ----------------
    pred_path = processed / "predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"predictions.parquet 없음: {pred_path} — model 스텝을 먼저 실행하세요.")
    pred = pd.read_parquet(pred_path)
    test = pred[pred["split"].astype(str) == "test"].copy()
    if test.empty:
        raise ValueError("predictions.parquet에 split=='test' 행이 없습니다.")
    test_year = int(test["year"].max())
    if test["year"].nunique() > 1:
        log.warning("test split에 연도가 %d개 존재 — 계약(§3)상 단일 연도. 최종 연도 %d만 사용.",
                    test["year"].nunique(), test_year)
        test = test[test["year"] == test_year]
    test = test.drop_duplicates(subset="brand_id", keep="last").reset_index(drop=True)

    p, pd_col = _pick_pd_column(test)
    test["pd_1y"] = p.values
    log.info("test 연도=%d, 브랜드 %d개, PD 컬럼=%s", test_year, len(test), pd_col)

    # ---------------- 2) 점포수 결합 (exposure 스케일) ----------------
    stores, meta, store_src = _load_store_counts(processed)
    test, n_store_fallback = _attach_store_counts(test[["brand_id", "year", "pd_1y"]], stores)
    if n_store_fallback:
        log.info("점포수 (brand,year) 정확 매칭 실패 %d건 → 브랜드별 최신 관측치로 폴백", n_store_fallback)
    n_drop = int((test["n_stores"].isna() | (test["n_stores"] <= 0)).sum())
    if n_drop:
        log.warning("점포수 결측/비양수 %d개 브랜드는 포트폴리오 후보에서 제외", n_drop)
    test = test[test["n_stores"].notna() & (test["n_stores"] > 0)].reset_index(drop=True)
    if test.empty:
        raise ValueError("점포수가 유효한 test 브랜드가 없어 포트폴리오를 구성할 수 없습니다.")

    # ---------------- 3) 위험등급 (test 전체 분포 내 순위 기반) ----------------
    # 값 임계 컷(p >= q0.90)은 보정확률 동률 블록이 경계에 걸리면 High가 설계 비율
    # (상위 10%)을 크게 초과한다(리뷰 확정 결함). 순위(pct rank, method='first') 기반으로
    # 부여해 등급 비율이 항상 설계값과 일치하게 한다. 컷 값은 참고용으로만 기록.
    grades_cfg = pcfg["risk_grades"]
    test = test.copy()
    test["pd_rank_pct"] = test["pd_1y"].rank(pct=True, method="first")
    hi_cut = float(np.quantile(test["pd_1y"], float(grades_cfg["high"])))
    med_cut = float(np.quantile(test["pd_1y"], float(grades_cfg["medium"])))

    # ---------------- 4) 예시 포트폴리오 합성 (seed 고정 — ⚠️ 합성 exposure) ----------------
    rng = np.random.default_rng(cfg["seed"])
    n_sample = min(int(pcfg["n_brands_sampled"]), len(test))
    eligible = test.sort_values("brand_id").reset_index(drop=True)  # 입력 행순서 무관 결정성
    take = np.sort(rng.choice(len(eligible), size=n_sample, replace=False))
    port = eligible.iloc[take].reset_index(drop=True)

    scale = float(pcfg["exposure_scale_per_store_mkrw"])
    sigma = float(pcfg["exposure_lognorm_sigma"])
    lognorm = rng.lognormal(mean=0.0, sigma=sigma, size=n_sample)
    port["exposure_mkrw"] = port["n_stores"].to_numpy(dtype=float) * scale * lognorm
    port["exposure_ekw"] = port["exposure_mkrw"] / MKRW_PER_EKW

    total_exposure = float(port["exposure_mkrw"].sum())
    port["exposure_share"] = port["exposure_mkrw"] / total_exposure

    port["risk_grade"] = np.select(
        [port["pd_rank_pct"] > float(grades_cfg["high"]),
         port["pd_rank_pct"] > float(grades_cfg["medium"])],
        ["High", "Medium"], default="Low")

    # ---------------- 5) 집중도 ----------------
    shares_desc = port["exposure_share"].sort_values(ascending=False)
    hhi = float((shares_desc ** 2).sum())
    top10_share = float(shares_desc.head(10).sum())
    top1_share = float(shares_desc.iloc[0])

    # ---------------- 6) 스트레스 PD: 위험 상위 stress_top_pct 브랜드 PD × multiplier (cap 1.0) ----
    stress_top_pct = float(pcfg["stress_top_pct"])
    stress_mult = float(pcfg["stress_pd_multiplier"])
    n_stress = max(1, math.ceil(len(port) * stress_top_pct))
    stress_idx = port["pd_1y"].nlargest(n_stress).index
    port["is_stressed"] = port.index.isin(stress_idx)
    port["pd_stressed"] = np.where(
        port["is_stressed"], np.minimum(port["pd_1y"] * stress_mult, 1.0), port["pd_1y"])

    # ---------------- 7) 시나리오별 예상손실 EL = exposure × PD × LGD ----------------
    lgd_scenarios = sorted(float(x) for x in pcfg["lgd_scenarios"])
    el_rows = []
    for lgd in lgd_scenarios:
        key = f"lgd{round(lgd * 100):02d}"
        port[f"el_{key}_mkrw"] = port["exposure_mkrw"] * port["pd_1y"] * lgd
        port[f"stress_el_{key}_mkrw"] = port["exposure_mkrw"] * port["pd_stressed"] * lgd
        el = float(port[f"el_{key}_mkrw"].sum())
        sel = float(port[f"stress_el_{key}_mkrw"].sum())
        el_rows.append({
            "lgd": lgd,
            "el_mkrw": el,
            "el_ekw": _to_ekw(el),
            "el_pct_of_exposure": el / total_exposure,
            "stress_el_mkrw": sel,
            "stress_el_ekw": _to_ekw(sel),
            "stress_el_pct_of_exposure": sel / total_exposure,
        })

    # ---------------- 8) 브랜드 메타 결합 + CSV 저장 ----------------
    if meta is not None:
        port = port.merge(meta, on="brand_id", how="left")
    if "brand_name" not in port.columns:
        port["brand_name"] = port["brand_id"]
    else:
        port["brand_name"] = port["brand_name"].fillna(port["brand_id"])
    port["exposure_is_synthetic"] = True  # ⚠️ 합성 exposure 명시 (실여신 아님)

    lead = [c for c in ["brand_id", "brand_name", "industry_major", "industry_mid",
                        "year", "n_stores", "pd_1y", "risk_grade",
                        "exposure_mkrw", "exposure_ekw", "exposure_share",
                        "is_stressed", "pd_stressed"] if c in port.columns]
    rest = [c for c in port.columns if c not in lead]
    port = port[lead + rest].sort_values("exposure_mkrw", ascending=False).reset_index(drop=True)

    csv_path = outputs / "portfolio.csv"
    port.to_csv(csv_path, index=False, encoding="utf-8")
    log.info("portfolio.csv 저장: %s (%d 브랜드)", csv_path, len(port))

    # ---------------- 9) 요약 JSON (헤드라인 + 가정 명세) ----------------
    mid = el_rows[len(el_rows) // 2]  # 중간 LGD 시나리오를 헤드라인에 사용
    headline = (
        f"[합성 예시 여신] {test_year}년 test 브랜드 {len(port)}개로 구성한 포트폴리오의 "
        f"총 여신은 {_to_ekw(total_exposure):,.0f}억원이며, "
        f"중간 시나리오(LGD {mid['lgd']:.0%}) 기준 1년 예상손실은 "
        f"{mid['el_ekw']:,.1f}억원(여신의 {mid['el_pct_of_exposure']:.2%}), "
        f"위험 상위 {stress_top_pct:.0%} 브랜드 동시 악화 스트레스 시 "
        f"{mid['stress_el_ekw']:,.1f}억원으로 확대됩니다. "
        f"상위 10개 브랜드 여신 집중도는 {top10_share:.1%}, HHI는 {hhi:.3f}입니다.")

    grade_counts = port["risk_grade"].value_counts().to_dict()
    grade_exposure = port.groupby("risk_grade")["exposure_mkrw"].sum().to_dict()

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "synthetic": True,
        "test_year": test_year,
        "n_brands": int(len(port)),
        "pd_column_used": pd_col,
        "total_exposure_mkrw": total_exposure,
        "total_exposure_ekw": _to_ekw(total_exposure),
        "concentration": {
            "hhi": hhi,
            "top10_share": top10_share,
            "top1_share": top1_share,
        },
        "risk_grades": {
            "high_quantile": float(grades_cfg["high"]),
            "medium_quantile": float(grades_cfg["medium"]),
            "high_pd_cut": hi_cut,
            "medium_pd_cut": med_cut,
            "counts": {str(k): int(v) for k, v in grade_counts.items()},
            "exposure_mkrw": {str(k): float(v) for k, v in grade_exposure.items()},
        },
        "expected_loss_scenarios": el_rows,
        "stress": {
            "top_pct": stress_top_pct,
            "pd_multiplier": stress_mult,
            "pd_cap": 1.0,
            "n_stressed_brands": int(n_stress),
        },
        "headline": headline,
        "assumptions": {
            "synthetic_exposure": True,
            "statement_ko": (
                "본 포트폴리오의 여신 exposure는 실제 여신 데이터가 아니라 seed 고정 난수로 "
                "합성한 예시값입니다(방법론 실증용 — 자동 여신결정 아님). "
                "exposure_i = n_stores_i × 점포당 평균여신(백만원) × LogNormal(0, sigma)."),
            "exposure_scale_per_store_mkrw": scale,
            "exposure_lognorm_sigma": sigma,
            "seed": int(cfg["seed"]),
            "units": {"exposure": "백만원(MKRW)", "display": "억원 = 백만원 / 100"},
            "pd_definition": (
                "PD는 모델이 예측한 '1년 내 구조악화 전환 확률'을 부도확률의 대용 지표로 "
                "사용한 것으로, 실제 부도율과 다를 수 있습니다."),
            "el_formula": "EL_i = exposure_i × PD_i × LGD (LGD 시나리오 3종)",
            "stress_rule": (
                f"PD 상위 {stress_top_pct:.0%} 브랜드({n_stress}개)의 PD × {stress_mult} "
                "(상한 1.0) 동시 악화 가정"),
            "risk_grade_basis": (
                "test 전체 브랜드 예측확률 분포의 분위수 컷 "
                f"(High ≥ q{float(grades_cfg['high']):.2f}, Medium ≥ q{float(grades_cfg['medium']):.2f})"),
            "store_count_source": store_src,
            "store_count_fallback_rows": int(n_store_fallback),
            "brands_dropped_no_stores": int(n_drop),
        },
    }

    json_path = outputs / "portfolio_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info("portfolio_summary.json 저장: %s", json_path)
    log.info("헤드라인: %s", headline)


if __name__ == "__main__":
    build_portfolio(load_config())
