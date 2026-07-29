# -*- coding: utf-8 -*-
"""M2 라벨 생성 — '현재 양호(t) → 미래 악화 전환(t+1)' 라벨 (INTERFACES.md §2).

계약 요약
---------
- ``label=1`` ⇔ t년에 양호(healthy)했던 브랜드가 t+1년에 악화 사건을 발동. 행의 year는 t.
- 악화 사건 3종 (각각 **t+1년** 지표, 업종그룹×연도 내 분위수, config ``label.events``):
    ① store_growth_rate 하위 quantile **그리고** |Δ점포| ≥ ``min_abs_store_change``
    ② real_sales_growth 하위 quantile
    ③ contract_end_rate 상위 quantile
- 발동: ``min_events``개 이상 **or** 1개가 극단(``extreme_quantile`` 하위/상위 5%).
- healthy_at_t: t년(t−1→t 지표)에 사건이 하나도 발동하지 않음.
  ``healthy_gate_at_t=true``면 healthy_at_t=True 행만 표본으로 반환.
- 게이트: n_stores_t ≥ ``min_stores_at_t``. t+1 행이 없으면 표본 제외.
- 분위수 임계값은 사건 지표의 t+1년 **실측 분포**로 계산 (라벨 정의 — 피처 아님, 누출 아님).

구현상 해석 (계약 모호점 — 정직성 원칙에 따라 명시)
---------------------------------------------------
1. 분위수 경계 비교는 **엄격 부등호**(<, >). 그룹 전원이 동일값이면 아무도
   '하위/상위 20%'가 아니라고 본다 (동률 경계값 미발동).
2. 극단(extreme)은 해당 사건의 발동 조건(점포 사건의 |Δ점포| 게이트 포함)을 함께
   통과한 경우에만 인정한다 — 극단 ⊂ 사건.
3. 지표가 NaN(관측 첫해, 연도 갭, 매출 결측 등)이면 그 사건은 '발동 안 함' 처리.
   따라서 관측 첫해는 사건 판정 불가 → healthy로 간주된다.
4. |Δ점포| 게이트는 계약 §2의 식 그대로 **절대값** 기준(성장·감소 무관 |Δ| ≥ 3).
5. 연도 갭(전년 미관측)·전년 점포수 0이면 전년 기반 지표는 NaN.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common import (
    get_logger,
    industry_group_col,
    load_config,
    make_synthetic_panel,
    set_seed,
)

log = get_logger("labels")

# 사건 metric → 반환 컬럼 접미사 (계약 §2 반환 스키마 고정)
_METRIC_SHORT = {
    "store_growth_rate": "store",
    "real_sales_growth": "sales",
    "contract_end_rate": "contract",
}
_EV_COLS = ["ev_store", "ev_sales", "ev_contract"]


# ---------------------------------------------------------------------------
# 파생 지표 (계약 §1 — features.py와 공용, 단일 정의 원칙)
# ---------------------------------------------------------------------------

def compute_derived_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    """계약 §1 파생 지표를 (brand_id, year) 행마다 붙여 반환한다.

    추가 컬럼:
      store_growth_rate, delta_stores, contract_end_rate, sales_growth,
      real_sales_growth, direct_ratio, industry_group,
      _prev_stores, _prev_direct, _prev_direct_ratio (내부용)

    연도 갭(전년 미관측)이거나 전년 점포수 ≤ 0이면 전년 기반 지표는 NaN.
    """
    df = panel.sort_values(["brand_id", "year"]).reset_index(drop=True).copy()
    g = df.groupby("brand_id", sort=False)
    consec = g["year"].shift(1).eq(df["year"] - 1)

    prev_stores = g["n_stores"].shift(1).where(consec)
    prev_stores = prev_stores.where(prev_stores > 0)  # 분모 0 방지
    prev_sales = g["avg_sales"].shift(1).where(consec)
    prev_sales = prev_sales.where(prev_sales > 0)
    prev_direct = g["n_direct"].shift(1).where(consec)

    df["_prev_stores"] = prev_stores
    df["_prev_direct"] = prev_direct
    df["delta_stores"] = df["n_stores"] - prev_stores
    df["store_growth_rate"] = (df["n_stores"] - prev_stores) / prev_stores
    df["contract_end_rate"] = (df["n_contract_end"] + df["n_contract_cancel"]) / prev_stores
    df["sales_growth"] = df["avg_sales"] / prev_sales - 1.0

    denom = df["n_direct"] + df["n_stores"]
    df["direct_ratio"] = np.where(denom > 0, df["n_direct"] / denom, np.nan)
    g2 = df.groupby("brand_id", sort=False)
    df["_prev_direct_ratio"] = g2["direct_ratio"].shift(1).where(consec)

    # 업종그룹 (해당 연도 industry_mid 크기 ≥ 30이면 mid, 아니면 major)
    df["industry_group"] = industry_group_col(df)
    med = df.groupby(["industry_group", "year"])["sales_growth"].transform("median")
    df["real_sales_growth"] = df["sales_growth"] - med
    return df


# ---------------------------------------------------------------------------
# 사건 플래그
# ---------------------------------------------------------------------------

def _event_flags(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """각 행(=그 연도 지표)에 대해 사건/극단 발동 플래그를 계산한다.

    분위수 임계값은 업종그룹×연도 셀 내 실측 분포로 계산 (패널 전 행 사용 —
    표본 게이트와 무관하게 임계값 풀에는 모든 브랜드가 들어간다).
    """
    lab = cfg["label"]
    xq = float(lab["extreme_quantile"])
    out = pd.DataFrame(index=df.index)
    for name, spec in lab["events"].items():
        metric = spec["metric"]
        short = _METRIC_SHORT[metric]
        q = float(spec["quantile"])
        side = spec["side"]
        grp = df.groupby(["industry_group", "year"])[metric]
        thr = grp.transform(lambda s, q=q: s.quantile(q))
        if side == "lower":
            xthr = grp.transform(lambda s, xq=xq: s.quantile(xq))
            fired = df[metric] < thr
            extreme = df[metric] < xthr
        elif side == "upper":
            xthr = grp.transform(lambda s, xq=xq: s.quantile(1.0 - xq))
            fired = df[metric] > thr
            extreme = df[metric] > xthr
        else:
            raise ValueError(f"label.events.{name}: unknown side '{side}'")
        if metric == "store_growth_rate":
            # 노이즈 게이트: 절대 점포수 변화 (극단 판정에도 동일 적용)
            gate = (df["delta_stores"].abs() >= float(lab["min_abs_store_change"])).fillna(False)
            fired = fired & gate
            extreme = extreme & gate
        fired = fired.fillna(False).astype(bool)
        extreme = (extreme.fillna(False).astype(bool)) & fired
        out[f"ev_{short}"] = fired
        out[f"ex_{short}"] = extreme
    return out


# ---------------------------------------------------------------------------
# 라벨 빌드
# ---------------------------------------------------------------------------

def build_labels(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """계약 §2: [brand_id, year, label, healthy_at_t, ev_store, ev_sales,
    ev_contract, n_events, extreme_fired] 반환. 행의 year는 t (피처 시점),
    ev_*/n_events/extreme_fired는 **t+1년** 사건이다."""
    lab = cfg["label"]

    df = compute_derived_metrics(panel)
    ev = _event_flags(df, cfg)
    df = pd.concat([df, ev], axis=1)
    ex_cols = [c for c in ev.columns if c.startswith("ex_")]
    df["n_events"] = df[_EV_COLS].sum(axis=1).astype(int)
    df["extreme_fired"] = df[ex_cols].any(axis=1)
    df["deteriorated"] = (df["n_events"] >= int(lab["min_events"])) | df["extreme_fired"]
    df["healthy"] = df["n_events"] == 0

    # t 행에 t+1년 사건을 붙인다 (t+1 미관측 → inner merge에서 자동 제외)
    nxt = df[["brand_id", "year", *_EV_COLS, "n_events", "extreme_fired", "deteriorated"]].copy()
    nxt["year"] = nxt["year"] - 1
    cur = df[["brand_id", "year", "n_stores", "healthy"]].rename(columns={"healthy": "healthy_at_t"})
    out = cur.merge(nxt, on=["brand_id", "year"], how="inner")

    n_with_next = len(out)
    out = out[out["n_stores"] >= float(lab["min_stores_at_t"])]
    n_after_stores = len(out)
    if lab.get("healthy_gate_at_t", True):
        out = out[out["healthy_at_t"]]
    n_after_healthy = len(out)

    out = out.copy()
    out["label"] = out["deteriorated"].astype(int)
    out = out[["brand_id", "year", "label", "healthy_at_t", *_EV_COLS, "n_events", "extreme_fired"]]
    out = out.sort_values(["brand_id", "year"]).reset_index(drop=True)

    log.info(
        "labels: %d rows with t+1 -> %d after min_stores_at_t(%s) -> %d after healthy gate(%s)",
        n_with_next, n_after_stores, lab["min_stores_at_t"], n_after_healthy, lab.get("healthy_gate_at_t", True),
    )
    if len(out):
        log.info(
            "labels: positive_rate=%.4f | event rates(t+1): store=%.4f sales=%.4f contract=%.4f | extreme=%.4f",
            out["label"].mean(), out["ev_store"].mean(), out["ev_sales"].mean(),
            out["ev_contract"].mean(), out["extreme_fired"].mean(),
        )
    log.info(
        "labels: rule interpretation — strict quantile inequality; extreme subset of event "
        "(store gate applies); NaN metric never fires; |delta stores| gate is absolute value."
    )
    _write_composition(out, cfg)
    return out


def _write_composition(sample: pd.DataFrame, cfg: dict) -> None:
    """사건별 발동률·중복 분포·연도별 양성률 → outputs/label_composition.csv."""
    rows: list[dict] = []
    n = len(sample)
    rows.append({"section": "overall", "key": "n_sample", "value": float(n), "n": n})
    rows.append({
        "section": "overall", "key": "positive_rate",
        "value": float(sample["label"].mean()) if n else np.nan, "n": int(sample["label"].sum()) if n else 0,
    })
    for c in _EV_COLS:
        rows.append({
            "section": "event_rate", "key": c,
            "value": float(sample[c].mean()) if n else np.nan, "n": int(sample[c].sum()) if n else 0,
        })
    counts = sample["n_events"].value_counts() if n else {}
    for k in range(0, len(_EV_COLS) + 1):
        cnt = int(counts.get(k, 0))
        rows.append({"section": "overlap", "key": f"{k}_events", "value": (cnt / n) if n else np.nan, "n": cnt})
    if n:
        for year, grp in sample.groupby("year"):
            rows.append({
                "section": "year_positive_rate", "key": str(int(year)),
                "value": float(grp["label"].mean()), "n": len(grp),
            })
    comp = pd.DataFrame(rows, columns=["section", "key", "value", "n"])
    out_dir = Path(cfg["paths"]["outputs"])  # ⚠️ 호출 시점에 해석 (demo 격리 대응)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "label_composition.csv"
    comp.to_csv(path, index=False, encoding="utf-8")
    log.info("labels: composition -> %s", path)


# ---------------------------------------------------------------------------
# 단독 실행 (합성 패널 스모크 — 산출물은 outputs/_smoke 격리)
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    # 합성 스모크 산출물이 제출 지표(outputs/)를 오염시키지 않도록 격리
    smoke_cfg = {**cfg, "paths": {**cfg["paths"], "outputs": cfg["paths"]["demo_outputs"]}}
    panel = make_synthetic_panel(cfg)
    labels = build_labels(panel, smoke_cfg)
    print("=== labels smoke (synthetic panel; outputs isolated to _smoke) ===")
    print(f"rows={len(labels)}  positive_rate={labels['label'].mean():.4f}")
    print(labels.groupby("year")["label"].agg(n="count", positive_rate="mean").to_string())


if __name__ == "__main__":
    main()
