"""M1 패널: 스냅샷 → brand_id×year 롱포맷 패널 + 생존성 진단표.

핵심 규칙 (문서화):
- 패널 year = 정보공개서 기준연도(yr) − 1. 공시 수치(가맹점수·변동·평균매출)는 통상
  직전 회계연도 실적이므로 실적연도 기준으로 정렬한다. (모든 변수 동일 규칙 → 내부 정합)
- avg_sales(avrgSlsAmt)·avg_sales_per_area == 0 → 결측(NaN) 처리 (영세 브랜드 미기재 관행).
  행 자체는 유지한다 — 점포수 기반 신호는 살아있기 때문.
- n_direct(직영점수)는 본 API에 없음 → NaN (한계 명시). 대신 brand_overview에서
  업력(biz_start_year)·임직원수(emp_cnt, exec_cnt)를 보조 피처로 병합.
- 표본 필터(제출 표본): industry_major ∈ config, 브랜드 연속 관측 ≥ min_consecutive_years,
  브랜드 최대 가맹점수 ≥ min_stores. (관측 단위 하한은 labels의 min_stores_at_t가 담당)
- 산출: data/processed/panel_full.parquet(전체), panel.parquet(필터 표본),
  industry_baseline.parquet(업종 기준선), outputs/survival_report.csv(진단표)
"""
from __future__ import annotations

import pandas as pd

from src.collect import load_snapshots
from src.common import get_logger, load_config
from src.entity import build_master, normalize_name

log = get_logger("panel")

_NUM_COLS = ["n_stores", "n_new", "n_contract_end", "n_contract_cancel", "n_name_change",
             "avg_sales", "avg_sales_per_area"]


def build_panel(cfg: dict, master: pd.DataFrame | None = None) -> pd.DataFrame:
    if master is None:
        master = build_master(cfg)
    df = master.copy()

    df["year"] = df["_yr"].astype(int) - 1  # 실적연도 정렬 (상단 규칙 참조)
    df["n_stores"] = pd.to_numeric(df["frcsCnt"], errors="coerce")
    df["n_new"] = pd.to_numeric(df["newFrcsRgsCnt"], errors="coerce")
    df["n_contract_end"] = pd.to_numeric(df["ctrtEndCnt"], errors="coerce")
    df["n_contract_cancel"] = pd.to_numeric(df["ctrtCncltnCnt"], errors="coerce")
    df["n_name_change"] = pd.to_numeric(df["nmChgCnt"], errors="coerce")
    df["avg_sales"] = pd.to_numeric(df["avrgSlsAmt"], errors="coerce")
    df["avg_sales_per_area"] = pd.to_numeric(df["arUnitAvrgSlsAmt"], errors="coerce")
    df["n_direct"] = float("nan")  # API 미제공 — 한계 명시

    if cfg["sample"].get("drop_zero_sales", True):
        for c in ("avg_sales", "avg_sales_per_area"):
            df.loc[df[c] <= 0, c] = float("nan")

    # (brand_id, year) 중복 제거 — 점포수 최대 행 유지, 개수 로깅
    dups = df.duplicated(["brand_id", "year"], keep=False).sum()
    if dups:
        log.warning("(brand_id, year) 중복 %d행 → 점포수 최대 행만 유지", int(dups))
        df = (df.sort_values("n_stores", ascending=False)
                .drop_duplicates(["brand_id", "year"], keep="first"))

    # 브랜드 개요 병합 (업력·임직원수 — 라벨 차원 밖 보조 피처)
    df = _merge_overview(df, cfg)

    cols = ["brand_id", "brand_name", "company_name", "industry_major", "industry_mid",
            "year"] + ["n_stores", "n_direct"] + _NUM_COLS[1:] + \
           ["biz_start_year", "emp_cnt", "exec_cnt"]
    cols = list(dict.fromkeys([c for c in cols if c in df.columns or c in
                               ("n_stores", "n_direct")]))
    panel_full = df[cols].sort_values(["brand_id", "year"]).reset_index(drop=True)

    # 업종 기준선 저장
    _save_industry_baseline(cfg)

    # 표본 필터
    panel = apply_sample_filter(panel_full, cfg)

    proc = cfg["paths"]["processed"]
    panel_full.to_parquet(proc / "panel_full.parquet", index=False)
    panel.to_parquet(proc / "panel.parquet", index=False)
    log.info("panel_full: %d행 %d브랜드 / panel(표본): %d행 %d브랜드",
             len(panel_full), panel_full["brand_id"].nunique(),
             len(panel), panel["brand_id"].nunique())
    return panel


def _merge_overview(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    try:
        rows = load_snapshots(cfg, "brand_overview")
    except Exception:  # noqa: BLE001
        rows = []
    if not rows:
        log.warning("brand_overview 스냅샷 없음 — 업력·임직원수 피처 생략")
        df["biz_start_year"] = float("nan")
        df["emp_cnt"] = float("nan")
        df["exec_cnt"] = float("nan")
        return df
    ov = pd.DataFrame(rows)
    ov["norm_brand"] = ov["brandNm"].map(normalize_name)
    ov["industry_major"] = ov["indutyLclasNm"].astype(str).str.strip()
    ov["biz_start_year"] = pd.to_numeric(
        ov.get("jngBizStrtDate", pd.Series(dtype=str)).astype(str).str[:4], errors="coerce")
    ov["emp_cnt"] = pd.to_numeric(ov.get("empCnt"), errors="coerce")
    ov["exec_cnt"] = pd.to_numeric(ov.get("allExctvCnt"), errors="coerce")
    ov = (ov.groupby(["norm_brand", "industry_major", "_yr"], as_index=False)
            [["biz_start_year", "emp_cnt", "exec_cnt"]].first())
    merged = df.merge(ov, on=["norm_brand", "industry_major", "_yr"], how="left")
    rate = merged["biz_start_year"].notna().mean()
    log.info("brand_overview 병합률: %.1f%%", 100 * rate)
    return merged


def _save_industry_baseline(cfg: dict) -> None:
    try:
        rows = load_snapshots(cfg, "industry_openclose")
    except Exception:  # noqa: BLE001
        return
    if not rows:
        return
    ib = pd.DataFrame(rows)
    ib = ib.rename(columns={
        "tpbizLclsfNm": "industry_major", "tpbizMclsfNm": "industry_mid",
        "newFrcsRt": "ind_new_rate", "endCncltnRt": "ind_end_rate",
        "frcsCnt": "ind_n_stores",
    })
    ib["year"] = pd.to_numeric(ib["jngBizCrtrYr"], errors="coerce").astype("Int64") - 1
    keep = ["year", "industry_major", "industry_mid", "ind_new_rate", "ind_end_rate", "ind_n_stores"]
    ib = ib[[c for c in keep if c in ib.columns]]
    ib.to_parquet(cfg["paths"]["processed"] / "industry_baseline.parquet", index=False)
    log.info("industry_baseline 저장: %d행", len(ib))


def apply_sample_filter(panel_full: pd.DataFrame, cfg: dict, relaxed: bool = False) -> pd.DataFrame:
    s = cfg["sample"]["relaxed"] if relaxed else cfg["sample"]
    majors = s.get("industry_major", cfg["sample"]["industry_major"])
    min_stores = s.get("min_stores", cfg["sample"]["min_stores"])
    min_years = cfg["sample"]["min_consecutive_years"]

    df = panel_full[panel_full["industry_major"].isin(majors)].copy()

    # 브랜드별 최장 연속 관측 구간 산출 → min_years 이상 브랜드만
    def _max_consecutive(years: pd.Series) -> int:
        ys = sorted(set(years))
        best = cur = 1
        for a, b in zip(ys, ys[1:]):
            cur = cur + 1 if b - a == 1 else 1
            best = max(best, cur)
        return best if ys else 0

    runs = df.groupby("brand_id")["year"].agg(_max_consecutive)
    keep_run = set(runs[runs >= min_years].index)

    max_stores = df.groupby("brand_id")["n_stores"].max()
    keep_size = set(max_stores[max_stores >= min_stores].index)

    keep = keep_run & keep_size
    out = df[df["brand_id"].isin(keep)].reset_index(drop=True)
    if relaxed:
        (cfg["paths"]["outputs"] / "sample_relaxed.flag").write_text("relaxed", encoding="utf-8")
    return out


def survival_report(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """생존성 진단표 (M1 DoD): 필터 단계별 표본 수 + (가능 시) 라벨 양성률."""
    proc = cfg["paths"]["processed"]
    full_path = proc / "panel_full.parquet"
    full = pd.read_parquet(full_path) if full_path.exists() else panel

    steps: list[dict] = []

    def _add(step: str, df: pd.DataFrame) -> None:
        steps.append({
            "단계": step,
            "행수": len(df),
            "브랜드수": df["brand_id"].nunique() if len(df) else 0,
            "연도범위": f"{df['year'].min()}~{df['year'].max()}" if len(df) else "-",
            "매출보유율%": round(100 * df["avg_sales"].notna().mean(), 1) if len(df) else 0,
        })

    _add("① 전체(정합 후)", full)
    ext = full[full["industry_major"].isin(cfg["sample"]["industry_major"])]
    _add(f"② 업종 필터({','.join(cfg['sample']['industry_major'])})", ext)
    sample = apply_sample_filter(full, cfg)
    _add(f"③ +점포{cfg['sample']['min_stores']}+ & {cfg['sample']['min_consecutive_years']}년연속", sample)

    report = pd.DataFrame(steps)

    # 라벨 양성률 (labels 모듈이 준비된 경우)
    label_info = ""
    try:
        from src import labels as labels_mod
        lab = labels_mod.build_labels(sample, cfg)
        overall = 100 * lab["label"].mean()
        by_year = (lab.groupby("year")["label"].agg(["count", "mean"])
                   .assign(양성률=lambda d: (100 * d["mean"]).round(1)))
        label_info = f"라벨 표본 {len(lab)}행, 양성률 {overall:.1f}%"
        by_year.to_csv(cfg["paths"]["outputs"] / "label_rate_by_year.csv", encoding="utf-8-sig")
        log.info("연도별 라벨: \n%s", by_year[["count", "양성률"]].to_string())
    except Exception as e:  # noqa: BLE001
        label_info = f"라벨 미산출 (labels 모듈 대기): {e}"
    log.info("생존성 진단: %s", label_info)

    out_path = cfg["paths"]["outputs"] / "survival_report.csv"
    report.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("\n===== 생존성 진단표 (M1 DoD) =====")
    print(report.to_string(index=False))
    print(label_info)
    return report


if __name__ == "__main__":
    cfg = load_config()
    p = build_panel(cfg)
    survival_report(p, cfg)
