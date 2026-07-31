"""FranSCORE 파이프라인 러너.

사용법 (프로젝트 루트에서):
    python run_pipeline.py --step all            # M1→M2→M3→M4 전체
    python run_pipeline.py --step collect        # 개별 스텝
    python run_pipeline.py --step panel|features|labels|model|evaluate|portfolio|correlation|news
    python run_pipeline.py --step all --demo     # 합성 패널 스모크 (outputs/_smoke 격리)

각 스텝은 파일 기반 중간 산출물로 독립 실행 가능하다.
--demo 는 제출 지표를 오염시키지 않도록 outputs/_smoke 로 격리된다.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time

import pandas as pd

from src.common import get_logger, load_config, make_synthetic_panel, set_seed

log = get_logger("pipeline")

STEPS = ["collect", "panel", "features", "labels", "model", "evaluate", "portfolio",
         "correlation", "news"]


def _extended_cfg(cfg: dict) -> dict:
    """확장 표본 트랙(명세 §11 플랜B): 전 업종·점포20+. 산출물은 outputs/extended/ 로 격리.

    ⚠️ 체리피킹 방지 — 기본 트랙과 **둘 다** 공개하는 것을 전제로 한 사전 선언 트랙이다.
    """
    ext = copy.deepcopy(cfg)
    rel = cfg["sample"]["relaxed"]
    ext["sample"]["industry_major"] = list(rel["industry_major"])
    ext["sample"]["min_stores"] = int(rel["min_stores"])
    base_out = cfg["_root"] / "outputs" / "extended"
    ext["paths"]["outputs"] = base_out
    ext["paths"]["processed"] = base_out / "processed"
    for p in (ext["paths"]["outputs"], ext["paths"]["processed"]):
        p.mkdir(parents=True, exist_ok=True)
    return ext


def _demo_cfg(cfg: dict) -> dict:
    """--demo: 산출물 경로를 스모크 디렉토리로 격리한 cfg 사본."""
    demo = copy.deepcopy(cfg)
    smoke = cfg["_root"] / "outputs" / "_smoke"
    demo["paths"]["outputs"] = smoke
    demo["paths"]["processed"] = smoke / "processed"
    for p in (demo["paths"]["outputs"], demo["paths"]["processed"]):
        p.mkdir(parents=True, exist_ok=True)
    return demo


def _load_panel(cfg: dict, demo: bool) -> pd.DataFrame:
    if demo:
        return make_synthetic_panel(cfg)
    panel_path = cfg["paths"]["processed"] / "panel.parquet"
    if not panel_path.exists():
        log.error("panel.parquet 없음 — 먼저 `--step panel` 실행 필요: %s", panel_path)
        sys.exit(1)
    return pd.read_parquet(panel_path)


def run_step(step: str, cfg: dict, demo: bool) -> None:
    t0 = time.time()
    log.info("=== step: %s%s ===", step, " (DEMO/합성)" if demo else "")

    if step == "collect":
        if demo:
            log.info("demo 모드에서는 수집 생략 (합성 패널 사용)")
            return
        from src import collect
        collect.collect_all(cfg)

    elif step == "panel":
        if demo:
            panel = make_synthetic_panel(cfg)
            panel.to_parquet(cfg["paths"]["processed"] / "panel.parquet", index=False)
            from src import panel as panel_mod
            panel_mod.survival_report(panel, cfg)
        else:
            from src import entity
            from src import panel as panel_mod
            master = entity.build_master(cfg)
            panel = panel_mod.build_panel(cfg, master)
            panel_mod.survival_report(panel, cfg)

    elif step == "features":
        from src import features
        panel = _load_panel(cfg, demo)
        feats = features.build_features(panel, cfg)
        feats.to_parquet(cfg["paths"]["processed"] / "features.parquet", index=False)
        log.info("features: %s rows × %s cols", *feats.shape)

    elif step == "labels":
        from src import labels
        panel = _load_panel(cfg, demo)
        # 분위수 풀은 업종 범위 전체 패널로 계산하고,
        # 학습/평가 표본은 실시간 자격(eligible_t) 행만 남긴다 (룩어헤드 선별 방지).
        lab = labels.build_labels(panel, cfg)
        if "eligible_t" in panel.columns:
            elig = panel[["brand_id", "year", "eligible_t"]]
            n0 = len(lab)
            lab = (lab.merge(elig, on=["brand_id", "year"], how="left"))
            lab = lab[lab["eligible_t"].fillna(False)].drop(columns=["eligible_t"])
            log.info("labels: 실시간 자격 필터 %d → %d행", n0, len(lab))
        lab.to_parquet(cfg["paths"]["processed"] / "labels.parquet", index=False)
        log.info("labels: %s rows, 양성률 %.1f%%", len(lab), 100 * lab["label"].mean())

    elif step == "model":
        from src import model
        feats = pd.read_parquet(cfg["paths"]["processed"] / "features.parquet")
        lab = pd.read_parquet(cfg["paths"]["processed"] / "labels.parquet")
        model.train_all(feats, lab, cfg)

    elif step == "evaluate":
        from src import evaluate
        evaluate.evaluate_all(cfg)

    elif step == "portfolio":
        from src import portfolio
        portfolio.build_portfolio(cfg)

    elif step == "correlation":
        # 브랜드 공통요인 상관 실증 (portfolio 이후 — 손실 영향 계산에 exposure·PD가 필요)
        if demo:
            log.info("demo 모드에서는 상관 실증 생략 (지역 원본 스냅샷 필요)")
            return
        from src import correlation
        correlation.run(cfg)

    elif step == "news":
        from src import news_llm
        preds_path = cfg["paths"]["processed"] / "predictions.parquet"
        panel = _load_panel(cfg, demo)
        if preds_path.exists():
            preds = pd.read_parquet(preds_path)
            col = "p_calibrated" if "p_calibrated" in preds.columns else "p_lgbm"
            top = (preds[preds["split"] == "test"].sort_values(col, ascending=False)
                   .head(15)["brand_id"].tolist())
            names = (panel[panel["brand_id"].isin(top)][["brand_id", "brand_name"]]
                     .drop_duplicates("brand_id")["brand_name"].tolist())
        else:
            names = panel["brand_name"].drop_duplicates().head(10).tolist()
        news_llm.run_news(names, cfg)

    else:
        raise ValueError(f"unknown step: {step}")

    log.info("=== step %s 완료 (%.1fs) ===", step, time.time() - t0)


def main() -> None:
    ap = argparse.ArgumentParser(description="FranSCORE pipeline")
    ap.add_argument("--step", default="all", choices=[*STEPS, "all"])
    ap.add_argument("--demo", action="store_true", help="합성 패널 스모크 (outputs/_smoke 격리)")
    ap.add_argument("--scope", default="primary", choices=["primary", "extended"],
                    help="primary=명세 표본(외식·점포30+) / extended=플랜B 확장(전업종·점포20+)")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    if args.scope == "extended":
        cfg = _extended_cfg(cfg)
        log.info("확장 표본 트랙(플랜B): 업종=%s, 점포하한=%s → 산출물 %s",
                 cfg["sample"]["industry_major"], cfg["sample"]["min_stores"],
                 cfg["paths"]["outputs"])
    if args.demo:
        cfg = _demo_cfg(cfg)

    steps = STEPS if args.step == "all" else [args.step]
    if args.step == "all":
        # news는 네트워크·LLM 의존이라 all에서는 마지막·실패 허용
        for s in [s for s in steps if s != "news"]:
            run_step(s, cfg, args.demo)
        try:
            run_step("news", cfg, args.demo)
        except Exception as e:
            log.warning("news 스텝 실패 (전체 파이프라인은 유효): %s", e)
    else:
        run_step(args.step, cfg, args.demo)


if __name__ == "__main__":
    main()
