"""M3.5 운영 점수 산출 (serving) — **아직 라벨이 없는 최신 코호트**를 점수화한다.

왜 이 모듈이 필요한가 (자체 감사 critical 지적):
    model.py 는 features × labels 를 inner join 하므로 **라벨이 확정된 과거 연도만** 점수를
    갖는다. 라벨은 t+1년 실적으로 만들어지고 그 실적은 t+2년 공시로 확정되므로, 정의상
    **가장 최근 연도에는 라벨이 없다** — 그런데 은행이 정작 알고 싶은 것은 바로 그 최신
    코호트다. 이 경로가 없으면 이 저장소는 예측기가 아니라 **백테스트 하네스**에 그친다.

    이 모듈은 학습을 하지 않는다. 학습이 남긴 산출물만 **읽어서** 점수를 낸다:
      outputs/model_lgbm.txt   (LightGBM booster)
      outputs/calibrator.joblib (valid에서 적합된 보정기 + 방법)
      outputs/split_years.json  (피처명 ASCII 치환 매핑 — 학습 때와 같은 이름으로 정렬)
    따라서 운영에서 재학습 없이 매 공시 주기마다 이 스텝만 돌리면 된다.

⛔ 시점 안전:
    점수 대상 행의 피처는 features.py 계약상 **t 이하 연도만** 사용한다(누출 테스트로 보증).
    보정기는 과거 valid에서 적합된 것을 **그대로 적용만** 한다(재적합 금지 — 재적합하면
    최신 코호트의 결과를 보고 보정하는 셈이라 실전에서 불가능한 정보를 쓰게 된다).

산출: outputs/scores_latest.csv  (심사역이 그대로 받아 쓰는 점검 큐)
      brand_id·brand_name·year·pd_1y·risk_grade·pd_rank_pct·n_stores·상위 위험요인 3종

실행: 프로젝트 루트에서 `python -m src.score`  또는  `python run_pipeline.py --step score`
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.common import get_logger, load_config, set_seed
from src.evaluate import _apply_calibrator
from src.feature_names import feature_korean_name
from src.model import _numeric_feature_frame, _sanitize_feature_names

log = get_logger("score")

TOP_FACTORS = 3   # 심사역 화면·CSV에 싣는 위험요인 개수


def _load_artifacts(out_dir: Path) -> tuple[lgb.Booster, dict, dict]:
    """학습이 남긴 산출물 로드 (booster, 보정기 번들, 피처명 매핑)."""
    mp = out_dir / "model_lgbm.txt"
    if not mp.exists():
        raise FileNotFoundError(f"{mp} 없음 — 먼저 `--step model` 실행")
    booster = lgb.Booster(model_str=mp.read_text(encoding="utf-8"))

    cal_path = out_dir / "calibrator.joblib"
    cal = joblib.load(cal_path) if cal_path.exists() else {"method": "identity", "model": None}
    if not cal_path.exists():
        log.warning("calibrator.joblib 없음 — 원점수를 그대로 사용(보정 미적용)")

    name_map: dict[str, str] = {}
    sy = out_dir / "split_years.json"
    if sy.exists():
        name_map = json.loads(sy.read_text(encoding="utf-8")).get("feature_name_map") or {}
    return booster, cal, name_map


def score_latest(cfg: dict, year: int | None = None) -> pd.DataFrame:
    """최신(또는 지정) 연도의 자격 브랜드를 점수화해 outputs/scores_latest.csv 로 저장."""
    set_seed(cfg["seed"])
    proc, out_dir = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["outputs"])
    booster, cal, name_map = _load_artifacts(out_dir)

    feats = pd.read_parquet(proc / "features.parquet")
    panel = pd.read_parquet(proc / "panel.parquet")
    keep = [c for c in ("brand_id", "year", "brand_name", "industry_major", "industry_mid",
                        "n_stores", "avg_sales", "eligible_t") if c in panel.columns]
    df = feats.merge(panel[keep], on=["brand_id", "year"], how="left")

    target_year = int(year if year is not None else df["year"].max())
    cohort = df[df["year"] == target_year].copy()
    if "eligible_t" in cohort.columns:
        n0 = len(cohort)
        cohort = cohort[cohort["eligible_t"].fillna(False).astype(bool)]
        log.info("자격 필터: %d → %d행 (점포 %s+ 누적 & %s년 연속 — 과거 정보만 사용)",
                 n0, len(cohort), cfg["sample"]["min_stores"],
                 cfg["sample"]["min_consecutive_years"])
    if cohort.empty:
        raise ValueError(f"{target_year}년 자격 브랜드가 없습니다 (features/panel 확인)")

    # 학습 때와 **완전히 같은** 피처 행렬 구성 (이름·순서·dtype)
    X_orig = _numeric_feature_frame(cohort, log)
    safe, _ = _sanitize_feature_names(X_orig.columns)
    X = X_orig.set_axis(safe, axis=1)
    trained = list(booster.feature_name())
    missing = [c for c in trained if c not in X.columns]
    extra = [c for c in X.columns if c not in trained]
    if missing:
        # 결측 피처를 0으로 채우면 조용히 틀린 점수가 나간다 — 명시적으로 실패시킨다.
        raise ValueError(f"학습 피처 {len(missing)}개가 현재 피처에 없습니다: {missing[:8]} "
                         f"— features 재생성 또는 모델 재학습이 필요합니다.")
    if extra:
        log.info("학습에 없던 피처 %d개는 제외: %s", len(extra), extra[:5])
    X = X[trained]

    p_raw = np.clip(booster.predict(X), 0.0, 1.0)
    p_cal = _apply_calibrator(cal.get("model"), p_raw, str(cal.get("method", "identity")), cfg)
    # 보정 계단으로 생기는 동률을 원점수 순위로 미세 분리 (evaluate.py 와 동일 규칙)
    tb = pd.Series(p_raw).rank(method="first").to_numpy(dtype=float)
    p_cal = np.clip(p_cal + tb / max(len(tb), 1) * 1e-5, 0.0, 1.0)

    res = cohort[["brand_id", "year"]].copy()
    for c in ("brand_name", "industry_major", "industry_mid", "n_stores"):
        if c in cohort.columns:
            res[c] = cohort[c].to_numpy()
    res["pd_1y"] = p_cal
    res["pd_raw"] = p_raw
    res["pd_rank_pct"] = res["pd_1y"].rank(method="first", pct=True)

    # 등급: 학습·평가와 동일한 **순위 기반** 규칙 (값 임계 컷은 동률에 취약 — 기존 결함 수정 유지)
    g = cfg["portfolio"]["risk_grades"]
    hi, mid = float(g["high"]), float(g["medium"])
    res["risk_grade"] = np.where(res["pd_rank_pct"] >= hi, "High",
                                 np.where(res["pd_rank_pct"] >= mid, "Medium", "Low"))

    # 상위 위험요인 (SHAP) — 심사역이 '왜'를 바로 볼 수 있게 CSV에 함께 싣는다
    try:
        contrib = booster.predict(X, pred_contrib=True)[:, :-1]  # 마지막 열은 base value
        order = np.argsort(-np.abs(contrib), axis=1)[:, :TOP_FACTORS]
        cols = np.array(trained)
        for i in range(TOP_FACTORS):
            idx = order[:, i]
            # 원시 피처명(f_ind_contract_end_pct)이 그대로 나가면 아무도 읽을 수 없다 →
            # 화면·CSV·메모가 공유하는 한국어 표시명으로 변환해서 저장한다.
            res[f"factor{i + 1}"] = [feature_korean_name(name_map.get(str(c), str(c)))
                                     for c in cols[idx]]
            res[f"factor{i + 1}_shap"] = np.round(contrib[np.arange(len(idx)), idx], 5)
    except Exception as exc:
        log.warning("SHAP 기여도 산출 생략: %s", exc)

    res = res.sort_values("pd_1y", ascending=False).reset_index(drop=True)
    dest = out_dir / "scores_latest.csv"
    res.to_csv(dest, index=False, encoding="utf-8-sig")

    meta = {
        "scored_year": target_year,
        "n_scored": len(res),
        "model_file": "model_lgbm.txt",
        "calibration_method": str(cal.get("method")),
        "calibration_fitted_on": str(cal.get("fitted_on", "valid")),
        "pd_floor": float((cfg.get("evaluate") or {}).get("pd_floor", 0.0)),
        "pd_cap": float((cfg.get("evaluate") or {}).get("pd_cap", 1.0)),
        "grade_counts": res["risk_grade"].value_counts().to_dict(),
        "pd_min": float(res["pd_1y"].min()), "pd_max": float(res["pd_1y"].max()),
        "note": ("라벨 미확정 코호트의 사전 점수. 보정기는 과거 valid 적합본을 적용만 했으며 "
                 "재적합하지 않았다. 성능 근거는 metrics.csv·walkforward_metrics.csv 참조."),
    }
    (out_dir / "scores_latest_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("운영 점수 산출: %d년 %d개 브랜드 → %s (High %d / Medium %d / Low %d), "
             "PD %.4f~%.4f", target_year, len(res), dest.name,
             meta["grade_counts"].get("High", 0), meta["grade_counts"].get("Medium", 0),
             meta["grade_counts"].get("Low", 0), meta["pd_min"], meta["pd_max"])
    log.info("⚠️ 이 코호트는 아직 라벨이 없어 사후 검증이 불가하다 — 성능 근거는 "
             "과거 백테스트(metrics.csv·walkforward_metrics.csv)에 있다.")
    return res


if __name__ == "__main__":
    score_latest(load_config())
