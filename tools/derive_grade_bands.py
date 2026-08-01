"""등급 밴드 도출 — 순위 백분위가 아니라 **관측된 악화율로 앵커링된 절대 컷**.

왜 바꾸는가
    기존 등급은 `pd_rank_pct >= 0.90 → High` 라는 순위 규칙이었다. 그러면 매 산출
    시점에 **항상 정확히 10%가 주의**가 되어, 산업 전체가 나빠져도 등급 분포가 고정된다.
    등급별 확률이 정의되지 않으므로 어떤 검증도 성립하지 않는다.

어떻게 앵커링하는가 (신용평가사 관행)
    평가사는 등급을 규칙이 아니라 **실적표**로 정의한다 — 등급별 장기 평균부도율을
    공표하고, 그 표가 등급의 의미가 된다. 여기서도 같은 방식을 쓴다.
    컷은 한 번 도출해 고정하고, 등급별 실현 악화율을 함께 공표한다.

시점 정합 (오염 방지)
    fold t 의 확률은 **t 이전 fold 의 OOS 로만** 보정한다. 미래를 보고 보정한 확률로
    컷을 정하면 그 컷은 재현되지 않는다. 그래서 2021 fold 는 보정 재료가 없어
    검증표에서 빠진다 — 표본이 줄지만 그게 정직하다.
    배포용 보정기는 워크포워드 OOS 전체(2021~2023)로 적합한다. 모든 fold 가 자기
    학습구간 이후 시점이므로 out-of-time 이고, valid 한 해(689행)보다 3배 두껍다.

컷 선택 규칙 (사후 조정 없이 재현 가능해야 한다)
    격자 탐색으로 아래를 **모두** 만족하는 후보 중 인접 등급 신뢰구간 분리폭이
    가장 큰 것을 고른다.
      1. 등급별 표본 n >= MIN_GRADE_N
      2. 실현 악화율이 등급 순으로 단조 증가
      3. 인접 등급의 Wilson 95% 신뢰구간이 겹치지 않음
      4. 검증에 쓰는 **모든 연도에서** 단조성이 유지됨

⚠️ 등급 3개인 이유
    4등급 이상은 진짜 OOS(보정연도 제외)에서 인접 신뢰구간이 겹치고, 연도를 하나씩
    빼고 재도출하면 최상위 등급이 붕괴한다(실측). 평가사도 18~20 notch 로 부여하되
    부도율 통계는 6개 버킷으로만 공표한다 — 표본이 지지하지 않는 세분화는 접는다.
"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.isotonic import IsotonicRegression

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

for _s in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, OSError):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src.common import get_logger, load_config  # noqa: E402

log = get_logger("bands")

GRADES = ["FS1", "FS2", "FS3"]
GRADE_KR = {"FS1": "안정", "FS2": "관찰", "FS3": "주의"}
MIN_GRADE_N = 150          # 등급별 최소 표본 (풀링 기준)
MIN_CAL_N = 200            # 보정에 필요한 최소 선행 표본
GRID = np.round(np.arange(0.02, 0.32, 0.005), 4)


def wilson(k: int, n: int) -> tuple[float, float]:
    """Clopper-Pearson 95% 구간 (소표본에서 보수적)."""
    if n == 0:
        return (0.0, 0.0)
    lo = float(beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
    hi = float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
    return lo, hi


def time_consistent_oos(wf: pd.DataFrame) -> pd.DataFrame:
    """fold 별로 **선행 fold 만으로** 보정한 OOS 확률을 만든다."""
    out = []
    for y in sorted(wf["year"].unique()):
        prior = wf[wf["year"] < y]
        if len(prior) < MIN_CAL_N:
            log.info("fold %s: 선행 보정표본 %d행 (< %d) → 검증표 제외",
                     y, len(prior), MIN_CAL_N)
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(prior["p_lgbm"].to_numpy(), prior["y_true"].to_numpy())
        cur = wf[wf["year"] == y].copy()
        cur["p_cal"] = iso.predict(cur["p_lgbm"].to_numpy())
        out.append(cur)
        log.info("fold %s: 선행 %d행으로 보정 → 적용 %d행 (예측평균 %.4f, 실제 %.4f)",
                 y, len(prior), len(cur), cur["p_cal"].mean(), cur["y_true"].mean())
    if not out:
        raise SystemExit("검증 가능한 fold 가 없습니다 — 워크포워드 산출물을 확인하세요")
    return pd.concat(out, ignore_index=True)


def search_cuts(oos: pd.DataFrame) -> tuple[list[float], list[dict], float]:
    """규칙을 만족하는 컷 중 인접 구간 분리폭이 가장 큰 것."""
    y = oos["y_true"].to_numpy()
    p = oos["p_cal"].to_numpy()
    years = sorted(oos["year"].unique())
    best = None
    for c1 in GRID:
        for c2 in GRID[c1 < GRID]:
            g = np.digitize(p, [c1, c2])
            rows, ok = [], True
            for i in range(3):
                m = g == i
                n, k = int(m.sum()), int(y[m].sum())
                if n < MIN_GRADE_N:
                    ok = False
                    break
                lo, hi = wilson(k, n)
                rows.append({"grade": GRADES[i], "n": n, "events": k,
                             "rate": k / n, "ci_lo": lo, "ci_hi": hi})
            if not ok or not (rows[0]["rate"] < rows[1]["rate"] < rows[2]["rate"]):
                continue
            if not (rows[0]["ci_hi"] < rows[1]["ci_lo"] and rows[1]["ci_hi"] < rows[2]["ci_lo"]):
                continue
            if not all(
                (lambda r: r[0] < r[1] < r[2])(
                    [float(y[(g == i) & (oos["year"].to_numpy() == yr)].mean())
                     for i in range(3)])
                for yr in years
            ):
                continue
            sep = min(rows[1]["ci_lo"] - rows[0]["ci_hi"],
                      rows[2]["ci_lo"] - rows[1]["ci_hi"])
            if best is None or sep > best[2]:
                best = ([float(c1), float(c2)], rows, float(sep))
    if best is None:
        raise SystemExit(
            "조건을 만족하는 3등급 컷이 없습니다. 표본이 등급 분리를 지지하지 않으므로 "
            "등급 수를 줄이거나 순위 등급을 유지해야 합니다 — 억지로 만들지 마십시오.")
    return best


def per_year_table(oos: pd.DataFrame, cuts: list[float]) -> pd.DataFrame:
    g = np.digitize(oos["p_cal"].to_numpy(), cuts)
    rows = []
    for yr in sorted(oos["year"].unique()):
        for i in range(3):
            m = (g == i) & (oos["year"].to_numpy() == yr)
            n, k = int(m.sum()), int(oos["y_true"].to_numpy()[m].sum())
            lo, hi = wilson(k, n)
            rows.append({"year": int(yr), "grade": GRADES[i], "grade_kr": GRADE_KR[GRADES[i]],
                         "n": n, "events": k, "realized_rate": round(k / n, 6) if n else None,
                         "ci_lo": round(lo, 6), "ci_hi": round(hi, 6)})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    out_dir = Path(cfg["paths"]["outputs"])
    wf_p = out_dir / "walkforward_predictions.parquet"
    if not wf_p.exists():
        raise SystemExit(f"{wf_p} 없음 — `python run_pipeline.py --step evaluate` 를 먼저 실행")
    wf = pd.read_parquet(wf_p)

    oos = time_consistent_oos(wf)
    log.info("검증용 OOS: %d행 · 악화 %d건 (%.2f%%)",
             len(oos), int(oos["y_true"].sum()), oos["y_true"].mean() * 100)

    cuts, rows, sep = search_cuts(oos)
    log.info("채택 컷 %s · 인접 신뢰구간 최소 분리 %.2f%%p", cuts, sep * 100)
    for r in rows:
        log.info("  %s %s  n=%d  악화 %d건  실현율 %.2f%%  95%%CI [%.1f, %.1f]",
                 r["grade"], GRADE_KR[r["grade"]], r["n"], r["events"],
                 r["rate"] * 100, r["ci_lo"] * 100, r["ci_hi"] * 100)

    # 배포용 보정기 — 워크포워드 OOS 전체로 적합 (모든 fold 가 out-of-time)
    deploy = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    deploy.fit(wf["p_lgbm"].to_numpy(), wf["y_true"].to_numpy())
    import joblib
    joblib.dump({"model": deploy, "method": "isotonic",
                 "fitted_on": "walkforward_oos_2021_2023", "n": len(wf)},
                out_dir / "calibrator_deploy.joblib")

    per_year = per_year_table(oos, cuts)
    per_year.to_csv(out_dir / "grade_validation.csv", index=False, encoding="utf-8-sig")

    bands = {
        "cuts": cuts,
        "grades": GRADES,
        "grade_kr": GRADE_KR,
        "anchored_on": "walkforward OOS, fold별 선행 fold 보정 (시점 정합)",
        "validation_years": [int(y) for y in sorted(oos["year"].unique())],
        "excluded_years": [int(y) for y in sorted(wf["year"].unique())
                           if y not in set(oos["year"].unique())],
        "n_validation": len(oos),
        "n_events": int(oos["y_true"].sum()),
        "min_ci_separation": round(sep, 6),
        "pooled": [{**r, "grade_kr": GRADE_KR[r["grade"]],
                    "rate": round(r["rate"], 6),
                    "ci_lo": round(r["ci_lo"], 6), "ci_hi": round(r["ci_hi"], 6)}
                   for r in rows],
        "deploy_calibrator": "calibrator_deploy.joblib",
        "label_note": ("악화 = 공정거래위원회 가맹사업 공시 지표(가맹점수·계약종료·"
                       "평균매출·면적당매출)가 업종×연도 하위 구간에 진입하는 사건이다. "
                       "차주의 채무불이행 확률이 아니다."),
        "rule": {"min_grade_n": MIN_GRADE_N,
                 "criteria": ["등급별 최소 표본", "실현율 단조 증가",
                              "인접 Clopper-Pearson 95% 구간 비중첩",
                              "검증 전 연도에서 단조성 유지"]},
    }
    (out_dir / "grade_bands.json").write_text(
        json.dumps(bands, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("저장: grade_bands.json · grade_validation.csv · calibrator_deploy.joblib")
    print(json.dumps({k: v for k, v in bands.items() if k != "pooled"},
                     ensure_ascii=False, indent=2))
    print(per_year.to_string(index=False))


if __name__ == "__main__":
    main()
