"""확률 보정 — 등온회귀 + 계단별 경험적 베이즈 축소.

왜 새로 만드는가 (검증 실패 추적 결과)
    기존 보정은 **워크포워드 fold 예측**으로 적합했다. 그런데 워크포워드는 fold 마다
    모형을 재학습하므로 원점수 척도가 fold 마다 다르다 — 실측 평균 0.2826 / 0.3306 /
    0.1698 로 최대 44.8% 벌어진다. 높은 척도에서 적합한 보정기를 낮은 척도에 적용하면
    등온회귀 곡선의 아래쪽으로 매핑돼 **체계적으로 과소예측**한다.
    실제로 그랬다: 예측 7.47% vs 실제 9.87%, Spiegelhalter Z=+3.41, HL p=0.00027(기각).

    기저율 이동은 원인이 아니었다. 2023 fold 는 선행 기저율(9.90%)이 실제(8.11%)보다
    **높았는데도** 과소예측했다. 척도 이동이 그것을 뒤집을 만큼 컸다.

무엇을 바꿨나
    1. **운영 모형 점수로 보정한다.** 운영은 모형이 하나이므로 척도가 안정적이다
       (실측 연도별 평균 0.1752 / 0.1853 / 0.1698). 대신 학습 연도의 점수는 in-sample
       이라 쓰지 않는다 — 써 봤더니 Z=+17.8 로 훨씬 나빠졌다(과적합된 분리력이 그대로
       보정 곡선에 들어간다).
    2. **계단별 경험적 베이즈 축소.** 등온회귀의 극단 계단은 소표본에서 불안정하다.
       실측: 최하위 구간 예측 0.46% vs 실제 2.10%. 계단 j 의 (n_j, k_j) 를
       `(k_j + α·B) / (n_j + α)` 로 당긴다(B = 보정표본 기저율).
    3. **α 는 보정표본 안에서만 K-겹 교차검증으로 고른다.** 검증 연도를 보고 α 를
       고르면 그건 검정에 하이퍼파라미터를 맞추는 것이다.

왜 순위 기반 보정을 쓰지 않았나
    fold 척도 문제는 순위(백분위)로 보정하면 깨끗이 사라진다. 그러나 그러면
    `p = g(코호트 내 순위)` 가 되어 **등급 컷이 다시 순위 컷이 된다** — 절대 등급으로
    바꾼 이유가 통째로 무효화된다. 그래서 값 기반을 유지하고 척도 쪽을 고쳤다.

결과 (2022 로 적합 → 2023 검증, 둘 다 학습연도 밖)
    예측 10.00% vs 실제 8.11% · HL p=0.158 통과 · Spiegelhalter Z=−1.109 통과
    남은 편차는 **과대예측** 방향이다 — 은행에 안전한 쪽이다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

ALPHAS: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
CV_FOLDS = 5
CV_SEED = 42
_EPS = 1e-6


class ShrunkIsotonic:
    """등온회귀 + 계단별 축소. joblib 으로 저장·복원된다.

    ⚠️ 클래스 참조가 pickle 에 들어가므로 모듈 경로를 옮기면 기존 산출물을 못 읽는다.
       위치를 바꿀 때는 보정기를 다시 만들 것.
    """

    def __init__(self, iso: IsotonicRegression, mapping: dict[float, float],
                 base: float, alpha: float, fitted_on: str, n: int, events: int):
        self.iso = iso
        self.mapping = mapping
        self.base = float(base)
        self.alpha = float(alpha)
        self.fitted_on = str(fitted_on)
        self.n = int(n)
        self.events = int(events)

    def predict(self, x) -> np.ndarray:
        lv = np.round(self.iso.predict(np.asarray(x, dtype=float)), 10)
        return np.clip([self.mapping.get(v, self.base) for v in lv], _EPS, 1.0 - _EPS)

    def describe(self) -> dict:
        return {"method": "isotonic+eb_shrinkage", "alpha": self.alpha,
                "fitted_on": self.fitted_on, "n": self.n, "events": self.events,
                "base_rate": round(self.base, 6), "n_levels": len(self.mapping)}


def _fit_one(x: np.ndarray, y: np.ndarray, alpha: float) -> ShrunkIsotonic:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(x, y)
    base = float(np.mean(y))
    t = (pd.DataFrame({"lv": np.round(iso.predict(x), 10), "y": y})
         .groupby("lv").agg(n=("y", "size"), k=("y", "sum")))
    mp = dict(zip(t.index, (t["k"] + alpha * base) / (t["n"] + alpha), strict=True))
    return ShrunkIsotonic(iso, mp, base, alpha, "", len(y), int(np.sum(y)))


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def choose_alpha(x: np.ndarray, y: np.ndarray) -> tuple[float, dict[float, float]]:
    """보정표본 **안에서만** K-겹 교차검증으로 축소 강도를 고른다.

    검증 연도를 보고 고르면 검정에 하이퍼파라미터를 맞추는 것이 된다.
    """
    kf = KFold(CV_FOLDS, shuffle=True, random_state=CV_SEED)
    scores: dict[float, list[float]] = {a: [] for a in ALPHAS}
    for tr, te in kf.split(x):
        if y[tr].sum() == 0 or y[te].sum() == 0:
            continue
        for a in ALPHAS:
            scores[a].append(_logloss(y[te], _fit_one(x[tr], y[tr], a).predict(x[te])))
    mean = {a: float(np.mean(v)) for a, v in scores.items() if v}
    if not mean:
        return 10.0, {}
    return min(mean, key=lambda k: mean[k]), mean


def fit(x, y, *, fitted_on: str = "") -> tuple[ShrunkIsotonic, dict]:
    """보정기 적합. (보정기, CV 로그손실표) 반환."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    alpha, cv = choose_alpha(x, y)
    cal = _fit_one(x, y, alpha)
    cal.fitted_on = fitted_on
    return cal, {str(k): round(v, 6) for k, v in cv.items()}
