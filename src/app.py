"""FranSCORE Streamlit 대시보드 (M5 — INTERFACES.md §5).

파이프라인 산출물 파일만 읽어 표시한다(재계산 없음). 모든 파일 로드는 방어적으로
처리하여 파일이 없으면 어떤 스텝을 실행해야 하는지 한국어로 안내하고, 앱은 절대
크래시하지 않는다.

실행: streamlit run src/app.py  (프로젝트 루트에서)

읽는 산출물(계약 §5 — 재계산 없음):
  data/processed/panel.parquet, data/processed/predictions.parquet,
  outputs/shap_values.parquet, outputs/portfolio.csv, outputs/portfolio_summary.json,
  outputs/news_signals.json, outputs/metrics.csv, outputs/split_years.json

주의: 경로는 항상 호출 시점에 cfg['paths']로 해석한다 (데모 러너가 스모크 격리를
위해 paths를 교체할 수 있음).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 반드시 pyplot import 전에 (Windows/서버 환경)
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# 프로젝트 루트를 sys.path에 보장 (streamlit run src/app.py 대응)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.common import get_logger, load_config, set_seed  # noqa: E402

import streamlit as st  # noqa: E402

_LOG = get_logger("app")

# ---------------------------------------------------------------------------
# 산출물 레지스트리: key -> (cfg['paths'] 키, 파일명, 종류, 생성 스텝 안내)
# ---------------------------------------------------------------------------
_ARTIFACTS: dict[str, dict] = {
    "panel": {
        "paths_key": "processed", "filename": "panel.parquet",
        "kind": "parquet", "step": "panel",
    },
    "predictions": {
        "paths_key": "processed", "filename": "predictions.parquet",
        "kind": "parquet", "step": "model",
    },
    "shap_values": {
        "paths_key": "outputs", "filename": "shap_values.parquet",
        "kind": "parquet", "step": "evaluate",
    },
    "portfolio": {
        "paths_key": "outputs", "filename": "portfolio.csv",
        "kind": "csv", "step": "portfolio",
    },
    "portfolio_summary": {
        "paths_key": "outputs", "filename": "portfolio_summary.json",
        "kind": "json", "step": "portfolio",
    },
    "news_signals": {
        "paths_key": "outputs", "filename": "news_signals.json",
        "kind": "json", "step": "news",
    },
    "metrics": {
        "paths_key": "outputs", "filename": "metrics.csv",
        "kind": "csv", "step": "evaluate",
    },
    "split_years": {
        "paths_key": "outputs", "filename": "split_years.json",
        "kind": "json", "step": "model",
    },
}

GRADE_COLORS = {"High": "#d9534f", "Medium": "#f0ad4e", "Low": "#5cb85c"}

GRADE_RECO = {
    "High": "**신규취급 강화검토 · 집중한도 점검**을 권고합니다.",
    "Medium": "**모니터링 강화**를 권고합니다.",
    "Low": "**정기 모니터링**을 유지합니다.",
}

_DISCLAIMER = "본 지표는 2선 리스크 관리 참고용이며 자동 여신 결정에 사용되지 않습니다."
_SYNTH_WARN = "여신 익스포저는 합성 예시입니다 (방법론 실증용)"
_NEWS_CAPTION = "점수 미반영 · 사실관계 별도 확인 필요"

_EVENT_COLORS = {
    "본부분쟁": "#d9534f", "재무이슈": "#f0ad4e",
    "집단폐점": "#8e44ad", "기타": "#7f8c8d", "무관": "#bdc3c7",
}

# ---------------------------------------------------------------------------
# f_* 피처명 -> 한국어 표시명 매핑 (정확 매칭 우선, 이하 토큰 규칙 폴백)
# ---------------------------------------------------------------------------
_FEATURE_KR_EXACT = {
    "f_lvl_n_stores": "[수준] 가맹점 수",
    "f_lvl_avg_sales": "[수준] 평균매출(log)",
    "f_lvl_avg_sales_log1p": "[수준] 평균매출(log)",
    "f_lvl_direct_ratio": "[수준] 직영점 비중",
    "f_lvl_brand_age": "[수준] 브랜드 연차",
    "f_chg_store_growth_rate": "[변화] 점포 증가율",
    "f_chg_sales_growth": "[변화] 매출 증가율",
    "f_chg_contract_end_rate": "[변화] 계약종료율",
    "f_chg_new_open_rate": "[변화] 신규개점률",
    "f_chg_name_change_rate": "[변화] 명의변경률",
}

_PREFIX_KR = [
    ("f_lvl_", "[수준] "),
    ("f_chg_", "[변화] "),
    ("f_trd_", "[추세] "),
    ("f_ind_", "[업종상대] "),
    ("f_struct_", "[구조] "),
]

# 긴 토큰 우선 (부분문자열 우선순위)
_TOKEN_KR = [
    ("real_sales_growth", "실질 매출 증가율"),
    ("store_growth_rate", "점포 증가율"),
    ("contract_end_rate", "계약종료율"),
    ("store_growth", "점포 증가율"),
    ("sales_growth", "매출 증가율"),
    ("contract_end", "계약종료"),
    ("direct_ratio", "직영점 비중"),
    ("brand_age", "브랜드 연차"),
    ("name_change", "명의변경"),
    ("new_open", "신규개점"),
    ("open_close_gap", "개점-종료 격차"),
    ("n_stores", "가맹점 수"),
    ("n_direct", "직영점 수"),
    ("avg_sales", "평균매출"),
    ("n_new", "신규개점"),
    ("major", "업종 대분류"),
    ("gap", "격차"),
]

_QUAL_KR = [
    ("mean", "평균"),
    ("slope", "기울기"),
    ("std", "변동성"),
    ("rank_pct", "업종 내 백분위"),
    ("rank", "업종 내 백분위"),
    ("pct", "백분위"),
    ("diff", "변화"),
    ("2y", "최근 2년"),
    ("3y", "최근 3년"),
    ("log1p", ""),
    ("log", ""),
]


def feature_korean_name(feat: str) -> str:
    """f_* 피처명을 한국어 표시명으로 변환. 미지의 이름도 안전하게 처리."""
    if feat in _FEATURE_KR_EXACT:
        return _FEATURE_KR_EXACT[feat]
    name = str(feat)
    prefix_kr = ""
    for pre, kr in _PREFIX_KR:
        if name.startswith(pre):
            prefix_kr, name = kr, name[len(pre):]
            break
    base_kr, rest = None, name
    for token, tkr in _TOKEN_KR:
        if token in name:
            base_kr = tkr
            rest = name.replace(token, "", 1)
            break
    if base_kr is None:
        return prefix_kr + name
    quals = []
    for token, tkr in _QUAL_KR:
        if token and token in rest and tkr:
            quals.append(tkr)
            rest = rest.replace(token, "", 1)
    label = base_kr + ((" " + "·".join(quals)) if quals else "")
    return prefix_kr + label


# ---------------------------------------------------------------------------
# 순수 로더 (streamlit 비의존 — 단독 파이썬에서 테스트 가능)
# ---------------------------------------------------------------------------

def _read_parquet_file(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _read_csv_file(path: str | Path) -> pd.DataFrame:
    # utf-8-sig는 BOM 유무와 무관하게 UTF-8을 안전하게 읽는다 (Windows 대응)
    return pd.read_csv(path, encoding="utf-8-sig")


def _read_json_file(path: str | Path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def artifact_path(cfg: dict, key: str) -> Path:
    """호출 시점에 cfg['paths']로 산출물 경로 해석 (데모 러너 경로 교체 대응)."""
    spec = _ARTIFACTS[key]
    return Path(cfg["paths"][spec["paths_key"]]) / spec["filename"]


def pick_prob_column(df: pd.DataFrame) -> tuple[str, bool]:
    """(사용할 확률 컬럼명, 보정확률 여부). p_calibrated 우선."""
    if "p_calibrated" in df.columns and df["p_calibrated"].notna().any():
        return "p_calibrated", True
    return "p_lgbm", False


def assign_grade(probs: pd.Series, cuts: dict) -> pd.Series:
    """확률 분위수 컷(config portfolio.risk_grades)으로 High/Medium/Low 부여."""
    high_q = float(cuts.get("high", 0.90))
    med_q = float(cuts.get("medium", 0.70))
    p = probs.astype(float)
    hi_cut = p.quantile(high_q)
    md_cut = p.quantile(med_q)
    out = pd.Series("Low", index=p.index, dtype=object)
    out[p >= md_cut] = "Medium"
    out[p >= hi_cut] = "High"
    return out


def parse_news_signals(obj) -> tuple[list[dict], bool]:
    """news_signals.json 형태(리스트 or {signals: [...], llm_used: bool})를 정규화."""
    if obj is None:
        return [], False
    if isinstance(obj, list):
        sigs = [s for s in obj if isinstance(s, dict)]
        llm_used = any(bool(s.get("llm_used")) for s in sigs)
        return sigs, llm_used
    if isinstance(obj, dict):
        llm_used = bool(obj.get("llm_used", False))
        for k in ("signals", "items", "results", "data"):
            if isinstance(obj.get(k), list):
                return [s for s in obj[k] if isinstance(s, dict)], llm_used
        # 브랜드명 -> 리스트 매핑 형태
        sigs = []
        for v in obj.values():
            if isinstance(v, list):
                sigs.extend(s for s in v if isinstance(s, dict))
        return sigs, llm_used
    return [], False


def find_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """후보명(소문자 정확일치 → 부분일치 순)으로 실제 컬럼명 탐색."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for cand in candidates:
        for lc, orig in lower.items():
            if cand.lower() in lc:
                return orig
    return None


def pick_el_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(LGD 시나리오 EL 컬럼들, 스트레스 EL 컬럼들)."""
    el_cols, stress_cols = [], []
    for c in df.columns:
        lc = c.lower()
        if "stress" in lc and "el" in lc:
            stress_cols.append(c)
        elif lc.startswith("el") or "_el" in lc or lc.startswith("expected_loss"):
            el_cols.append(c)
    return el_cols, stress_cols


def to_eok(mkrw) -> float:
    """백만원 -> 억원."""
    try:
        return float(mkrw) / 100.0
    except (TypeError, ValueError):
        return float("nan")


def flatten_numeric(obj, prefix: str = "") -> dict[str, float]:
    """중첩 dict/list에서 숫자 리프만 {경로: 값}으로 평탄화 (요약 JSON 방어적 탐색)."""
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten_numeric(v, f"{prefix}{k}."))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(flatten_numeric(v, f"{prefix}{i}."))
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out[prefix.rstrip(".")] = float(obj)
    return out


def find_num(flat: dict[str, float], include: tuple[str, ...],
             exclude: tuple[str, ...] = ()) -> float | None:
    """평탄화된 숫자 dict에서 경로에 include 전부 포함·exclude 미포함인 첫 값."""
    for path, val in flat.items():
        lp = path.lower()
        if all(tok in lp for tok in include) and not any(tok in lp for tok in exclude):
            return val
    return None


# ---------------------------------------------------------------------------
# streamlit 캐시 로더
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _cached_load(kind: str, path_str: str, mtime: float):
    """mtime을 캐시 키에 포함 → 파일 갱신 시 자동 무효화."""
    if kind == "parquet":
        return _read_parquet_file(path_str)
    if kind == "csv":
        return _read_csv_file(path_str)
    if kind == "json":
        return _read_json_file(path_str)
    raise ValueError(f"unknown kind: {kind}")


def load_artifact(key: str):
    """산출물 로드. 없거나 읽기 실패 시 None (호출부에서 안내 표시)."""
    cfg = load_config()
    p = artifact_path(cfg, key)
    if not p.exists():
        return None
    try:
        return _cached_load(_ARTIFACTS[key]["kind"], str(p), p.stat().st_mtime)
    except Exception as e:  # noqa: BLE001 — 대시보드는 절대 크래시 금지
        _LOG.warning("artifact load failed: %s (%s)", p, e)
        return None


def missing_notice(key: str) -> None:
    spec = _ARTIFACTS[key]
    st.info(
        f"'{spec['filename']}' 파일이 아직 없습니다. 프로젝트 루트에서 "
        f"`python run_pipeline.py --step {spec['step']}` 를 먼저 실행해 주세요."
    )


# ---------------------------------------------------------------------------
# UI 헬퍼
# ---------------------------------------------------------------------------

def grade_badge_html(grade: str) -> str:
    color = GRADE_COLORS.get(grade, "#999999")
    return (
        f"<span style='background:{color};color:white;padding:4px 16px;"
        f"border-radius:14px;font-weight:700;font-size:1.15rem'>{grade}</span>"
    )


def event_badge_html(event_type: str) -> str:
    color = _EVENT_COLORS.get(event_type, "#7f8c8d")
    return (
        f"<span style='background:{color};color:white;padding:2px 10px;"
        f"border-radius:10px;font-size:0.8rem'>{event_type}</span>"
    )


def brand_name_map(panel: pd.DataFrame | None) -> dict[str, str]:
    if panel is None or "brand_id" not in getattr(panel, "columns", []):
        return {}
    try:
        latest = panel.sort_values("year").groupby("brand_id").tail(1)
        return dict(zip(latest["brand_id"].astype(str), latest["brand_name"].astype(str)))
    except Exception:  # noqa: BLE001
        return {}


def render_sidebar(cfg: dict) -> str:
    st.sidebar.title("FranSCORE")
    st.sidebar.caption("프랜차이즈 구조악화 조기경보 · KB AI Challenge 프로토타입")
    view = st.sidebar.radio("화면 선택", ["① 브랜드 상세", "② 포트폴리오 뷰"])
    st.sidebar.warning(_SYNTH_WARN)

    st.sidebar.markdown("### 데이터 상태")
    for key, spec in _ARTIFACTS.items():
        p = artifact_path(cfg, key)
        mark = "✅" if p.exists() else "❌"
        st.sidebar.caption(f"{mark} {spec['filename']}")

    split_years = load_artifact("split_years")
    if isinstance(split_years, dict) and split_years:
        pairs = " · ".join(
            f"{k}={v}" for k, v in split_years.items()
            if not str(k).startswith("_") and not isinstance(v, (dict, list))
        )
        if pairs:
            st.sidebar.caption(f"시간분할: {pairs}")
    else:
        st.sidebar.caption("시간분할: (split_years.json 없음 — --step model)")
    return view


# ---------------------------------------------------------------------------
# 화면 1 — 브랜드 상세
# ---------------------------------------------------------------------------

def render_brand_view(cfg: dict) -> None:
    st.header("① 브랜드 상세")

    preds = load_artifact("predictions")
    if preds is None:
        missing_notice("predictions")
        return
    if "split" not in preds.columns:
        st.info("predictions.parquet에 split 컬럼이 없습니다. --step model 재실행이 필요합니다.")
        return
    test = preds[preds["split"] == "test"].copy()
    if test.empty:
        st.info("test 분할 예측이 비어 있습니다. `python run_pipeline.py --step model` 을 확인하세요.")
        return

    prob_col, calibrated = pick_prob_column(test)
    if prob_col not in test.columns:
        st.info("predictions.parquet에 확률 컬럼(p_calibrated/p_lgbm)이 없습니다.")
        return
    test["prob"] = test[prob_col].astype(float)
    test = test.sort_values("prob", ascending=False).reset_index(drop=True)
    test["grade"] = assign_grade(test["prob"], cfg.get("portfolio", {}).get("risk_grades", {}))

    panel = load_artifact("panel")
    names = brand_name_map(panel)

    # 포트폴리오 등급이 있으면 그것을 우선 (M4 산출과 일관성)
    pf = load_artifact("portfolio")
    pf_grade: dict[str, str] = {}
    if pf is not None:
        gcol = find_col(pf, ("grade", "risk_grade"))
        idcol = find_col(pf, ("brand_id",))
        if gcol and idcol:
            pf_grade = dict(zip(pf[idcol].astype(str), pf[gcol].astype(str)))

    def _fmt(bid: str) -> str:
        row = test[test["brand_id"].astype(str) == str(bid)].iloc[0]
        nm = names.get(str(bid), str(bid))
        return f"{nm} ({bid}) — P={row['prob']:.1%}"

    brand_ids = test["brand_id"].astype(str).tolist()
    sel = st.selectbox("브랜드 선택 (test 분할 · 위험 내림차순)", brand_ids, format_func=_fmt)
    row = test[test["brand_id"].astype(str) == str(sel)].iloc[0]
    brand_name = names.get(str(sel), str(sel))
    grade = pf_grade.get(str(sel), row["grade"])
    test_year = int(row["year"]) if "year" in row.index and pd.notna(row["year"]) else None

    # --- 등급 배지 + 확률 ---
    c1, c2, c3 = st.columns([1.2, 1.2, 2.6])
    with c1:
        st.markdown("**위험등급**")
        st.markdown(grade_badge_html(str(grade)), unsafe_allow_html=True)
    with c2:
        if calibrated:
            st.metric("P(악화 전환) — 보정확률", f"{row['prob']:.1%}")
        else:
            st.metric("위험 점수 (비보정)", f"{row['prob']:.3f}")
            st.caption("보정확률(p_calibrated) 미산출 — 점수·등급으로 표시합니다.")
    with c3:
        st.markdown("**리스크관리 권고**")
        st.markdown(GRADE_RECO.get(str(grade), GRADE_RECO["Low"]))
        st.caption(_DISCLAIMER)

    st.divider()

    # --- 공시 핵심지표 미니 시계열 ---
    st.subheader("공시 핵심지표 추이")
    if panel is None:
        missing_notice("panel")
    else:
        bp = panel[panel["brand_id"].astype(str) == str(sel)].sort_values("year")
        if bp.empty:
            st.info("패널에서 해당 브랜드 이력을 찾지 못했습니다.")
        else:
            ts = bp.set_index(bp["year"].astype(int))
            t1, t2 = st.columns(2)
            with t1:
                st.caption("가맹점 수 (연말)")
                st.line_chart(ts[["n_stores"]].rename(columns={"n_stores": "가맹점 수"}))
            with t2:
                st.caption("가맹점 평균매출액 (천원)")
                if "avg_sales" in ts.columns and ts["avg_sales"].notna().any():
                    st.line_chart(ts[["avg_sales"]].rename(columns={"avg_sales": "평균매출(천원)"}))
                else:
                    st.info("평균매출 결측 (공시 미기재)")

    # --- SHAP 상위 4 요인 ---
    st.subheader("위험요인 기여도 (SHAP 상위 4)")
    shap_df = load_artifact("shap_values")
    shap_top_ctx: list[dict] = []
    if shap_df is None:
        missing_notice("shap_values")
    else:
        try:
            sb = shap_df[shap_df["brand_id"].astype(str) == str(sel)]
            if test_year is not None and "year" in sb.columns:
                sb_y = sb[sb["year"] == test_year]
                if not sb_y.empty:
                    sb = sb_y
            if sb.empty:
                st.info("해당 브랜드의 SHAP 값이 없습니다 (--step evaluate 확인).")
            else:
                sb = sb.copy()
                sb["abs_shap"] = sb["shap_value"].astype(float).abs()
                top4 = sb.sort_values("abs_shap", ascending=False).head(4)
                top4["요인"] = top4["feature"].map(feature_korean_name)
                chart = top4.set_index("요인")[["shap_value"]].rename(
                    columns={"shap_value": "SHAP 기여도"})
                st.bar_chart(chart, horizontal=True)
                st.caption("양(+)의 값 = 악화 위험을 높이는 방향 · 음(−)의 값 = 낮추는 방향")
                for _, r in top4.iterrows():
                    fv = r.get("feature_value", float("nan"))
                    fv_txt = f"{fv:,.3f}" if pd.notna(fv) else "결측"
                    st.caption(f"· {r['요인']} — 값 {fv_txt}, 기여도 {float(r['shap_value']):+.4f}")
                    shap_top_ctx.append({
                        "feature": str(r["feature"]),
                        "feature_kr": str(r["요인"]),
                        "shap_value": float(r["shap_value"]),
                        "feature_value": (float(fv) if pd.notna(fv) else None),
                    })
        except Exception as e:  # noqa: BLE001
            st.info(f"SHAP 표시 중 문제가 발생했습니다: {e}")

    # --- 뉴스 경보 카드 ---
    st.subheader("뉴스 경보")
    news_obj = load_artifact("news_signals")
    brand_news_ctx: list[dict] = []
    if news_obj is None:
        missing_notice("news_signals")
    else:
        signals, llm_used = parse_news_signals(news_obj)
        matched = []
        for s in signals:
            s_brand = str(s.get("brand", ""))
            if not brand_name:
                continue
            if brand_name in s_brand or s_brand in (brand_name,) or (
                    s_brand and s_brand in brand_name):
                matched.append(s)
        matched = [s for s in matched if str(s.get("event_type", "")) != "무관"]
        if not llm_used and matched:
            st.caption("규칙기반 라이트 신호 (LLM 미사용, llm_used=false)")
        if not matched:
            st.caption("해당 브랜드 관련 뉴스 신호가 없습니다.")
        for s in matched[:5]:
            with st.container(border=True):
                title = s.get("title") or s.get("evidence_sentence") or "(제목 없음)"
                src = s.get("source") or s.get("source_url") or ""
                date = s.get("published") or s.get("date") or ""
                conf = s.get("confidence", "")
                et = str(s.get("event_type", "기타"))
                st.markdown(
                    f"{event_badge_html(et)} **{title}**", unsafe_allow_html=True)
                meta = " · ".join(x for x in [str(src), str(date),
                                              f"확신도 {conf}" if conf else ""] if x)
                if meta:
                    st.caption(meta)
                st.caption(_NEWS_CAPTION)
            brand_news_ctx.append(s)

    # --- 집중 여신·상관손실 시나리오 (포트폴리오 카드) ---
    st.subheader("집중 여신 · 상관손실 시나리오 (합성 예시)")
    pf_ctx: dict = {}
    if pf is None:
        missing_notice("portfolio")
    else:
        idcol = find_col(pf, ("brand_id",))
        prow = pf[pf[idcol].astype(str) == str(sel)] if idcol else pf.iloc[0:0]
        if prow.empty:
            st.caption("이 브랜드는 예시 포트폴리오 표본에 포함되지 않았습니다 "
                       f"(표본 {len(pf)}개 브랜드).")
        else:
            prow = prow.iloc[0]
            expo_col = find_col(pf, ("exposure_mkrw", "exposure_million", "exposure"))
            el_cols, stress_cols = pick_el_columns(pf)
            cols = st.columns(2 + len(el_cols) + len(stress_cols))
            i = 0
            if expo_col:
                expo_eok = to_eok(prow[expo_col])
                cols[i].metric("여신 익스포저", f"{expo_eok:,.1f} 억원")
                pf_ctx["exposure_eokwon"] = expo_eok
                i += 1
            for c in el_cols:
                v = to_eok(prow[c])
                cols[i].metric(f"EL ({c})", f"{v:,.2f} 억원")
                pf_ctx[f"el_{c}_eokwon"] = v
                i += 1
            for c in stress_cols:
                v = to_eok(prow[c])
                cols[i].metric(f"스트레스 ({c})", f"{v:,.2f} 억원")
                pf_ctx[f"stress_{c}_eokwon"] = v
                i += 1
            st.caption("⚠️ " + _SYNTH_WARN + " · EL = exposure × PD × LGD (LGD 시나리오 3종)")

    # --- 심사메모 ---
    with st.expander("심사메모 (LLM 생성 — 입력 근거만 인용)"):
        memo_key = f"memo_{sel}"
        if st.button("심사메모 생성", key=f"memo_btn_{sel}"):
            panel_metrics: dict = {}
            if panel is not None:
                bp2 = panel[panel["brand_id"].astype(str) == str(sel)].sort_values("year")
                if not bp2.empty:
                    last = bp2.iloc[-1]
                    for k in ("year", "n_stores", "n_direct", "n_new",
                              "n_contract_end", "n_contract_cancel", "avg_sales"):
                        if k in last.index and pd.notna(last[k]):
                            panel_metrics[k] = float(last[k])
            context = {
                "brand_name": brand_name,
                "grade": str(grade),
                "prob": float(row["prob"]),
                "shap_top": shap_top_ctx,
                "panel_metrics": panel_metrics,
                "news": brand_news_ctx,
                "portfolio": pf_ctx,
            }
            try:
                from src import memo_llm
                with st.spinner("심사메모 생성 중..."):
                    memo = memo_llm.generate_memo(context, cfg)
                st.session_state[memo_key] = memo
            except Exception as e:  # noqa: BLE001
                st.info(f"심사메모 생성에 실패했습니다: {e} — "
                        "memo_llm 모듈/API 키(ANTHROPIC_API_KEY)를 확인하세요.")
        if memo_key in st.session_state:
            st.markdown(st.session_state[memo_key])
            st.caption(_DISCLAIMER)


# ---------------------------------------------------------------------------
# 화면 2 — 포트폴리오 뷰
# ---------------------------------------------------------------------------

def render_portfolio_view(cfg: dict) -> None:
    st.header("② 포트폴리오 뷰")
    st.caption("⚠️ " + _SYNTH_WARN)

    pf = load_artifact("portfolio")
    summary = load_artifact("portfolio_summary")
    if pf is None and summary is None:
        missing_notice("portfolio")
        return

    flat = flatten_numeric(summary) if isinstance(summary, dict) else {}

    expo_col = find_col(pf, ("exposure_mkrw", "exposure_million", "exposure")) if pf is not None else None
    gcol = find_col(pf, ("grade", "risk_grade")) if pf is not None else None
    pcol = None
    if pf is not None:
        pcol = find_col(pf, ("p_calibrated", "pd", "p_lgbm", "prob", "p"))

    # --- KPI ---
    total_expo = find_num(flat, ("exposure",), ("stress", "scale", "per_store"))
    if total_expo is None and pf is not None and expo_col:
        total_expo = float(pf[expo_col].sum())
    hhi = find_num(flat, ("hhi",))
    top10 = find_num(flat, ("top10",)) or find_num(flat, ("top_10",)) or find_num(flat, ("top", "10"))
    n_high = None
    if pf is not None and gcol:
        n_high = int((pf[gcol].astype(str) == "High").sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("총 익스포저 (합성)", f"{to_eok(total_expo):,.0f} 억원" if total_expo is not None else "—")
    k2.metric("HHI (집중도)", f"{hhi:.4f}" if hhi is not None else "—")
    if top10 is not None:
        top10_disp = top10 * 100 if top10 <= 1.0 else top10
        k3.metric("상위 10 집중도", f"{top10_disp:.1f}%")
    else:
        k3.metric("상위 10 집중도", "—")
    k4.metric("위험(High) 브랜드 수", f"{n_high}" if n_high is not None else "—")

    if isinstance(summary, dict):
        headline = summary.get("headline") or summary.get("headline_sentence")
        if isinstance(headline, str) and headline:
            st.info(headline)

    st.divider()

    if pf is None:
        missing_notice("portfolio")
        return

    # 확률·등급 보강 (portfolio.csv에 없으면 predictions에서 결합)
    dfp = pf.copy()
    idcol = find_col(dfp, ("brand_id",))
    if pcol is None:
        preds = load_artifact("predictions")
        if preds is not None and "split" in preds.columns and idcol:
            t = preds[preds["split"] == "test"].copy()
            pc, _ = pick_prob_column(t)
            if pc in t.columns:
                m = dict(zip(t["brand_id"].astype(str), t[pc].astype(float)))
                dfp["_prob"] = dfp[idcol].astype(str).map(m)
                pcol = "_prob"
    if pcol is None:
        st.info("포트폴리오에 결합할 확률 컬럼을 찾지 못했습니다 (--step model/evaluate 확인).")
        return
    dfp["_p"] = dfp[pcol].astype(float)
    if gcol is None:
        dfp["_grade"] = assign_grade(dfp["_p"], cfg.get("portfolio", {}).get("risk_grades", {}))
        gcol = "_grade"

    # --- 산점도: 위험 vs 익스포저 ---
    st.subheader("위험 × 익스포저 산점")
    if expo_col is None:
        st.info("portfolio.csv에서 exposure 컬럼을 찾지 못했습니다.")
    else:
        try:
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            expo_eok = dfp[expo_col].astype(float) / 100.0
            for g, color in GRADE_COLORS.items():
                mask = dfp[gcol].astype(str) == g
                if mask.any():
                    ax.scatter(dfp.loc[mask, "_p"], expo_eok[mask], c=color,
                               label=g, alpha=0.8, s=38, edgecolors="none")
            ax.set_xlabel("P(deterioration)")
            ax.set_ylabel("Exposure (100M KRW, log scale)")
            if (expo_eok > 0).all():
                ax.set_yscale("log")
            ax.set_title("Portfolio risk vs exposure (synthetic exposure)")
            ax.legend(title="Grade", loc="best")
            ax.grid(alpha=0.25)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            st.info(f"산점도 표시 중 문제가 발생했습니다: {e}")

    # --- 위험 상위 20 브랜드 테이블 ---
    st.subheader("위험 상위 20 브랜드")
    try:
        panel = load_artifact("panel")
        names = brand_name_map(panel)
        top20 = dfp.sort_values("_p", ascending=False).head(20).copy()
        ncol = find_col(top20, ("brand_name", "brand"))
        disp = pd.DataFrame()
        if ncol:
            disp["브랜드"] = top20[ncol].astype(str)
        elif idcol:
            disp["브랜드"] = top20[idcol].astype(str).map(lambda b: names.get(b, b))
        disp["등급"] = top20[gcol].astype(str).values
        disp["P(악화)"] = top20["_p"].map(lambda v: f"{v:.1%}").values
        if expo_col:
            disp["익스포저(억원)"] = (top20[expo_col].astype(float) / 100.0).map(
                lambda v: f"{v:,.1f}").values
        el_cols, stress_cols = pick_el_columns(dfp)
        for c in el_cols + stress_cols:
            disp[f"{c}(억원)"] = (top20[c].astype(float) / 100.0).map(
                lambda v: f"{v:,.2f}").values
        st.dataframe(disp, hide_index=True)
    except Exception as e:  # noqa: BLE001
        st.info(f"랭킹 표 표시 중 문제가 발생했습니다: {e}")

    # --- LGD 시나리오별 총 EL 비교 ---
    st.subheader("LGD 시나리오별 총 예상손실(EL) 비교")
    el_cols, stress_cols = pick_el_columns(dfp)
    if el_cols:
        totals = {c: float(dfp[c].astype(float).sum()) / 100.0 for c in el_cols}
        el_chart = pd.DataFrame(
            {"총 EL (억원)": list(totals.values())},
            index=list(totals.keys()))
        st.bar_chart(el_chart)
        st.caption("EL = exposure × PD × LGD · LGD 시나리오 "
                   f"{cfg.get('portfolio', {}).get('lgd_scenarios', [])}")
    else:
        el_flat = {k: v for k, v in flat.items()
                   if "el" in k.lower() and "stress" not in k.lower() and "headline" not in k.lower()}
        if el_flat:
            st.bar_chart(pd.DataFrame(
                {"총 EL (억원)": [to_eok(v) for v in el_flat.values()]},
                index=list(el_flat.keys())))
        else:
            st.info("EL 시나리오 컬럼을 찾지 못했습니다 (--step portfolio 확인).")

    # --- 스트레스 vs 기본 비교 ---
    st.subheader("스트레스 시나리오 비교")
    base_total = None
    stress_total = None
    if el_cols:
        mid_idx = len(el_cols) // 2
        base_total = float(dfp[el_cols[mid_idx]].astype(float).sum()) / 100.0
    if stress_cols:
        stress_total = float(dfp[stress_cols[0]].astype(float).sum()) / 100.0
    if base_total is None:
        v = find_num(flat, ("el",), ("stress",))
        base_total = to_eok(v) if v is not None else None
    if stress_total is None:
        v = find_num(flat, ("stress",))
        stress_total = to_eok(v) if v is not None else None
    if base_total is not None and stress_total is not None:
        cmp_chart = pd.DataFrame(
            {"EL (억원)": [base_total, stress_total]},
            index=["기본 (중간 LGD)", "스트레스 (상위 위험 PD×1.5)"])
        st.bar_chart(cmp_chart)
        mult = cfg.get("portfolio", {}).get("stress_pd_multiplier", 1.5)
        top_pct = cfg.get("portfolio", {}).get("stress_top_pct", 0.10)
        st.caption(f"스트레스: 위험 상위 {top_pct:.0%} 브랜드 PD × {mult} (cap 1.0) 동시 악화 가정")
    else:
        st.info("스트레스 EL 값을 찾지 못했습니다 (--step portfolio 확인).")

    # --- 가정 명세 ---
    if isinstance(summary, dict):
        with st.expander("포트폴리오 요약 원본 (portfolio_summary.json — 합성 가정 명세)"):
            st.json(summary)

    # --- 백테스트 성능 요약 ---
    with st.expander("백테스트 성능 요약 (metrics.csv — 기준모형 대비 우위 증거)"):
        metrics = load_artifact("metrics")
        if metrics is None:
            missing_notice("metrics")
        else:
            st.dataframe(metrics, hide_index=True)
            st.caption(
                "주모형 LightGBM(p_lgbm) vs 기준모형 3종(persistence · single · logistic). "
                "완전 시간분할의 test 연도 Lift@10% · Precision@10% · PR-AUC가 핵심 비교 지표입니다.")


# ---------------------------------------------------------------------------
# 엔트리포인트
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="FranSCORE", layout="wide")
    cfg = load_config()
    set_seed(cfg.get("seed", 42))

    view = render_sidebar(cfg)

    st.warning("⚠️ " + _SYNTH_WARN + " — " + _DISCLAIMER)

    if view.startswith("①"):
        render_brand_view(cfg)
    else:
        render_portfolio_view(cfg)


if __name__ == "__main__":
    main()
