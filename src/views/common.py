"""화면 공용 — 데이터 로딩·브랜드 마크·소견 표시·차트.

모든 화면이 같은 산출물을 같은 방식으로 읽도록 로더를 여기 하나로 모은다.
파일 수정시각을 캐시 키에 넣어 파이프라인이 다시 돌면 화면이 자동으로 갱신된다.
"""
from __future__ import annotations

import base64
import colorsys
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import theme
from src.common import load_config

GRADE_KR = {"High": "주의", "Medium": "관찰", "Low": "양호"}
GRADE_ACTION = {
    "High": "신규 취급 전 최근 공시와 본부 재무를 함께 확인하고, "
            "이 브랜드에 나간 여신 총액이 한도 안인지 점검합니다.",
    "Medium": "분기 단위로 점포 수와 계약종료율의 방향을 확인합니다.",
    "Low": "정기 모니터링을 유지합니다.",
}
CATEGORY_ICON = {"성장": "▮", "계약": "▮", "매출": "▮", "재무": "▮",
                 "구조": "▮", "수요": "▮", "평판": "▮"}


# ---------------------------------------------------------------------------
# 로더
# ---------------------------------------------------------------------------

def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def _csv(path: str, m: float) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def _parquet(path: str, m: float, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=list(columns) if columns else None)


@st.cache_data(show_spinner=False)
def _json(path: str, m: float):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def cfg() -> dict:
    return load_config()


def out_dir() -> Path:
    return Path(cfg()["paths"]["outputs"])


def proc_dir() -> Path:
    return Path(cfg()["paths"]["processed"])


def load_scores() -> tuple[pd.DataFrame | None, dict]:
    """운영 점수표. 확장 트랙이 최신이면 그것을, 아니면 기본 트랙을 쓴다.

    ⚠️ 예전에는 확장 트랙을 무조건 우선했다. 확장 트랙을 다시 돌리지 않은 상태에서
       기본 트랙만 재학습하면 화면이 **오래된 점수**를 보여주게 된다(실측: 화면은
       7월 31일 산출물, 실제 최신은 8월 1일). 수정시각을 비교해 최신을 고른다.
    """
    cands = []
    for path, label in ((out_dir() / "extended" / "scores_latest.csv", "전 업종"),
                        (out_dir() / "scores_latest.csv", "외식업")):
        if path.exists():
            cands.append((_mtime(path), path, label))
    if not cands:
        return None, {}
    mt, path, label = max(cands)
    df = _csv(str(path), mt)
    meta_p = path.parent / "scores_latest_meta.json"
    meta = _json(str(meta_p), _mtime(meta_p)) if meta_p.exists() else {}
    meta["scope_label"] = label
    meta["scope_dir"] = str(path.parent)
    return df, meta


def load_diagnosis_summary() -> pd.DataFrame | None:
    _, meta = load_scores()
    base = Path(meta.get("scope_dir") or out_dir())
    p = base / "brand_diagnosis_summary.csv"
    if not p.exists():
        p = out_dir() / "brand_diagnosis_summary.csv"
    return _csv(str(p), _mtime(p)) if p.exists() else None


def load_findings(brand_id: str | None = None) -> pd.DataFrame | None:
    _, meta = load_scores()
    base = Path(meta.get("scope_dir") or out_dir())
    p = base / "brand_diagnosis.parquet"
    if not p.exists():
        p = out_dir() / "brand_diagnosis.parquet"
    if not p.exists():
        return None
    df = _parquet(str(p), _mtime(p))
    if brand_id is not None:
        df = df[df["brand_id"].astype(str) == str(brand_id)]
    return df


def load_panel(full: bool = False) -> pd.DataFrame | None:
    name = "panel_full.parquet" if full else "panel.parquet"
    p = proc_dir() / name
    return _parquet(str(p), _mtime(p)) if p.exists() else None


def load_hq_financials() -> pd.DataFrame | None:
    p = proc_dir() / "hq_financials.parquet"
    return _parquet(str(p), _mtime(p)) if p.exists() else None


def load_demand() -> dict:
    p = out_dir() / "demand_trends.json"
    if not p.exists():
        return {}
    obj = _json(str(p), _mtime(p))
    return obj if isinstance(obj, dict) else {}


def load_portfolio() -> tuple[pd.DataFrame | None, dict]:
    _, meta = load_scores()
    base = Path(meta.get("scope_dir") or out_dir())
    pf, summ = base / "portfolio.csv", base / "portfolio_summary.json"
    if not pf.exists():
        pf, summ = out_dir() / "portfolio.csv", out_dir() / "portfolio_summary.json"
    df = _csv(str(pf), _mtime(pf)) if pf.exists() else None
    s = _json(str(summ), _mtime(summ)) if summ.exists() else {}
    return df, (s if isinstance(s, dict) else {})


def scored_year() -> str:
    _, meta = load_scores()
    return str(meta.get("scored_year", "-"))


# ---------------------------------------------------------------------------
# 브랜드 마크 (로고)
# ---------------------------------------------------------------------------

_LOGO_CACHE = "data/raw/naver/logos.json"


@st.cache_data(show_spinner=False, max_entries=4096)
def _logo_data_uri(brand_name: str, m: float) -> str:
    """브랜드 로고 파일 → data URI. 없으면 빈 문자열.

    파일 경로는 브랜드명 해시로 **계산**한다 — 색인 파일(logos.json)을 거치지 않는다.
    수집이 백그라운드로 돌면서 색인을 통째로 덮어쓰면 방금 저장한 파일 참조가 사라지는
    경합이 실제로 났다(PNG 556장이 디스크에 있는데 색인은 0건). 경로를 계산하면
    색인과 무관하게 항상 맞고, 수집 도중에도 화면이 정상 동작한다.

    ⚠️ 원본 URL 을 <img src> 로 쓰지 않는 이유: 기업 사이트 상당수가 외부 Referer
       요청을 막아 배포 화면에서 이미지가 깨진다. 파일을 우리가 들고 있으면 상대
       사이트 상태와 무관하게 항상 뜬다. 128px PNG 라 한 장에 5~15KB 다.
    """
    from src.naver import LOGO_DIR, logo_file_name
    p = Path(cfg()["_root"]) / LOGO_DIR / logo_file_name(brand_name)
    if not p.exists():
        return ""
    try:
        return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")
    except OSError:
        return ""


def logo_url(brand_name: str) -> str:
    """브랜드 로고를 화면에 바로 넣을 수 있는 형태로. 없으면 빈 문자열."""
    from src.naver import LOGO_DIR
    d = Path(cfg()["_root"]) / LOGO_DIR
    # 디렉토리 수정시각을 캐시 키에 넣어 수집이 진행되면 화면이 따라오게 한다
    return _logo_data_uri(str(brand_name), _mtime(d))


def _mark_colors(name: str) -> tuple[str, str, str]:
    """브랜드명에서 결정적으로 색을 만든다 (같은 브랜드는 항상 같은 색).

    같은 색조의 밝은 두 단계로 그라데이션을 만들고 글자는 진한 톤으로 둔다.
    단색 사각형에 글자만 얹으면 '자리표시자'로 보이지만, 색조가 브랜드마다 다르고
    질감이 있으면 목록에서 브랜드를 색으로 구분할 수 있다.
    """
    h = int(hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:8], 16)
    hue = (h % 360) / 360.0
    to_hex = lambda t: "#" + "".join(f"{int(v * 255):02X}" for v in t)  # noqa: E731
    return (to_hex(colorsys.hls_to_rgb(hue, 0.90, 0.62)),   # 배경 밝은 쪽
            to_hex(colorsys.hls_to_rgb(hue, 0.80, 0.58)),   # 배경 어두운 쪽
            to_hex(colorsys.hls_to_rgb(hue, 0.28, 0.66)))   # 글자


def _mark_label(brand_name: str) -> str:
    """마크에 넣을 글자 — 한글은 두 자, 영문은 이니셜."""
    s = "".join(ch for ch in str(brand_name) if ch.strip())
    if not s:
        return "?"
    kor = "".join(ch for ch in s if "가" <= ch <= "힣")
    if len(kor) >= 2:
        return kor[:2]
    words = [w for w in re.split(r"[\s()（）\[\]]+", str(brand_name)) if w]
    if len(words) >= 2 and all(w[0].isalpha() for w in words[:2]):
        return (words[0][0] + words[1][0]).upper()
    return s[:2].upper()


def brand_mark_html(brand_name: str, size: int = 64) -> str:
    """브랜드 마크 — 로고가 있으면 이미지, 없으면 브랜드 색 글자 마크.

    로고마다 원본 비율·여백이 제각각이라 그냥 넣으면 어떤 건 꽉 차고 어떤 건
    점처럼 보인다. **같은 크기의 타일 안에 contain 으로 앉히고** 안쪽 여백을
    일정하게 줘서 목록에서 크기가 고르게 보이도록 한다.
    cover 를 쓰면 가로로 긴 로고의 좌우가 잘려 다른 브랜드처럼 보인다.
    """
    url = logo_url(brand_name)
    radius = int(size * 0.26)
    pad = max(5, int(size * 0.13))
    style = (f"width:{size}px;height:{size}px;border-radius:{radius}px;"
             f"flex:0 0 {size}px;box-sizing:border-box;")
    if url:
        return (f"<div class='kb-mark' style='{style}padding:{pad}px'>"
                f"<img src='{url}' alt='{brand_name}'/></div>")
    c1, c2, fg = _mark_colors(brand_name)
    return (f"<div class='kb-mark' style='{style}"
            f"background:linear-gradient(150deg,{c1} 0%,{c2} 100%)'>"
            f"<span class='letter' style='color:{fg};font-size:{int(size * 0.34)}px'>"
            f"{_mark_label(brand_name)}</span></div>")


# ---------------------------------------------------------------------------
# 소견 표시
# ---------------------------------------------------------------------------

def render_findings(findings: pd.DataFrame, limit: int | None = None,
                    show_info: bool = True) -> None:
    """소견 목록을 문장 그대로 보여준다."""
    if findings is None or findings.empty:
        st.caption("표시할 소견이 없습니다.")
        return
    df = findings if show_info else findings[findings["direction"] != "info"]
    rows = df.head(limit) if limit else df
    html = []
    for _, r in rows.iterrows():
        sev = str(r["severity"]) if r["direction"] == "risk" else (
            "Low" if r["direction"] == "mitigant" else "Neutral")
        color_key = {"risk": str(r["severity"]), "mitigant": "Low"}.get(
            str(r["direction"]), "Neutral")
        mark = {"risk": "", "mitigant": "완화요인 · ", "info": "확인 필요 · "}.get(
            str(r["direction"]), "")
        html.append(theme.finding_html(
            title=f"{mark}{r['title']}",
            desc=str(r["detail"]),
            severity=color_key if color_key in theme.GRADE_COLOR else "Low",
            source=f"{r['category']} · {r['source']}"))
        del sev
    st.markdown("".join(html), unsafe_allow_html=True)


def category_summary(findings: pd.DataFrame) -> str:
    """위험 카테고리를 배지 줄로."""
    if findings is None or findings.empty:
        return ""
    risk = findings[findings["direction"] == "risk"]
    chips = []
    for cat, sub in risk.groupby("category", sort=False):
        worst = "High" if (sub["severity"] == "High").any() else (
            "Medium" if (sub["severity"] == "Medium").any() else "Low")
        chips.append(theme.chip(f"{cat} {len(sub)}", worst))
    return " ".join(chips)


# ---------------------------------------------------------------------------
# 차트
# ---------------------------------------------------------------------------

def line_chart(df: pd.DataFrame, x: str, y: str, name: str,
               unit: str = "", fill: bool = True) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x], y=df[y], mode="lines+markers", name=name,
        line={"color": theme.YELLOW_DEEP, "width": 2.5},
        marker={"size": 7, "color": theme.YELLOW_DEEP,
                "line": {"color": "#FFFFFF", "width": 1.5}},
        fill="tozeroy" if fill else None,
        fillcolor="rgba(255,204,0,0.13)",
        hovertemplate=f"%{{x}}<br><b>%{{y:,.0f}}</b>{unit}<extra></extra>"))
    fig.update_layout(showlegend=False, height=210,
                      margin={"l": 4, "r": 4, "t": 8, "b": 4})
    return fig


def bar_chart(labels, values, colors=None, unit: str = "",
              horizontal: bool = True) -> go.Figure:
    c = colors or theme.YELLOW_DEEP
    fig = go.Figure(go.Bar(
        x=values if horizontal else labels,
        y=labels if horizontal else values,
        orientation="h" if horizontal else "v",
        marker={"color": c, "line": {"width": 0}},
        hovertemplate=f"%{{y}}<br><b>%{{x:,.1f}}</b>{unit}<extra></extra>"
        if horizontal else f"%{{x}}<br><b>%{{y:,.1f}}</b>{unit}<extra></extra>"))
    fig.update_layout(showlegend=False, margin={"l": 4, "r": 4, "t": 8, "b": 4})
    return fig


def refresh_state() -> dict:
    """마지막 갱신 이력. 없으면 산출물 파일 시각으로 대체한다."""
    p = out_dir() / "refresh_state.json"
    if p.exists():
        try:
            return _json(str(p), _mtime(p)) or {}
        except (OSError, ValueError):
            pass
    s = out_dir() / "scores_latest.csv"
    if not s.exists():
        return {}
    return {"finished_at": datetime.fromtimestamp(_mtime(s)).strftime("%Y-%m-%d %H:%M"),
            "source": "file_mtime"}


def refresh_footer() -> None:
    """화면 하단 갱신 표기 — 이 숫자가 언제 것인지 밝히지 않으면 신뢰할 수 없다."""
    st.write("")
    r = refresh_state()
    when = r.get("finished_at") or "-"
    bits = [f"마지막 갱신 **{when}**"]
    if r.get("news_added"):
        bits.append(f"신규 뉴스 {int(r['news_added']):,}건")
    if r.get("demand_updated"):
        bits.append(f"검색수요 {int(r['demand_updated']):,}개 브랜드")
    if r.get("rescored"):
        bits.append(f"재산출 {int(r['rescored']):,}개 브랜드")
    if r.get("source") == "file_mtime":
        bits.append("자동 갱신 이력 없음 (산출물 파일 시각)")
    st.caption(" · ".join(bits) + " · 갱신 주기 1일")


RISK_LABEL = "브랜드 리스크"
_RISK_HELP = ("공시·본부재무·검색수요를 학습한 모델이 산출한, 이 브랜드가 향후 1년 안에 "
              "구조적으로 꺾일 가능성입니다. 개별 가맹점의 연체 확률이 아닙니다.")


@st.cache_data(show_spinner=False)
def grade_bounds(m: float) -> dict:
    """등급 컷을 **실제 확률 경계**로 환산한다.

    등급 규칙 자체는 순위 백분위(상위 10% = 주의)다. 그런데 화면에서 "어느 구간이면
    어떤 마크가 붙는가"를 물으면, 사용자가 기대하는 답은 순위가 아니라 **퍼센트 경계**다.
    그래서 이번 산출물에서 그 경계가 실제로 몇 %인지 계산해 함께 보여준다.
    (두 표기를 나란히 두어야 '상위 10%' 라는 상대 기준도 숨기지 않는다.)
    """
    df, _ = load_scores()
    if df is None or df.empty:
        return {}
    g = (cfg().get("portfolio") or {}).get("risk_grades") or {}
    hi, mid = float(g.get("high", 0.90)), float(g.get("medium", 0.70))
    p = pd.to_numeric(df["pd_1y"], errors="coerce").dropna()
    return {
        "high_pct": hi, "medium_pct": mid,
        "high_cut": float(p.quantile(hi)) * 100,
        "medium_cut": float(p.quantile(mid)) * 100,
        "min": float(p.min()) * 100, "max": float(p.max()) * 100,
        "counts": {k: int(v) for k, v in df["risk_grade"].value_counts().items()},
    }


def signal_html(grade: str, pd_1y: float | None = None, *, size: int = 13,
                show_value: bool = True) -> str:
    """신호등 — 위험도를 '빨간 숫자'가 아니라 **켜진 등**으로 보여준다.

    빨간 글씨의 확률은 두 가지를 한꺼번에 요구한다: 숫자를 읽고, 그 숫자가 높은지
    낮은지를 스스로 판단하기. 신호등은 판단을 이미 끝낸 상태로 전달한다.
    숫자는 남기되(근거를 감추지 않는다) 보조 정보로 내린다.
    """
    order = ("High", "Medium", "Low")
    on = str(grade) if str(grade) in order else "Low"
    dots = "".join(
        f"<span style='width:{size}px;height:{size}px;border-radius:50%;"
        f"background:{theme.GRADE_COLOR[g] if g == on else '#D8D3CC'};"
        f"box-shadow:{f'0 0 0 3px {theme.GRADE_SOFT[g]}' if g == on else 'none'};"
        f"display:inline-block'></span>" for g in order)
    label = (f"<div style='font-size:.92rem;font-weight:700;color:{theme.GRADE_COLOR[on]};"
             f"margin-top:6px;letter-spacing:-.01em'>{theme.GRADE_KR[on]}</div>")
    val = ""
    if show_value and pd_1y is not None and pd.notna(pd_1y):
        val = (f"<div style='font-size:.82rem;color:{theme.TEXT_MUTED};margin-top:1px'>"
               f"{RISK_LABEL} {float(pd_1y) * 100:.1f}%</div>")
    return (f"<div style='display:inline-flex;flex-direction:column;align-items:center'>"
            f"<div style='display:flex;gap:7px;align-items:center;padding:7px 10px;"
            f"border-radius:999px;background:{theme.SURFACE};"
            f"border:1px solid {theme.BORDER}'>{dots}</div>{label}{val}</div>")


def grade_legend_html(bounds: dict) -> str:
    """등급 구간표 — 어떤 구간에 들어가면 어떤 마크가 붙는지."""
    if not bounds:
        return ""
    rows = [
        ("High", f"{bounds['high_cut']:.1f}% 이상",
         f"상위 {(1 - bounds['high_pct']) * 100:.0f}%", "즉시 점검"),
        ("Medium", f"{bounds['medium_cut']:.1f}% ~ {bounds['high_cut']:.1f}%",
         f"상위 {(1 - bounds['medium_pct']) * 100:.0f}%", "추이 관찰"),
        ("Low", f"{bounds['medium_cut']:.1f}% 미만",
         f"하위 {bounds['medium_pct'] * 100:.0f}%", "정기 점검"),
    ]
    cnt = bounds.get("counts", {})
    cells = "".join(
        f"<div style='display:flex;align-items:center;gap:10px;padding:9px 2px;"
        f"border-bottom:1px solid {theme.BORDER}'>"
        f"<span style='width:11px;height:11px;border-radius:50%;flex:0 0 auto;"
        f"background:{theme.GRADE_COLOR[g]};box-shadow:0 0 0 3px {theme.GRADE_SOFT[g]}'></span>"
        f"<span style='font-weight:700;color:{theme.INK};min-width:32px'>{theme.GRADE_KR[g]}</span>"
        f"<span style='color:{theme.TEXT};font-variant-numeric:tabular-nums'>{rng}</span>"
        f"<span style='color:{theme.TEXT_MUTED};font-size:.88rem'>({rank})</span>"
        f"<span style='margin-left:auto;color:{theme.TEXT_SUB};font-size:.9rem'>"
        f"{cnt.get(g, 0):,}개 · {act}</span></div>"
        for g, rng, rank, act in rows)
    return (f"<div style='font-size:.97rem;line-height:1.5'>{cells}"
            f"<div style='color:{theme.TEXT_MUTED};font-size:.86rem;margin-top:9px'>"
            f"등급은 <b>순위 기준</b>으로 나눕니다 — 매 산출 시점의 상위 10%가 주의, "
            f"상위 30%까지가 관찰입니다. 위 퍼센트는 이번 산출에서 그 순위 경계가 "
            f"실제로 몇 %였는지를 환산한 값입니다.</div></div>")


def risk_gauge(pd_1y: float, rank_pct: float | None = None) -> go.Figure:
    """브랜드 리스크 게이지 — 0~100% 전 구간을 등급 색으로 나눠 보여준다.

    축을 0~50 으로 자르면 '절반이 찼다'는 인상이 실제 위험보다 과장된다.
    확률은 0~100% 가 정의역이므로 축도 그대로 0~100 을 쓴다.
    """
    v = float(pd_1y) * 100
    b = grade_bounds(_mtime(out_dir() / "scores_latest.csv"))
    hi = b.get("high_cut", 25.0)
    mid = b.get("medium_cut", 10.0)
    color = (theme.DANGER if v >= hi else theme.WARN if v >= mid else theme.SAFE)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=v,
        number={"suffix": "%", "font": {"size": 34, "color": theme.INK}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "dtick": 25,
                     "ticksuffix": "%",
                     "tickfont": {"size": 10, "color": theme.TEXT_MUTED}},
            "bar": {"color": color, "thickness": 0.7},
            "bgcolor": "#F0EEEA",
            "borderwidth": 0,
            "steps": [{"range": [0, mid], "color": theme.SAFE_SOFT},
                      {"range": [mid, hi], "color": theme.WARN_SOFT},
                      {"range": [hi, 100], "color": theme.DANGER_SOFT}],
        }))
    fig.update_layout(height=170, margin={"l": 12, "r": 12, "t": 8, "b": 4})
    del rank_pct
    return fig
