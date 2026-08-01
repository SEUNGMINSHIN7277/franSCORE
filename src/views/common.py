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
def _csv(path: str, _m: float) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def _parquet(path: str, _m: float, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=list(columns) if columns else None)


@st.cache_data(show_spinner=False)
def _json(path: str, _m: float):
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
def _logo_data_uri(brand_name: str, _mtime: float) -> str:
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


def brand_mark_html(brand_name: str, size: int = 52) -> str:
    """로고가 있으면 이미지, 없으면 브랜드 색 마크.

    로고 이미지는 `object-fit: contain` 으로 넣는다 — cover 로 채우면 정사각형이
    아닌 로고의 좌우가 잘려 다른 브랜드처럼 보인다.
    """
    url = logo_url(brand_name)
    radius = int(size * 0.24)
    if url:
        # loading='lazy' 는 쓰지 않는다 — 한 화면에 8~12장뿐이라 이득이 없고,
        # 뷰포트 밖 카드가 빈칸으로 보이는 시간이 생긴다.
        return (f"<img src='{url}' alt='{brand_name}' "
                f"style='width:{size}px;height:{size}px;border-radius:{radius}px;"
                f"object-fit:contain;padding:{max(2, int(size * 0.08))}px;"
                f"border:1px solid {theme.BORDER};background:#FFFFFF;"
                f"flex:0 0 {size}px;box-sizing:border-box'/>")
    c1, c2, fg = _mark_colors(brand_name)
    return (f"<div style='width:{size}px;height:{size}px;border-radius:{radius}px;"
            f"background:linear-gradient(135deg,{c1} 0%,{c2} 100%);color:{fg};"
            f"display:flex;align-items:center;justify-content:center;font-weight:800;"
            f"font-size:{int(size * 0.36)}px;letter-spacing:-0.04em;"
            f"flex:0 0 {size}px;box-shadow:inset 0 0 0 1px rgba(0,0,0,.05)'>"
            f"{_mark_label(brand_name)}</div>")


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


def risk_gauge(pd_1y: float, rank_pct: float | None = None) -> go.Figure:
    """악화 가능성 게이지 — 숫자 하나를 크게, 위치를 색으로."""
    v = float(pd_1y) * 100
    color = (theme.DANGER if v >= 25 else theme.WARN if v >= 10 else theme.SAFE)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=v,
        number={"suffix": "%", "font": {"size": 34, "color": theme.INK}},
        gauge={
            "axis": {"range": [0, 50], "tickwidth": 0,
                     "tickfont": {"size": 10, "color": theme.TEXT_MUTED}},
            "bar": {"color": color, "thickness": 0.7},
            "bgcolor": "#F0EEEA",
            "borderwidth": 0,
            "steps": [{"range": [0, 10], "color": "#FAFAF8"},
                      {"range": [10, 25], "color": "#F6F4F0"},
                      {"range": [25, 50], "color": "#F1EEE9"}],
        }))
    fig.update_layout(height=170, margin={"l": 12, "r": 12, "t": 8, "b": 4})
    del rank_pct
    return fig
