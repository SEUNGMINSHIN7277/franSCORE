"""KB 디자인 시스템 — 색·타이포·컴포넌트를 한 곳에 정의한다.

색값의 출처
    KB국민은행 운영 사이트(kbstar.com)의 계산된 스타일에서 직접 추출했다.
    시그니처 옐로 #FFCC00, 보조 옐로 #FFE85A, 텍스트 계열 #4E473F·#5A5A5A,
    웜 그레이 보더 #EAE5DF. 추정이 아니라 실제 서비스가 쓰는 값이다.

왜 별도 모듈인가
    화면마다 색을 직접 적으면 한 곳을 고쳐도 다른 화면이 따라오지 않는다.
    차트(plotly)·배지·카드·표가 모두 같은 팔레트를 참조하도록 토큰을 여기 한 곳에 둔다.

사용법
    import streamlit as st
    from src import theme
    theme.setup(page_title="...")      # set_page_config + CSS 주입 (앱 최상단 1회)
    theme.page_header("제목", "설명")
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---------------------------------------------------------------------------
# 색 토큰
# ---------------------------------------------------------------------------

YELLOW = "#FFCC00"          # KB 시그니처 (kbstar.com 실측)
YELLOW_DEEP = "#F0B400"     # 눌림·강조
YELLOW_SOFT = "#FFF6D6"     # 배경 틴트
YELLOW_LINE = "#FFE85A"     # 보조 라인 (실측)

INK = "#26221E"             # 최상위 제목
TEXT = "#3D3833"            # 본문
TEXT_SUB = "#6B635A"        # 보조 설명
TEXT_MUTED = "#9A9289"      # 캡션·출처

BG = "#F5F4F1"              # 페이지 배경 (웜 오프화이트)
SURFACE = "#FFFFFF"
BORDER = "#E8E4DE"          # 웜 그레이 보더 (실측 #EAE5DF 계열)
BORDER_STRONG = "#D6D0C7"

NAV_BG = "#332F2A"          # 사이드바 (KB 그레이 계열 딥톤)
NAV_TEXT = "#CFC8BE"
NAV_ACTIVE_BG = YELLOW
NAV_ACTIVE_TEXT = "#26221E"

# 의미색 — 은행 화면이므로 채도를 낮춰 경고가 과장되지 않게 한다
DANGER = "#C8433B"
DANGER_SOFT = "#FBEDEB"
WARN = "#D9860F"
WARN_SOFT = "#FDF4E3"
SAFE = "#1F8A5B"
SAFE_SOFT = "#EAF5EF"
INFO = "#2F6FB5"
INFO_SOFT = "#EDF3FA"

GRADE_COLOR = {"High": DANGER, "Medium": WARN, "Low": SAFE}
GRADE_SOFT = {"High": DANGER_SOFT, "Medium": WARN_SOFT, "Low": SAFE_SOFT}
GRADE_KR = {"High": "주의", "Medium": "관찰", "Low": "양호"}

# 차트 계열색 — 옐로를 1순위로 두되 인접색이 서로 구분되게 배열
SERIES = [YELLOW_DEEP, "#4E473F", INFO, SAFE, "#B07CC6", WARN, "#5FA8A0", DANGER]

FONT_STACK = ("Pretendard, 'Pretendard Variable', -apple-system, BlinkMacSystemFont, "
              "'Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif")


# ---------------------------------------------------------------------------
# Plotly 템플릿
# ---------------------------------------------------------------------------

def _register_plotly() -> None:
    """KB 팔레트 plotly 템플릿을 등록하고 기본값으로 지정한다."""
    if "kb" in pio.templates:
        return
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        font={"family": FONT_STACK, "size": 13, "color": TEXT},
        title={"font": {"size": 15, "color": INK}, "x": 0, "xanchor": "left"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=SERIES,
        margin={"l": 8, "r": 8, "t": 34, "b": 8},
        xaxis={"gridcolor": BORDER, "linecolor": BORDER_STRONG, "zerolinecolor": BORDER,
               "tickfont": {"size": 12, "color": TEXT_SUB}, "automargin": True},
        yaxis={"gridcolor": BORDER, "linecolor": BORDER_STRONG, "zerolinecolor": BORDER,
               "tickfont": {"size": 12, "color": TEXT_SUB}, "automargin": True},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0,
                "font": {"size": 12}, "bgcolor": "rgba(0,0,0,0)"},
        hoverlabel={"font": {"family": FONT_STACK, "size": 12}, "bgcolor": SURFACE,
                    "bordercolor": BORDER_STRONG},
        separators=".,",
    )
    pio.templates["kb"] = tpl
    pio.templates.default = "plotly_white+kb"


_register_plotly()


def plot(fig: go.Figure, *, height: int | None = None, key: str | None = None) -> None:
    """차트 공통 렌더 — 모드바를 숨기고 컨테이너 폭에 맞춘다."""
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True, key=key,
                    config={"displayModeBar": False, "responsive": True})


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = f"""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');

html, body, [class*="st-"], button, input, textarea, select {{
    font-family: {FONT_STACK};
    -webkit-font-smoothing: antialiased;
}}
/* ⚠️ 위 규칙의 `[class*="st-"]` 는 Streamlit 의 **아이콘 span 까지** 잡는다
   (클래스가 st-emotion-cache-… 라서). 그러면 Material Symbols 글리프가 폰트를
   못 찾아 'keyboard_double_arrow_right' 라는 **글자 이름 그대로** 화면에 찍힌다.
   실제로 사이드바 접기 버튼이 그렇게 보였다. 아이콘 요소만 되돌린다. */
[data-testid="stIconMaterial"], .material-symbols-rounded, .material-symbols-outlined,
.material-icons, span[class*="material-symbols"], span[class*="material-icons"] {{
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined',
                 'Material Icons' !important;
    font-weight: normal !important;
    font-style: normal !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    white-space: nowrap;
    word-wrap: normal;
    direction: ltr;
    -webkit-font-feature-settings: 'liga';
    -webkit-font-smoothing: antialiased;
}}

/* ── 타이포 스케일 ───────────────────────────────────────────
   Streamlit 기본값(본문 14px 상당)은 1440px 이상 화면에서 너무 작아 가독성이
   떨어진다. 루트를 17px 로 올리고 전 요소를 rem 으로 잡아 한 곳에서 조절한다. */
/* Streamlit 이 `[data-testid=stMarkdownContainer] p {{font-size:.9rem}}` 처럼 자체
   규칙을 갖고 있어 같은 선택자로는 밀린다(실측: 1rem 지정 → 15.3px 렌더).
   테마 계층이므로 !important 로 확정한다. */
html {{ font-size: 17px; }}
body, .stMarkdown p, .stMarkdown li,
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{
    font-size: 1.06rem !important; line-height: 1.7; color: {TEXT};
}}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] div, .stCaption, small {{
    font-size: .92rem !important; line-height: 1.62; color: {TEXT_SUB} !important;
}}
label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {{
    font-size: .96rem !important; font-weight: 600; color: {TEXT};
}}
/* 표·선택지 안의 글씨도 함께 (기본값이 12~13px 로 작다) */
[data-testid="stDataFrame"], [data-baseweb="select"], [role="option"],
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {{
    font-size: .98rem !important;
}}

/* ── 페이지 셸 ───────────────────────────────────────────── */
[data-testid="stAppViewContainer"] {{ background: {BG}; }}
[data-testid="stHeader"] {{ background: transparent; height: 0; }}
/* Streamlit 기본 크롬(Deploy 버튼·상태 위젯)은 서비스 화면의 것이 아니다 */
[data-testid="stToolbar"], [data-testid="stStatusWidget"],
[data-testid="stDecoration"], .stDeployButton, [data-testid="stAppDeployButton"] {{
    display: none !important;
}}
[data-testid="stMainBlockContainer"] {{
    padding: 1.8rem 2.6rem 4rem 2.6rem; max-width: 1560px;
}}
#MainMenu, footer {{ visibility: hidden; }}

/* Streamlit 자체 규칙이 `[data-testid=stMarkdownContainer] h1` 같은 형태라
   맨 h1 선택자는 밀린다(실측: 지정 1.72rem 인데 44px 로 렌더). 컨테이너를 함께 적어
   특이도를 맞춘다. */
h1, h2, h3, h4,
[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4,
[data-testid="stHeading"] h1, [data-testid="stHeading"] h2, [data-testid="stHeading"] h3 {{
    color: {INK}; letter-spacing: -0.02em; font-weight: 700; padding: 0;
}}
h1, [data-testid="stMarkdownContainer"] h1, [data-testid="stHeading"] h1 {{ font-size: 1.9rem; }}
h2, [data-testid="stMarkdownContainer"] h2, [data-testid="stHeading"] h2 {{ font-size: 1.4rem; }}
h3, [data-testid="stMarkdownContainer"] h3, [data-testid="stHeading"] h3 {{ font-size: 1.18rem; }}
h4, [data-testid="stMarkdownContainer"] h4 {{ font-size: 1.04rem; }}
h5, [data-testid="stMarkdownContainer"] h5 {{
    font-size: .95rem; font-weight: 700; color: {TEXT_SUB};
    letter-spacing: .01em; margin-bottom: .5rem;
}}
p, li, label, span, div {{ color: {TEXT}; }}
a {{ color: {INFO}; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
hr {{ border-color: {BORDER}; margin: 1.4rem 0; }}

/* ── 사이드바 ────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: {NAV_BG};
    border-right: 1px solid #241F1B;
    width: 286px !important;
}}
[data-testid="stSidebar"] * {{ color: {NAV_TEXT}; }}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{ padding-top: 0.6rem; }}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
    color: #FFFFFF;
}}
[data-testid="stSidebarCollapseButton"] button {{ color: {NAV_TEXT} !important; }}

/* 사이드바 라디오를 네비게이션 항목처럼 */
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 2px; }}
[data-testid="stSidebar"] [role="radiogroup"] > label {{
    padding: 12px 14px; border-radius: 9px; margin: 0; cursor: pointer;
    transition: background .12s ease;
}}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover {{ background: #423D37; }}
[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child {{ display: none; }}
[data-testid="stSidebar"] [role="radiogroup"] > label p {{
    font-size: 1.02rem; font-weight: 500; color: {NAV_TEXT};
}}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {{
    background: {NAV_ACTIVE_BG};
}}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p {{
    color: {NAV_ACTIVE_TEXT}; font-weight: 700;
}}

/* ── 브랜드 헤더 (사이드바 상단) ─────────────────────────── */
.kb-brand {{
    display: flex; align-items: center; gap: 10px;
    padding: 14px 12px 16px 12px; margin-bottom: 6px;
    border-bottom: 1px solid #453F39;
}}
.kb-brand .mark {{
    width: 38px; height: 38px; border-radius: 10px; background: {YELLOW};
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 1rem; color: #26221E; letter-spacing: -0.04em;
    flex: 0 0 38px;
}}
.kb-brand .txt {{ line-height: 1.25; }}
.kb-brand .txt .t1 {{ font-size: 1.14rem; font-weight: 700; color: #FFFFFF; }}
.kb-brand .txt .t2 {{ font-size: 0.82rem; color: #A79E93; }}
.kb-navlabel {{
    padding: 16px 14px 7px 14px; font-size: 0.78rem; font-weight: 700;
    letter-spacing: .08em; color: #8C8378;
}}

/* ── 페이지 헤더 ─────────────────────────────────────────── */
.kb-page {{ margin-bottom: 1.1rem; }}
.kb-page .eyebrow {{
    display: inline-block; font-size: .78rem; font-weight: 700; letter-spacing: .07em;
    color: {TEXT_SUB}; background: {SURFACE}; border: 1px solid {BORDER};
    padding: 4px 11px; border-radius: 999px; margin-bottom: 10px;
}}
.kb-page h1 {{ margin: 0 0 4px 0; }}
.kb-page .sub {{ color: {TEXT_SUB}; font-size: 1.02rem; margin: 0; line-height: 1.6; }}
.kb-rule {{ height: 3px; width: 42px; background: {YELLOW}; border-radius: 2px; margin: 10px 0 0 0; }}

/* ── 카드 / 컨테이너 ─────────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {{
    border-radius: 12px;
}}
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {SURFACE}; border: 1px solid {BORDER} !important;
    border-radius: 12px; box-shadow: 0 1px 2px rgba(38,34,30,.04);
}}
.kb-card {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 16px 18px; box-shadow: 0 1px 2px rgba(38,34,30,.04);
}}

/* ── 지표(metric) ────────────────────────────────────────── */
[data-testid="stMetric"] {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 17px 19px; box-shadow: 0 1px 2px rgba(38,34,30,.04);
}}
[data-testid="stMetricLabel"] p {{
    font-size: .92rem !important; color: {TEXT_SUB} !important; font-weight: 600;
}}
[data-testid="stMetricValue"] {{
    font-size: 2rem !important; font-weight: 700; color: {INK};
    letter-spacing: -0.02em;
}}
[data-testid="stMetricDelta"] {{ font-size: .88rem; }}

/* ── 배지 ────────────────────────────────────────────────── */
.kb-chip {{
    display: inline-flex; align-items: center; gap: 5px; font-size: .84rem;
    font-weight: 700; padding: 4px 12px; border-radius: 999px; line-height: 1.5;
    border: 1px solid transparent; white-space: nowrap;
}}
.kb-chip.g-High   {{ background: {DANGER_SOFT}; color: {DANGER}; border-color: #F0CFCB; }}
.kb-chip.g-Medium {{ background: {WARN_SOFT};   color: {WARN};   border-color: #F3DFBB; }}
.kb-chip.g-Low    {{ background: {SAFE_SOFT};   color: {SAFE};   border-color: #C6E4D6; }}
.kb-chip.g-Info   {{ background: {INFO_SOFT};   color: {INFO};   border-color: #CBDDF0; }}
.kb-chip.g-Neutral{{ background: #F2F0EC;       color: {TEXT_SUB}; border-color: {BORDER}; }}

/* ── 소견 항목 ───────────────────────────────────────────── */
.kb-finding {{
    display: flex; gap: 13px; padding: 14px 0;
    border-bottom: 1px dashed {BORDER};
}}
.kb-finding:last-child {{ border-bottom: none; }}
.kb-finding .bar {{ width: 3px; border-radius: 2px; flex: 0 0 3px; }}
.kb-finding .body {{ flex: 1; }}
.kb-finding .head {{ font-weight: 700; font-size: 1.04rem; color: {INK}; margin-bottom: 4px; }}
.kb-finding .desc {{ font-size: .97rem; color: {TEXT}; line-height: 1.68; }}
.kb-finding .src  {{ font-size: .82rem; color: {TEXT_MUTED}; margin-top: 5px; }}

/* ── 버튼 ────────────────────────────────────────────────── */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
    border-radius: 8px; border: 1px solid {BORDER_STRONG}; background: {SURFACE};
    color: {TEXT}; font-weight: 600; font-size: .95rem; padding: .5rem 1.15rem;
    transition: all .12s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    border-color: {YELLOW_DEEP}; color: {INK}; background: {YELLOW_SOFT};
}}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{
    background: {YELLOW}; border-color: {YELLOW}; color: {INK}; font-weight: 700;
}}
.stButton > button[kind="primary"]:hover {{ background: {YELLOW_DEEP}; border-color: {YELLOW_DEEP}; }}

/* ── 입력 ────────────────────────────────────────────────── */
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-baseweb="select"] > div {{
    border-radius: 8px !important; border-color: {BORDER_STRONG} !important;
    background: {SURFACE} !important;
}}
[data-testid="stTextInput"] input:focus {{
    border-color: {YELLOW_DEEP} !important; box-shadow: 0 0 0 3px rgba(255,204,0,.22) !important;
}}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{ background: {YELLOW_DEEP}; }}

/* ── 탭 ──────────────────────────────────────────────────── */
[data-baseweb="tab-list"] {{ gap: 2px; border-bottom: 1px solid {BORDER}; }}
[data-baseweb="tab"] {{
    padding: 11px 19px; font-weight: 600; font-size: 1rem; color: {TEXT_SUB};
    border-radius: 8px 8px 0 0;
}}
[data-baseweb="tab"][aria-selected="true"] {{ color: {INK}; background: {SURFACE}; }}
[data-baseweb="tab-highlight"] {{ background: {YELLOW} !important; height: 3px; }}
[data-baseweb="tab-border"] {{ display: none; }}

/* ── 표 ──────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 10px; overflow: hidden; }}

/* ── 알림 ────────────────────────────────────────────────── */
[data-testid="stAlert"] {{ border-radius: 10px; border-width: 1px; }}

/* ── expander ────────────────────────────────────────────── */
[data-testid="stExpander"] {{
    border: 1px solid {BORDER}; border-radius: 10px; background: {SURFACE};
}}
[data-testid="stExpander"] summary {{ font-weight: 600; font-size: 1rem; color: {TEXT}; }}

/* ── 채팅 ────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 12px 14px;
}}
[data-testid="stChatInput"] {{ border-radius: 10px; border-color: {BORDER_STRONG}; }}

/* ── 스크롤바 ────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-thumb {{ background: {BORDER_STRONG}; border-radius: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
</style>
"""


def setup(page_title: str = "FranSCORE", page_icon: str = "◆") -> None:
    """페이지 설정 + CSS 주입. 앱 최상단에서 딱 한 번 호출한다."""
    st.set_page_config(page_title=page_title, page_icon=page_icon,
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 컴포넌트
# ---------------------------------------------------------------------------

def sidebar_brand(title: str, subtitle: str) -> None:
    st.sidebar.markdown(
        f"<div class='kb-brand'><div class='mark'>KB</div>"
        f"<div class='txt'><div class='t1'>{title}</div>"
        f"<div class='t2'>{subtitle}</div></div></div>",
        unsafe_allow_html=True)


def sidebar_label(text: str) -> None:
    st.sidebar.markdown(f"<div class='kb-navlabel'>{text}</div>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    bits = ["<div class='kb-page'>"]
    if eyebrow:
        bits.append(f"<span class='eyebrow'>{eyebrow}</span>")
    bits.append(f"<h1>{title}</h1>")
    if subtitle:
        bits.append(f"<p class='sub'>{subtitle}</p>")
    bits.append("<div class='kb-rule'></div></div>")
    st.markdown("".join(bits), unsafe_allow_html=True)


def chip(text: str, kind: str = "Neutral") -> str:
    """인라인 배지 HTML. kind: High/Medium/Low/Info/Neutral."""
    return f"<span class='kb-chip g-{kind}'>{text}</span>"


def grade_chip(grade: str) -> str:
    g = str(grade)
    return chip(GRADE_KR.get(g, g), g if g in GRADE_COLOR else "Neutral")


def finding_html(title: str, desc: str, severity: str, source: str = "") -> str:
    """진단 소견 한 줄. severity: High/Medium/Low."""
    color = GRADE_COLOR.get(severity, TEXT_MUTED)
    src = f"<div class='src'>{source}</div>" if source else ""
    return (f"<div class='kb-finding'><div class='bar' style='background:{color}'></div>"
            f"<div class='body'><div class='head'>{title}</div>"
            f"<div class='desc'>{desc}</div>{src}</div></div>")
