"""브랜드 조회 — 이름을 치면 그 브랜드의 모든 것을 한 화면에."""
from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import theme
from src.brand_search import ALIASES, normalize, search
from src.views import common as C


def render() -> None:
    df, meta = C.load_scores()
    theme.page_header(
        "브랜드 조회",
        "브랜드 이름을 입력하면 위험도·진단 소견·공시 추이·본부 재무를 함께 보여줍니다.",
        eyebrow="조회")
    if df is None:
        st.warning("아직 평가 결과가 없습니다.")
        return

    # ── 검색창 ──
    # st.text_input 이 아니라 selectbox 를 쓰는 이유: text_input 은 **Enter 를 눌러야**
    # 값이 서버로 넘어와서, 한 글자 칠 때마다 후보가 뜨는 자동완성이 원리상 불가능하다.
    # selectbox 는 타이핑에 맞춰 목록을 실시간으로 좁혀 준다(네이티브 타입어헤드).
    # accept_new_options=True 로 목록에 없는 통칭('메가커피')도 그대로 받아
    # 아래 search() 의 별칭 규칙으로 넘긴다 — 두 경로가 모두 살아 있다.
    q = st.selectbox(
        "브랜드 이름", options=_search_options(df), index=None,
        placeholder="브랜드 이름을 입력하세요 — 한 글자만 쳐도 후보가 나옵니다",
        accept_new_options=True, label_visibility="collapsed", key="brand_q")
    st.caption("목록에 없는 이름을 직접 입력해 조회할 수도 있습니다 "
               "(목록 맨 아래 항목을 선택).")

    if not q or not str(q).strip():
        _landing(df, meta)
        return

    query = _clean_option(q)
    hit, near = search(df, query)
    if hit.empty:
        _not_found(query, near)
        return

    hit = hit.copy()
    hit["_n"] = pd.to_numeric(hit["n_stores"], errors="coerce").fillna(0)
    hit = hit.sort_values("_n", ascending=False)
    if len(hit) > 1:
        st.caption(f"{len(hit)}개 브랜드가 검색됐습니다. 가맹점 수가 많은 순입니다.")
    for _, r in hit.head(5).iterrows():
        _brand_detail(r)


# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _search_options(df: pd.DataFrame) -> list[str]:
    """검색 후보 목록 — 가맹점이 많은 브랜드가 위에 오도록 정렬한다.

    통칭으로 찾는 브랜드는 등록명 뒤에 통칭을 덧붙여 타이핑으로도 걸리게 한다
    ('메가커피' 를 쳐도 '메가엠지씨커피(MEGA MGC COFFEE)' 가 후보에 뜬다).
    """
    d = df.assign(_n=pd.to_numeric(df["n_stores"], errors="coerce").fillna(0))
    names = d.sort_values("_n", ascending=False)["brand_name"].astype(str).tolist()
    alias_by_name: dict[str, list[str]] = {}
    for alias, token in ALIASES.items():
        for nm in names:
            if normalize(token) in normalize(nm):
                alias_by_name.setdefault(nm, [])
                if alias not in alias_by_name[nm]:
                    alias_by_name[nm].append(alias)
    out = []
    for nm in dict.fromkeys(names):
        extra = alias_by_name.get(nm)
        out.append(f"{nm}  ({'·'.join(extra[:2])})" if extra else nm)
    return out


def _clean_option(q: str) -> str:
    """선택지 라벨에서 통칭 꼬리표를 떼어 실제 브랜드명만 남긴다."""
    return re.sub(r"\s{2,}\([^)]*\)\s*$", "", str(q)).strip()


def _landing(df: pd.DataFrame, meta: dict) -> None:
    diag = C.load_diagnosis_summary()
    st.caption(f"{meta.get('scored_year', '-')}년 공시 기준 "
               f"{len(df):,}개 브랜드가 조회 가능합니다.")
    st.markdown("##### 가맹점이 많은 브랜드")
    top = df.assign(_n=pd.to_numeric(df["n_stores"], errors="coerce").fillna(0)) \
            .nlargest(12, "_n")
    dmap = ({str(r["brand_id"]): r for _, r in diag.iterrows()}
            if diag is not None and not diag.empty else {})
    cols = st.columns(3)
    for i, (_, r) in enumerate(top.iterrows()):
        with cols[i % 3], st.container(border=True):
            st.markdown(
                f"<div style='display:flex;gap:10px;align-items:center'>"
                f"{C.brand_mark_html(str(r['brand_name']), 54)}"
                f"<div style='min-width:0'><div style='font-weight:700;font-size:1.06rem;"
                f"color:{theme.INK};overflow:hidden;text-overflow:ellipsis;"
                f"white-space:nowrap'>{r['brand_name']}</div>"
                f"<div style='font-size:.88rem;color:{theme.TEXT_SUB}'>"
                f"가맹점 {int(r['n_stores']):,}개 · "
                f"{theme.GRADE_KR.get(str(r['risk_grade']), '')}</div></div></div>",
                unsafe_allow_html=True)
            d = dmap.get(str(r["brand_id"]))
            if d is not None:
                st.markdown(
                    f"<div style='font-size:.9rem;color:{theme.TEXT_SUB};margin-top:6px;"
                    f"line-height:1.5;height:2.9em;overflow:hidden'>"
                    f"{str(d['headline_detail'])[:78]}…</div>", unsafe_allow_html=True)


def _not_found(q: str, near: list[str]) -> None:
    st.warning(f"'{q}' 로 평가된 브랜드를 찾지 못했습니다.")
    if near:
        st.caption("혹시 이것을 찾으셨나요? " + " · ".join(near))
    panel = C.load_panel(full=True)
    if panel is None:
        return
    allb = panel[["brand_name", "industry_major", "year", "n_stores"]].drop_duplicates(
        "brand_name")
    h2, _ = search(allb, q)
    if not h2.empty:
        st.info(
            "**공시 데이터에는 있지만 평가 대상이 아닙니다.** 추세를 계산하려면 "
            "최근 3년 연속 관측과 일정 규모가 필요합니다.\n\n공시 기록: "
            + ", ".join(h2["brand_name"].astype(str).head(5)))
    else:
        st.info("**공시 데이터 자체에 없습니다.** 직영으로만 운영하는 브랜드는 "
                "가맹사업이 아니어서 공정거래위원회 공시 대상이 아닙니다.")


# ---------------------------------------------------------------------------

def _brand_detail(r: pd.Series) -> None:
    bid = str(r["brand_id"])
    name = str(r["brand_name"])
    grade = str(r["risk_grade"])
    findings = C.load_findings(bid)

    with st.container(border=True):
        # ── 헤더 ──
        h1, h2 = st.columns([2.6, 1])
        with h1:
            st.markdown(
                f"<div style='display:flex;gap:16px;align-items:center'>"
                f"{C.brand_mark_html(name, 84)}"
                f"<div><div style='font-size:1.75rem;font-weight:700;color:{theme.INK};"
                f"line-height:1.25;letter-spacing:-.02em'>{name} "
                f"{theme.grade_chip(grade)}</div>"
                f"<div style='font-size:.97rem;color:{theme.TEXT_SUB};margin-top:3px'>"
                f"{r.get('industry_major', '-')} · {r.get('industry_mid', '-')} · "
                f"가맹점 {int(r['n_stores']):,}개</div></div></div>",
                unsafe_allow_html=True)
            st.markdown(
                f"<div style='margin-top:14px;padding:11px 14px;border-radius:9px;"
                f"background:{theme.GRADE_SOFT.get(grade, theme.YELLOW_SOFT)};"
                f"font-size:.99rem;line-height:1.6'>"
                f"{C.GRADE_ACTION.get(grade, '')}</div>", unsafe_allow_html=True)
        with h2:
            theme.plot(C.risk_gauge(float(r["pd_1y"])), key=f"g_{bid}")
            rank = r.get("pd_rank_pct")
            if pd.notna(rank):
                st.markdown(
                    f"<div style='text-align:center;font-size:.9rem;"
                    f"color:{theme.TEXT_SUB};margin-top:-14px'>1년 내 악화 가능성 · "
                    f"전체 중 상위 {(1 - float(rank)) * 100:.1f}%</div>",
                    unsafe_allow_html=True)

        st.write("")
        tabs = st.tabs(["진단 소견", "공시 추이", "가맹본부 재무", "검색 수요"])
        with tabs[0]:
            _tab_findings(findings)
        with tabs[1]:
            _tab_trend(bid, name)
        with tabs[2]:
            _tab_hq(bid, name)
        with tabs[3]:
            _tab_demand(bid, name)


def _tab_findings(findings: pd.DataFrame | None) -> None:
    if findings is None or findings.empty:
        st.info("이 브랜드의 진단 소견이 아직 생성되지 않았습니다.")
        return
    chips = C.category_summary(findings)
    if chips:
        st.markdown(f"<div style='margin-bottom:10px'>{chips}</div>",
                    unsafe_allow_html=True)
    risk = findings[findings["direction"] == "risk"]
    other = findings[findings["direction"] != "risk"]
    C.render_findings(risk)
    if not other.empty:
        with st.expander(f"완화요인·확인 필요 사항 {len(other)}건"):
            C.render_findings(other)


def _tab_trend(brand_id: str, name: str) -> None:
    panel = C.load_panel()
    if panel is None:
        st.info("공시 패널을 찾지 못했습니다.")
        return
    bp = panel[panel["brand_id"].astype(str) == brand_id].sort_values("year")
    if bp.empty:
        st.info("공시 이력이 없습니다.")
        return
    bp = bp.assign(year=bp["year"].astype(int))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**가맹점 수**")
        theme.plot(C.line_chart(bp, "year", "n_stores", "가맹점", "개"),
                   key=f"t1_{brand_id}")
    with c2:
        st.markdown("**가맹점 평균매출**")
        if bp["avg_sales"].notna().any():
            d = bp.assign(_s=pd.to_numeric(bp["avg_sales"], errors="coerce") / 1e5)
            fig = C.line_chart(d, "year", "_s", "평균매출", "억원")
            fig.update_yaxes(ticksuffix="억")
            theme.plot(fig, key=f"t2_{brand_id}")
        else:
            st.caption("평균매출이 공시에 기재되지 않았습니다.")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**개점 · 종료**")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bp["year"], y=bp["n_new"], name="신규 개점",
                             marker_color=theme.SAFE,
                             hovertemplate="%{x}년<br>신규 <b>%{y:,.0f}</b>개<extra></extra>"))
        out = (pd.to_numeric(bp["n_contract_end"], errors="coerce").fillna(0)
               + pd.to_numeric(bp["n_contract_cancel"], errors="coerce").fillna(0))
        fig.add_trace(go.Bar(x=bp["year"], y=-out, name="종료·해지",
                             marker_color=theme.DANGER,
                             hovertemplate="%{x}년<br>종료·해지 <b>%{customdata:,.0f}</b>개"
                                           "<extra></extra>", customdata=out))
        fig.update_layout(barmode="relative", height=210,
                          margin={"l": 4, "r": 4, "t": 8, "b": 4})
        theme.plot(fig, key=f"t3_{brand_id}")
    with c4:
        st.markdown("**진출 지역 수**")
        if "n_regions" in bp.columns and bp["n_regions"].notna().any():
            theme.plot(C.line_chart(bp, "year", "n_regions", "지역", "곳", fill=False),
                       key=f"t4_{brand_id}")
        else:
            st.caption("지역별 공시가 없습니다.")
    del name


def _tab_hq(brand_id: str, name: str) -> None:
    from src.dart import norm_corp
    panel = C.load_panel()
    fin_all = C.load_hq_financials()
    if panel is None or fin_all is None:
        st.info("가맹본부 재무 자료를 찾지 못했습니다.")
        return
    hit = panel[panel["brand_id"].astype(str) == brand_id].sort_values("year")
    if hit.empty:
        st.info("가맹본부를 확인하지 못했습니다.")
        return
    company = str(hit["company_name"].iloc[-1])
    fin = fin_all[fin_all["key"] == norm_corp(company)].sort_values("fiscal_year")
    if fin.empty:
        st.info(f"**{company}**는 외부감사 대상이 아니어서 감사보고서를 제출하지 않습니다. "
                "본부의 자본잠식·적자 여부를 공시로 확인할 수 없으므로, 여신 심사 시 "
                "별도 재무자료를 징구해 확인해야 합니다.")
        return

    st.markdown(f"**{company}** · 금융감독원 전자공시")
    show = fin[["fiscal_year", "assets", "liabilities", "equity", "revenue",
                "operating_income", "net_income"]].copy()
    for c in show.columns[1:]:
        show[c] = (pd.to_numeric(show[c], errors="coerce") / 1e8).round(1)
    show.columns = ["결산연도", "자산", "부채", "자본", "매출", "영업이익", "순이익"]
    show["결산연도"] = show["결산연도"].astype(int)

    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.dataframe(show.set_index("결산연도"), use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(c, format="%.1f 억")
                                    for c in show.columns[1:]})
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=show["결산연도"], y=show["매출"], name="매출",
                             marker_color=theme.YELLOW_DEEP))
        fig.add_trace(go.Scatter(x=show["결산연도"], y=show["영업이익"], name="영업이익",
                                 mode="lines+markers", line={"color": theme.INFO, "width": 2.5},
                                 yaxis="y2"))
        fig.update_layout(height=214, yaxis={"title": None, "ticksuffix": "억"},
                          yaxis2={"overlaying": "y", "side": "right", "showgrid": False,
                                  "ticksuffix": "억"},
                          margin={"l": 4, "r": 4, "t": 24, "b": 4})
        theme.plot(fig, key=f"hq_{brand_id}")

    rc = fin["rcept_no"].dropna()
    if len(rc):
        st.markdown(
            f"[감사보고서 원문 보기 (DART)]"
            f"(https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rc.iloc[-1]})")
    del name


def _tab_demand(brand_id: str, name: str) -> None:
    obj = C.load_demand()
    if not obj.get("enabled"):
        st.info("네이버 검색어트렌드가 아직 연결되지 않았습니다. "
                "연결하면 이 브랜드를 찾는 사람이 늘고 있는지 줄고 있는지를 "
                "월 단위로 볼 수 있습니다 — 공시보다 1~2년 빠른 신호입니다.")
        return
    d = (obj.get("brands") or {}).get(brand_id)
    if not d:
        st.caption(f"'{name}'의 검색 추세는 아직 수집되지 않았습니다.")
        return

    b, cy = d.get("brand_yoy"), d.get("category_yoy")
    c1, c2 = st.columns(2)
    if b is not None:
        c1.metric("브랜드 검색량 (최근 12개월 vs 직전 12개월)", f"{b * 100:+.1f}%")
    if cy is not None:
        c2.metric(f"'{d.get('category')}' 카테고리", f"{cy * 100:+.1f}%")

    series = pd.DataFrame(d.get("series") or [])
    cser = pd.DataFrame(d.get("category_series") or [])
    if series.empty:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series["d"], y=series["v"], name=str(d.get("term")),
                             mode="lines", line={"color": theme.YELLOW_DEEP, "width": 2.5},
                             fill="tozeroy", fillcolor="rgba(255,204,0,0.14)"))
    if not cser.empty:
        fig.add_trace(go.Scatter(x=cser["d"], y=cser["v"], name=str(d.get("category")),
                                 mode="lines",
                                 line={"color": theme.TEXT_SUB, "width": 1.8, "dash": "dot"}))
    fig.update_layout(height=230, margin={"l": 4, "r": 4, "t": 24, "b": 4})
    theme.plot(fig, key=f"dm_{brand_id}")
    st.caption("네이버 데이터랩 검색어트렌드 · 기간 내 최대값을 100으로 놓은 상대지수입니다. "
               "절대 검색 건수는 네이버가 공개하지 않습니다.")
