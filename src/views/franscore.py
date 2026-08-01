"""FRANSCORE — 브랜드 리스크 현황과 개별 브랜드 상세를 하나의 화면에서.

왜 합쳤나
  '한눈에 보기'와 '브랜드 조회'는 사실 같은 일의 두 단계다 — 무엇을 볼지 고르고,
  고른 것을 자세히 본다. 화면이 나뉘어 있으면 목록에서 이름을 외워 다른 화면에
  다시 입력해야 했다. 여기서는 어느 브랜드든 누르면 그 자리에서 상세로 들어간다.
"""
from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import theme
from src.brand_search import ALIASES, normalize, search
from src.views import common as C

_SEL = "fs_selected"          # 상세를 보고 있는 brand_id (None 이면 목록)
_WATCH_N = 8


def render() -> None:
    df, meta = C.load_scores()
    if df is None:
        theme.page_header("FRANSCORE", "아직 평가 결과가 없습니다.", eyebrow="브랜드 리스크")
        st.warning("`python run_pipeline.py --step score` 를 먼저 실행하십시오.")
        return

    sel = st.session_state.get(_SEL)
    if sel is not None:
        hit = df[df["brand_id"].astype(str) == str(sel)]
        if not hit.empty:
            _detail_screen(hit.iloc[0])
            return
        st.session_state[_SEL] = None      # 데이터가 갱신돼 사라진 브랜드
    _list_screen(df, meta)


def select_brand(brand_id: str) -> None:
    st.session_state[_SEL] = str(brand_id)


# ---------------------------------------------------------------------------
# 목록 화면
# ---------------------------------------------------------------------------

def _list_screen(df: pd.DataFrame, meta: dict) -> None:
    yr = meta.get("scored_year", "-")
    theme.page_header(
        "FRANSCORE",
        f"{yr}년 공시 기준 · {len(df):,}개 프랜차이즈 브랜드의 리스크를 평가했습니다. "
        "브랜드를 누르면 그 브랜드의 진단 근거를 전부 볼 수 있습니다.",
        eyebrow="브랜드 리스크")

    _search_box(df)
    _kpis(df)

    st.write("")
    left, right = st.columns([1.55, 1])
    with left:
        _watchlist(df)
    with right:
        with st.container(border=True):
            st.markdown("##### 등급은 이렇게 나눕니다")
            st.markdown(C.grade_legend_html(C.grade_bounds(C._mtime(
                C.out_dir() / "scores_latest.csv"))), unsafe_allow_html=True)
        _distribution(df)
        _industry(df)

    C.refresh_footer()


def _search_box(df: pd.DataFrame) -> None:
    """자동완성 검색. 고르면 곧바로 상세로 들어간다."""
    q = st.selectbox(
        "브랜드 검색", options=_search_options(df), index=None,
        placeholder="브랜드 이름을 입력하세요 — 한 글자만 쳐도 후보가 나옵니다",
        accept_new_options=True, label_visibility="collapsed", key="fs_query")
    if not q or not str(q).strip():
        return
    query = _clean_option(q)
    hit, near = search(df, query)
    if hit.empty:
        _not_found(query, near)
        return
    hit = hit.assign(_n=pd.to_numeric(hit["n_stores"], errors="coerce").fillna(0))
    hit = hit.sort_values("_n", ascending=False)
    if len(hit) == 1:
        select_brand(str(hit.iloc[0]["brand_id"]))
        st.rerun()
    st.caption(f"'{query}' 로 {len(hit)}개 브랜드가 검색됐습니다. 가맹점 수가 많은 순입니다.")
    cols = st.columns(min(3, len(hit)))
    for i, (_, r) in enumerate(hit.head(6).iterrows()):
        with cols[i % len(cols)]:
            _brand_card(r, key=f"q{i}")
    st.divider()


@st.cache_data(show_spinner=False)
def _search_options(df: pd.DataFrame) -> list[str]:
    """검색 후보 — 가맹점이 많은 브랜드가 위에. 통칭도 함께 걸리게 한다."""
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
    return re.sub(r"\s{2,}\([^)]*\)\s*$", "", str(q)).strip()


def _not_found(q: str, near: list[str]) -> None:
    st.warning(f"'{q}' 로 평가된 브랜드를 찾지 못했습니다.")
    if near:
        st.caption("혹시 이것을 찾으셨나요? " + " · ".join(near))
    panel = C.load_panel(full=True)
    if panel is None:
        return
    allb = panel[["brand_name", "industry_major", "year", "n_stores"]] \
        .drop_duplicates("brand_name")
    h2, _ = search(allb, q)
    if not h2.empty:
        st.info("**공시 데이터에는 있지만 평가 대상이 아닙니다.** 추세를 계산하려면 "
                "최근 3년 연속 관측과 일정 규모가 필요합니다.\n\n공시 기록: "
                + ", ".join(h2["brand_name"].astype(str).head(5)))
    else:
        st.info("**공시 데이터 자체에 없습니다.** 직영으로만 운영하는 브랜드는 "
                "가맹사업이 아니어서 공정거래위원회 공시 대상이 아닙니다.")


def _kpis(df: pd.DataFrame) -> None:
    high = df[df["risk_grade"] == "High"]
    med = df[df["risk_grade"] == "Medium"]
    n_stores = pd.to_numeric(high.get("n_stores"), errors="coerce").fillna(0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("평가 브랜드", f"{len(df):,}")
    c2.metric("주의", f"{len(high):,}", help="즉시 점검이 필요한 상위 10% 브랜드입니다.")
    c3.metric("관찰", f"{len(med):,}", help="추이를 지켜봐야 하는 구간입니다.")
    c4.metric("주의 · 100점포 이상", f"{int((n_stores >= 100).sum()):,}",
              help="규모가 커서 부실 시 여신 영향이 큰 브랜드입니다.")


def _watchlist(df: pd.DataFrame) -> None:
    st.markdown("### 지금 봐야 할 브랜드")
    st.caption("리스크와 규모를 함께 본 순서입니다. 규모가 크면 같은 확률이라도 손실이 큽니다.")
    high = df[df["risk_grade"] == "High"].copy()
    if high.empty:
        st.success("주의 등급에 해당하는 브랜드가 없습니다.")
        return
    high["_priority"] = (pd.to_numeric(high["pd_1y"], errors="coerce").fillna(0)
                         * pd.to_numeric(high["n_stores"], errors="coerce").fillna(0))
    diag = C.load_diagnosis_summary()
    dmap = ({str(r["brand_id"]): r for _, r in diag.iterrows()}
            if diag is not None and not diag.empty else {})

    for i, (_, r) in enumerate(high.nlargest(_WATCH_N, "_priority").iterrows()):
        bid = str(r["brand_id"])
        with st.container(border=True):
            head, val = st.columns([3.1, 1])
            with head:
                st.markdown(
                    f"<div style='display:flex;gap:12px;align-items:center'>"
                    f"{C.brand_mark_html(str(r['brand_name']), 62)}"
                    f"<div><div style='font-size:1.26rem;font-weight:700;"
                    f"color:{theme.INK};line-height:1.3'>{r['brand_name']}</div>"
                    f"<div style='font-size:.9rem;color:{theme.TEXT_SUB}'>"
                    f"{r.get('industry_mid') or r.get('industry_major') or '-'} · "
                    f"가맹점 {int(r['n_stores']):,}개</div></div></div>",
                    unsafe_allow_html=True)
            with val:
                st.markdown(
                    f"<div style='text-align:right'>"
                    f"{C.signal_html(str(r['risk_grade']), r['pd_1y'])}</div>",
                    unsafe_allow_html=True)
            d = dmap.get(bid)
            if d is not None:
                st.markdown(
                    f"<div style='font-size:.99rem;color:{theme.TEXT};margin-top:8px;"
                    f"line-height:1.6'>{d['headline_detail']}</div>",
                    unsafe_allow_html=True)
                if str(d.get("categories") or ""):
                    cats = " ".join(theme.chip(c, "Neutral")
                                    for c in str(d["categories"]).split("·") if c)
                    st.markdown(
                        f"<div style='margin-top:8px'>{cats}"
                        f"<span style='font-size:.88rem;color:{theme.TEXT_MUTED};"
                        f"margin-left:8px'>소견 {int(d['n_risk'])}건</span></div>",
                        unsafe_allow_html=True)
            if st.button("진단 근거 보기", key=f"w{i}_{bid}", use_container_width=True):
                select_brand(bid)
                st.rerun()


def _brand_card(r: pd.Series, *, key: str) -> None:
    """작은 카드 — 검색 결과·인기 브랜드에 공통으로 쓴다."""
    bid = str(r["brand_id"])
    with st.container(border=True):
        st.markdown(
            f"<div style='display:flex;gap:10px;align-items:center'>"
            f"{C.brand_mark_html(str(r['brand_name']), 54)}"
            f"<div style='min-width:0'><div style='font-weight:700;font-size:1.06rem;"
            f"color:{theme.INK};overflow:hidden;text-overflow:ellipsis;"
            f"white-space:nowrap'>{r['brand_name']}</div>"
            f"<div style='font-size:.88rem;color:{theme.TEXT_SUB}'>"
            f"가맹점 {int(pd.to_numeric(r['n_stores'], errors='coerce') or 0):,}개</div>"
            f"</div></div>"
            f"<div style='margin-top:8px'>"
            f"{C.signal_html(str(r['risk_grade']), r['pd_1y'], size=11)}</div>",
            unsafe_allow_html=True)
        if st.button("상세 보기", key=f"c{key}_{bid}", use_container_width=True):
            select_brand(bid)
            st.rerun()


def _distribution(df: pd.DataFrame) -> None:
    """등급 색으로 나눈 분포 — '무엇의 분포인지'를 축과 색이 함께 말한다."""
    st.markdown("### 브랜드 리스크 분포")
    fig = go.Figure()
    for g in ("Low", "Medium", "High"):
        p = pd.to_numeric(df.loc[df["risk_grade"] == g, "pd_1y"],
                          errors="coerce").dropna() * 100
        if p.empty:
            continue
        fig.add_trace(go.Histogram(
            x=p, name=f"{theme.GRADE_KR[g]} {len(p):,}개",
            xbins={"start": 0, "end": 100, "size": 2},
            marker={"color": theme.GRADE_COLOR[g], "line": {"width": 0}},
            hovertemplate=(f"{theme.GRADE_KR[g]}<br>{C.RISK_LABEL} "
                           "%{x:.0f}%대<br>브랜드 <b>%{y}</b>개<extra></extra>")))
    fig.update_layout(barmode="stack", height=225, bargap=0.03,
                      xaxis_title=None, yaxis_title=None,
                      legend={"orientation": "h", "y": 1.16, "x": 0,
                              "font": {"size": 11}},
                      margin={"l": 4, "r": 4, "t": 30, "b": 4})
    fig.update_xaxes(range=[0, 100], dtick=25, ticksuffix="%")
    theme.plot(fig, key="fs_hist")
    p = pd.to_numeric(df["pd_1y"], errors="coerce").dropna() * 100
    st.caption(f"가로축은 브랜드 리스크(0~100%), 세로축은 그 구간에 속한 브랜드 수입니다. "
               f"대부분은 왼쪽 낮은 구간에 몰려 있고 오른쪽 꼬리가 점검 대상입니다 — "
               f"중앙값 {p.median():.1f}%, 가장 높은 브랜드 {p.max():.1f}%.")


def _industry(df: pd.DataFrame) -> None:
    """업종별 등급 구성 — 비율만 보면 표본 3개짜리 업종이 1위가 된다."""
    st.markdown("### 업종별 리스크 구성")
    key = "industry_mid" if "industry_mid" in df.columns else "industry_major"
    tab = (df.groupby(key)["risk_grade"].value_counts().unstack(fill_value=0)
             .reindex(columns=["High", "Medium", "Low"], fill_value=0))
    tab["합계"] = tab.sum(axis=1)
    tab = tab[tab["합계"] >= 20]
    if tab.empty:
        st.caption("표본이 20개 이상인 업종이 없습니다.")
        return
    tab["비율"] = tab["High"] / tab["합계"]
    tab = tab.sort_values("비율").tail(10)

    fig = go.Figure()
    for g in ("High", "Medium", "Low"):
        fig.add_trace(go.Bar(
            y=tab.index, x=(tab[g] / tab["합계"] * 100), orientation="h",
            name=theme.GRADE_KR[g], marker={"color": theme.GRADE_COLOR[g]},
            customdata=tab[g],
            hovertemplate=(f"%{{y}}<br>{theme.GRADE_KR[g]} <b>%{{customdata}}</b>개 "
                           "(%{x:.0f}%)<extra></extra>")))
    fig.update_layout(barmode="stack", height=max(230, 30 * len(tab)),
                      legend={"orientation": "h", "y": 1.1, "x": 0,
                              "font": {"size": 11}},
                      margin={"l": 4, "r": 4, "t": 26, "b": 4},
                      xaxis_title=None, yaxis_title=None)
    fig.update_xaxes(range=[0, 100], dtick=25, ticksuffix="%")
    theme.plot(fig, key="fs_ind")
    worst = tab.index[-1]
    st.caption(f"주의 비율이 높은 업종 순입니다. 표본 20개 미만 업종은 비율이 흔들려 "
               f"제외했습니다. 지금은 **{worst}** 가 가장 높습니다 — "
               f"{int(tab.loc[worst, 'High'])}개 / {int(tab.loc[worst, '합계'])}개. "
               f"이 업종에 여신이 쏠려 있다면 함께 봐야 합니다.")


# ---------------------------------------------------------------------------
# 상세 화면
# ---------------------------------------------------------------------------

def _detail_screen(r: pd.Series) -> None:
    bid, name = str(r["brand_id"]), str(r["brand_name"])
    grade = str(r["risk_grade"])

    if st.button("← 목록으로", key="fs_back"):
        st.session_state[_SEL] = None
        st.rerun()

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
        st.markdown(_summary_sentence(r), unsafe_allow_html=True)
    with h2:
        theme.plot(C.risk_gauge(float(r["pd_1y"])), key=f"g_{bid}")
        rank = r.get("pd_rank_pct")
        sub = (f"전체 {C.load_scores()[0].shape[0]:,}개 중 "
               f"상위 {(1 - float(rank)) * 100:.1f}%" if pd.notna(rank) else "")
        st.markdown(
            f"<div style='text-align:center;margin-top:-14px'>"
            f"<div style='font-size:.9rem;color:{theme.TEXT_SUB}'>"
            f"{C.RISK_LABEL} · {sub}</div>"
            f"<div style='margin-top:10px'>{C.signal_html(grade, None, show_value=False)}"
            f"</div></div>", unsafe_allow_html=True)

    st.write("")
    tabs = st.tabs(["진단 소견", "공시 추이", "가맹본부 재무", "검색 수요"])
    with tabs[0]:
        _tab_findings(C.load_findings(bid))
    with tabs[1]:
        _tab_trend(bid)
    with tabs[2]:
        _tab_hq(bid)
    with tabs[3]:
        _tab_demand(bid, name)
    C.refresh_footer()


def _summary_sentence(r: pd.Series) -> str:
    """상세 첫 문장 — 반드시 이 브랜드의 실제 수치로만 쓴다.

    '위험합니다' 같은 형용사는 근거가 없으면 아무것도 말하지 않는 것과 같다.
    여기서는 등급·확률·순위·규모라는 네 가지 관측값만으로 문장을 만든다.
    """
    grade = str(r["risk_grade"])
    p = float(r["pd_1y"]) * 100
    n = int(pd.to_numeric(r.get("n_stores"), errors="coerce") or 0)
    rank = r.get("pd_rank_pct")
    rank_txt = (f"평가 대상 가운데 상위 {(1 - float(rank)) * 100:.1f}%"
                if pd.notna(rank) else "순위 미산출")
    b = C.grade_bounds(C._mtime(C.out_dir() / "scores_latest.csv"))
    cut = (f"주의 경계는 {b['high_cut']:.1f}%, 관찰 경계는 {b['medium_cut']:.1f}%입니다."
           if b else "")
    return (f"<div style='margin-top:10px;font-size:.97rem;color:{theme.TEXT_SUB};"
            f"line-height:1.65'>이 브랜드의 {C.RISK_LABEL}는 <b>{p:.1f}%</b>로 "
            f"<b>{theme.GRADE_KR.get(grade, grade)}</b> 등급이며, {rank_txt}입니다. "
            f"가맹점은 {n:,}개입니다. {cut}</div>")


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


def _tab_trend(brand_id: str) -> None:
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


def _tab_hq(brand_id: str) -> None:
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
        st.dataframe(show.set_index("결산연도"), width="stretch",
                     column_config={c: st.column_config.NumberColumn(c, format="%.1f 억")
                                    for c in show.columns[1:]})
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=show["결산연도"], y=show["매출"], name="매출",
                             marker_color=theme.YELLOW_DEEP))
        fig.add_trace(go.Scatter(x=show["결산연도"], y=show["영업이익"], name="영업이익",
                                 mode="lines+markers",
                                 line={"color": theme.INFO, "width": 2.5}, yaxis="y2"))
        fig.update_layout(height=214, yaxis={"title": None, "ticksuffix": "억"},
                          yaxis2={"overlaying": "y", "side": "right",
                                  "showgrid": False, "ticksuffix": "억"},
                          margin={"l": 4, "r": 4, "t": 24, "b": 4})
        theme.plot(fig, key=f"hq_{brand_id}")

    rc = fin["rcept_no"].dropna()
    if len(rc):
        st.markdown(f"[감사보고서 원문 보기 (DART)]"
                    f"(https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rc.iloc[-1]})")


def _tab_demand(brand_id: str, name: str) -> None:
    obj = C.load_demand()
    if not obj.get("enabled"):
        st.info("네이버 검색어트렌드가 아직 연결되지 않았습니다. "
                "연결하면 이 브랜드를 찾는 사람이 늘고 있는지 줄고 있는지를 "
                "월 단위로 볼 수 있습니다 — 공시보다 1~2년 빠른 신호입니다.")
        return
    d = (obj.get("brands") or {}).get(brand_id)
    if not d:
        st.caption(f"'{name}'의 검색 추세는 아직 수집되지 않았습니다. "
                   "검색수요는 가맹점 수가 많은 상위 브랜드부터 수집합니다.")
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
                                 line={"color": theme.TEXT_SUB, "width": 1.8,
                                       "dash": "dot"}))
    fig.update_layout(height=230, margin={"l": 4, "r": 4, "t": 24, "b": 4})
    theme.plot(fig, key=f"dm_{brand_id}")
    st.caption("네이버 데이터랩 검색어트렌드 · 기간 내 최대값을 100으로 놓은 상대지수입니다. "
               "절대 검색 건수는 네이버가 공개하지 않습니다.")
