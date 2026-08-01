"""한눈에 보기 — 오늘 무엇을 봐야 하는가."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import theme
from src.views import common as C


def render() -> None:
    df, meta = C.load_scores()
    if df is None:
        st.warning("아직 평가 결과가 없습니다.")
        return
    diag = C.load_diagnosis_summary()
    yr = meta.get("scored_year", "-")

    theme.page_header(
        "브랜드 리스크 현황",
        f"{yr}년 공시 기준 · {len(df):,}개 프랜차이즈 브랜드가 내년에 구조적으로 "
        "꺾일 가능성을 평가했습니다.",
        eyebrow="현황")

    high = df[df["risk_grade"] == "High"]
    med = df[df["risk_grade"] == "Medium"]
    n_stores = pd.to_numeric(high.get("n_stores"), errors="coerce").fillna(0)
    big_high = high[n_stores >= 100]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("평가 브랜드", f"{len(df):,}")
    c2.metric("주의", f"{len(high):,}", help="위험도 상위 10%. 우선 점검 대상입니다.")
    c3.metric("관찰", f"{len(med):,}")
    c4.metric("주의 · 100점포 이상", f"{len(big_high):,}",
              help="규모가 커서 부실 시 여신 영향이 큰 브랜드입니다.")

    st.write("")
    left, right = st.columns([1.55, 1])
    with left:
        _watchlist(df, diag)
    with right:
        _distribution(df)
        _industry_table(df)


def _watchlist(df: pd.DataFrame, diag: pd.DataFrame | None) -> None:
    st.markdown("### 지금 봐야 할 브랜드")
    st.caption("위험도와 규모를 함께 본 순서입니다. 규모가 크면 같은 확률이라도 손실이 큽니다.")

    high = df[df["risk_grade"] == "High"].copy()
    if high.empty:
        st.success("주의 등급에 해당하는 브랜드가 없습니다.")
        return
    high["_priority"] = (pd.to_numeric(high["pd_1y"], errors="coerce").fillna(0)
                         * pd.to_numeric(high["n_stores"], errors="coerce").fillna(0))
    watch = high.nlargest(8, "_priority")

    dmap: dict[str, pd.Series] = {}
    if diag is not None and not diag.empty:
        dmap = {str(r["brand_id"]): r for _, r in diag.iterrows()}

    for _, r in watch.iterrows():
        d = dmap.get(str(r["brand_id"]))
        with st.container(border=True):
            head, val = st.columns([3.1, 1])
            with head:
                st.markdown(
                    f"<div style='display:flex;gap:12px;align-items:center'>"
                    f"{C.brand_mark_html(str(r['brand_name']), 46)}"
                    f"<div><div style='font-size:1.26rem;font-weight:700;"
                    f"color:{theme.INK};line-height:1.3'>{r['brand_name']}</div>"
                    f"<div style='font-size:.9rem;color:{theme.TEXT_SUB}'>"
                    f"{r.get('industry_mid') or r.get('industry_major') or '-'} · "
                    f"가맹점 {int(r['n_stores']):,}개</div></div></div>",
                    unsafe_allow_html=True)
            with val:
                st.markdown(
                    f"<div style='text-align:right'>"
                    f"<div style='font-size:1.75rem;font-weight:700;color:{theme.DANGER}'>"
                    f"{float(r['pd_1y']) * 100:.1f}%</div>"
                    f"<div style='font-size:.86rem;color:{theme.TEXT_SUB}'>"
                    f"1년 내 악화 가능성</div></div>", unsafe_allow_html=True)
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
            else:
                st.caption("상세 진단이 아직 생성되지 않았습니다.")


def _distribution(df: pd.DataFrame) -> None:
    st.markdown("### 위험도 분포")
    p = pd.to_numeric(df["pd_1y"], errors="coerce").dropna() * 100
    fig = go.Figure(go.Histogram(
        x=p, nbinsx=36, marker={"color": theme.YELLOW_DEEP, "line": {"width": 0}},
        hovertemplate="악화 가능성 %{x:.1f}%<br>브랜드 <b>%{y}</b>개<extra></extra>"))
    fig.update_layout(height=190, showlegend=False, bargap=0.04,
                      xaxis_title=None, yaxis_title=None,
                      margin={"l": 4, "r": 4, "t": 6, "b": 4})
    fig.update_xaxes(ticksuffix="%")
    theme.plot(fig, key="ov_hist")
    st.caption(f"대부분의 브랜드는 낮은 구간에 몰려 있고, 오른쪽 꼬리가 점검 대상입니다. "
               f"중앙값 {p.median():.1f}% · 상위 10% 경계 {p.quantile(0.9):.1f}%")


def _industry_table(df: pd.DataFrame) -> None:
    st.markdown("### 업종별 주의 비율")
    key = "industry_mid" if "industry_mid" in df.columns else "industry_major"
    tab = (df.groupby(key)["risk_grade"].value_counts().unstack(fill_value=0)
             .reindex(columns=["High", "Medium", "Low"], fill_value=0))
    tab["합계"] = tab.sum(axis=1)
    tab = tab[tab["합계"] >= 10]
    if tab.empty:
        st.caption("표본이 10개 이상인 업종이 없습니다.")
        return
    tab["비율"] = tab["High"] / tab["합계"]
    tab = tab.sort_values("비율", ascending=True).tail(10)
    fig = C.bar_chart(tab.index.tolist(), (tab["비율"] * 100).round(1).tolist(),
                      colors=theme.DANGER, unit="%")
    fig.update_layout(height=max(210, 26 * len(tab)))
    fig.update_xaxes(ticksuffix="%")
    theme.plot(fig, key="ov_ind")
    st.caption("특정 업종에 여신이 쏠려 있다면 그 업종의 주의 비율을 함께 봐야 합니다.")
