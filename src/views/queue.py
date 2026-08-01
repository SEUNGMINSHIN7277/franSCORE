"""점검 큐 — 이번 분기에 실제로 처리해야 할 브랜드 목록과 그 처리 상태.

이 화면이 하는 일은 하나다: **누가 무엇을 언제까지 확인할지 정하고, 결과를 남기는 것.**
'점수 목록을 필터링해서 CSV로 내려받는 화면'이 아니라 업무 흐름 화면이다.
배정 → 검토 → 처리결과 기록 → 반출 순서로 화면을 배치한다.
"""
from __future__ import annotations

import io
from datetime import UTC, datetime

import pandas as pd
import streamlit as st

from src import theme
from src.views import common as C

STATUS = ["미착수", "검토 중", "조치 완료", "이상 없음"]
STATUS_KIND = {"미착수": "High", "검토 중": "Medium",
               "조치 완료": "Low", "이상 없음": "Neutral"}
_KEY = "queue_state"


def _state() -> dict:
    """처리 상태 저장소 (브랜드ID → {status, owner, note}).

    ⚠️ 세션 저장이다. 브라우저를 닫으면 사라지므로 화면에 그 사실을 명시하고,
       내려받기로 반출할 수 있게 한다. '저장됐다'고 착각하게 두지 않는다.
    """
    if _KEY not in st.session_state:
        st.session_state[_KEY] = {}
    return st.session_state[_KEY]


def render() -> None:
    df, meta = C.load_scores()
    if df is None:
        st.warning("아직 평가 결과가 없습니다.")
        return
    diag = C.load_diagnosis_summary()
    yr = meta.get("scored_year", "-")

    theme.page_header(
        "점검 큐",
        f"{yr}년 공시 기준으로 우선 확인이 필요한 브랜드입니다. "
        "담당자를 지정하고 확인 결과를 기록하면 목록에서 정리됩니다.",
        eyebrow="업무")

    work = df[df["risk_grade"].isin(["High", "Medium"])].copy()
    if diag is not None and not diag.empty:
        work = work.merge(
            diag[["brand_id", "headline_detail", "n_risk", "n_high", "categories",
                  "watch_score"]],
            on="brand_id", how="left")
    else:
        work["headline_detail"] = ""
        work["watch_score"] = pd.to_numeric(work["pd_rank_pct"], errors="coerce") * 100

    state = _state()
    work["처리상태"] = work["brand_id"].astype(str).map(
        lambda b: state.get(b, {}).get("status", "미착수"))
    work["담당"] = work["brand_id"].astype(str).map(
        lambda b: state.get(b, {}).get("owner", ""))

    done = work[work["처리상태"].isin(["조치 완료", "이상 없음"])]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("점검 대상", f"{len(work):,}")
    k2.metric("미착수", f"{int((work['처리상태'] == '미착수').sum()):,}")
    k3.metric("검토 중", f"{int((work['처리상태'] == '검토 중').sum()):,}")
    k4.metric("처리 완료", f"{len(done):,}",
              delta=f"{len(done) / max(len(work), 1) * 100:.0f}%", delta_color="off")

    st.write("")
    tab_work, tab_all = st.tabs(["오늘 처리할 건", "전체 목록 · 반출"])
    with tab_work:
        _worklist(work)
    with tab_all:
        _fulltable(work, yr)


# ---------------------------------------------------------------------------

def _worklist(work: pd.DataFrame) -> None:
    f1, f2, f3 = st.columns([1.4, 1.4, 1.2])
    grades = f1.multiselect("등급", ["High", "Medium"], default=["High"],
                            format_func=lambda g: C.GRADE_KR.get(g, g))
    stats = f2.multiselect("처리상태", STATUS, default=["미착수", "검토 중"])
    min_stores = f3.number_input("최소 가맹점 수", min_value=0, value=0, step=10)

    view = work.copy()
    if grades:
        view = view[view["risk_grade"].isin(grades)]
    if stats:
        view = view[view["처리상태"].isin(stats)]
    if min_stores > 0:
        view = view[pd.to_numeric(view["n_stores"], errors="coerce").fillna(0)
                    >= min_stores]
    view = view.assign(
        _pri=(pd.to_numeric(view["pd_1y"], errors="coerce").fillna(0)
              * pd.to_numeric(view["n_stores"], errors="coerce").fillna(0))
    ).sort_values("_pri", ascending=False)

    st.caption(f"조건에 맞는 **{len(view):,}건** · 위험도×규모 순으로 상위 20건을 펼쳐 둡니다.")
    if view.empty:
        st.success("조건에 해당하는 미처리 건이 없습니다.")
        return

    state = _state()
    for _, r in view.head(20).iterrows():
        bid = str(r["brand_id"])
        cur = state.get(bid, {})
        with st.container(border=True):
            a, b = st.columns([3, 1.5])
            with a:
                st.markdown(
                    f"<div style='display:flex;gap:11px;align-items:center'>"
                    f"{C.brand_mark_html(str(r['brand_name']), 40)}"
                    f"<div><div style='font-weight:700;font-size:1.14rem;color:{theme.INK}'>"
                    f"{r['brand_name']} {theme.grade_chip(str(r['risk_grade']))} "
                    f"{theme.chip(cur.get('status', '미착수'), STATUS_KIND[cur.get('status', '미착수')])}"
                    f"</div>"
                    f"<div style='font-size:.9rem;color:{theme.TEXT_SUB}'>"
                    f"{r.get('industry_mid', '-')} · 가맹점 {int(r['n_stores']):,}개 · "
                    f"1년 내 악화 가능성 {float(r['pd_1y']) * 100:.1f}%</div></div></div>",
                    unsafe_allow_html=True)
                detail = str(r.get("headline_detail") or "")
                if detail and detail != "nan":
                    st.markdown(
                        f"<div style='font-size:.97rem;color:{theme.TEXT};margin-top:9px;"
                        f"line-height:1.6'>{detail}</div>", unsafe_allow_html=True)
            with b:
                owner = st.text_input("담당자", value=cur.get("owner", ""),
                                      key=f"own_{bid}", placeholder="이름 입력")
                status = st.selectbox("처리상태", STATUS,
                                      index=STATUS.index(cur.get("status", "미착수")),
                                      key=f"st_{bid}")
            note = st.text_input("확인 결과 메모", value=cur.get("note", ""),
                                 key=f"nt_{bid}",
                                 placeholder="예: 본부 재무자료 징구 완료, 자본잠식 아님")
            if (status != cur.get("status", "미착수") or owner != cur.get("owner", "")
                    or note != cur.get("note", "")):
                state[bid] = {"status": status, "owner": owner, "note": note,
                              "brand_name": str(r["brand_name"]),
                              "updated": datetime.now(UTC)
                                                 .astimezone().strftime("%Y-%m-%d %H:%M")}


def _fulltable(work: pd.DataFrame, yr) -> None:
    state = _state()
    cols = [c for c in ("brand_name", "industry_major", "industry_mid", "n_stores",
                        "pd_1y", "risk_grade", "watch_score", "n_risk", "n_high",
                        "categories", "처리상태", "담당", "headline_detail")
            if c in work.columns]
    view = work[cols].copy()
    view["risk_grade"] = view["risk_grade"].map(C.GRADE_KR).fillna(view["risk_grade"])
    view = view.sort_values("pd_1y", ascending=False)
    # ⚠️ pd_1y 는 0~1 비율이다. "%.1f%%" 서식은 값을 그대로 찍으므로 45.3%가
    #    "0.5%" 로 나온다(구버전 화면의 실제 결함). 표시 직전에 100을 곱한다.
    view["pd_1y"] = pd.to_numeric(view["pd_1y"], errors="coerce") * 100
    st.dataframe(
        view, hide_index=True,
        use_container_width=True, height=460,
        column_config={
            "brand_name": st.column_config.TextColumn("브랜드", width="medium"),
            "industry_major": st.column_config.TextColumn("업종"),
            "industry_mid": st.column_config.TextColumn("세부 업종"),
            "n_stores": st.column_config.NumberColumn("가맹점", format="%d"),
            "pd_1y": st.column_config.NumberColumn("악화 가능성", format="%.1f%%"),
            "risk_grade": st.column_config.TextColumn("등급"),
            "watch_score": st.column_config.ProgressColumn(
                "감시 우선순위", format="%.0f", min_value=0, max_value=100),
            "n_risk": st.column_config.NumberColumn("소견", format="%d"),
            "n_high": st.column_config.NumberColumn("중대", format="%d"),
            "categories": st.column_config.TextColumn("위험 영역"),
            "headline_detail": st.column_config.TextColumn("대표 소견", width="large"),
        })

    st.caption("처리 상태는 이 브라우저 세션에만 남습니다. 기록을 보존하려면 "
               "아래에서 내려받아 여신 파일에 첨부하십시오.")
    c1, c2 = st.columns(2)
    c1.download_button(
        "점검 목록 내려받기 (Excel)", _excel(view),
        file_name=f"franscore_점검큐_{yr}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True)
    log = pd.DataFrame([{"brand_id": k, **v} for k, v in state.items()])
    c2.download_button(
        f"처리 기록 내려받기 ({len(log)}건)",
        (log if not log.empty else pd.DataFrame(
            columns=["brand_id", "brand_name", "status", "owner", "note", "updated"])
         ).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"franscore_처리기록_{yr}.csv", mime="text/csv",
        disabled=log.empty, use_container_width=True)


def _excel(df: pd.DataFrame) -> bytes:
    """엑셀로 반출. openpyxl 이 없으면 CSV 바이트로 물러선다."""
    buf = io.BytesIO()
    try:
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="점검큐")
        return buf.getvalue()
    except Exception:
        return df.to_csv(index=False).encode("utf-8-sig")
