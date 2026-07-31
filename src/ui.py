"""대시보드 화면 — 현장에서 바로 읽히는 형태로 구성한다.

설계 원칙 (실사용 피드백 반영):
  1. **용어를 풀어쓴다.** PD·SHAP·HHI·Lift 같은 말을 화면 전면에 두지 않는다.
     "1년 내 악화 가능성", "이렇게 본 이유", "쏠림 정도" 처럼 뜻이 바로 통하는 말을 쓰고,
     원래 용어는 괄호나 도움말로 남겨 전문가가 확인할 수 있게 한다.
  2. **결론 → 근거 순서.** 무엇이 문제인지 먼저 말하고, 그 아래 근거를 붙인다.
     숫자를 늘어놓고 해석을 독자에게 떠넘기지 않는다.
  3. **안내문으로 채우지 않는다.** "여기를 누르세요" 대신 화면 자체가 읽히게 만든다.
  4. **모르는 것은 모른다고 쓴다.** 자격 미충족·데이터 없음을 빈칸으로 두지 않고 이유를 밝힌다.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.brand_search import search

# 등급 표기 — 영문 등급명을 그대로 노출하지 않는다
GRADE_KR = {"High": "주의", "Medium": "관찰", "Low": "양호"}
GRADE_COLOR = {"High": "#d9534f", "Medium": "#f0ad4e", "Low": "#5cb85c"}
GRADE_ACTION = {
    "High": "신규 취급 시 최근 공시·뉴스를 함께 확인하고, 이 브랜드에 나간 여신 총액이 한도 안인지 점검하십시오.",
    "Medium": "분기 단위로 점포 수와 계약종료율의 방향을 확인하십시오.",
    "Low": "정기 모니터링을 유지하십시오.",
}

WHAT_IS_DETERIORATION = (
    "**'악화'란** 이듬해에 ① 가맹점 수가 업종 하위 20% 수준으로 순감하거나 "
    "② 실질 평균매출이 업종 하위 20%로 떨어지거나 ③ 계약종료율이 업종 상위 20%로 뛰는 것 중 "
    "**두 가지 이상**이 동시에 일어나는 것을 말합니다(한 가지라도 극단값이면 포함). "
    "부도가 아니라 **브랜드가 구조적으로 꺾이는 시점**을 뜻합니다."
)


def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def load_scores(cfg: dict) -> tuple[pd.DataFrame | None, dict, str]:
    """점수표 로드. 전 업종(확장 트랙)이 있으면 그것을 쓴다 — 커버리지가 넓을수록 유용하다."""
    out = Path(cfg["paths"]["outputs"])
    for path, label in ((out / "extended" / "scores_latest.csv", "전 업종"),
                        (out / "scores_latest.csv", "외식업")):
        if path.exists():
            df = pd.read_csv(path)
            meta = _read_json(path.parent / "scores_latest_meta.json") or {}
            return df, meta, label
    return None, {}, ""


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def grade_badge(g: str) -> str:
    return f":{'red' if g == 'High' else 'orange' if g == 'Medium' else 'green'}[**{GRADE_KR.get(g, g)}**]"


# ---------------------------------------------------------------------------
# 화면 ⓪ 한눈에 보기
# ---------------------------------------------------------------------------

def render_overview(cfg: dict) -> None:
    df, meta, scope = load_scores(cfg)
    if df is None:
        st.info("점수가 아직 없습니다. `python run_pipeline.py --step score` 를 실행하세요.")
        return
    yr = meta.get("scored_year", "-")

    st.title("프랜차이즈 브랜드 리스크 현황")
    st.caption(
        f"{yr}년 공시 기준 · {scope} {len(df):,}개 브랜드를 평가했습니다. "
        "각 브랜드가 **내년에 구조적으로 꺾일 가능성**을 공시 데이터로 계산한 결과입니다."
    )

    high = df[df["risk_grade"] == "High"]
    med = df[df["risk_grade"] == "Medium"]
    big_high = high[pd.to_numeric(high["n_stores"], errors="coerce").fillna(0) >= 100]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("평가 브랜드", f"{len(df):,}개")
    c2.metric("주의 등급", f"{len(high):,}개", help="상위 10% 위험도. 우선 점검 대상입니다.")
    c3.metric("관찰 등급", f"{len(med):,}개")
    c4.metric("주의 + 100점포 이상", f"{len(big_high):,}개",
              help="규모가 커서 부실 시 여신 영향이 큰 브랜드입니다.")

    st.divider()

    # ── 지금 봐야 할 브랜드 ──
    st.subheader("지금 봐야 할 브랜드")
    st.caption("위험도가 높으면서 규모도 큰 순서입니다. 규모가 크면 같은 확률이라도 손실이 큽니다.")
    watch = high.copy()
    watch["_score"] = (pd.to_numeric(watch["pd_1y"], errors="coerce").fillna(0)
                       * pd.to_numeric(watch["n_stores"], errors="coerce").fillna(0))
    watch = watch.nlargest(8, "_score")
    if watch.empty:
        st.success("주의 등급에 해당하는 브랜드가 없습니다.")
    for _, r in watch.iterrows():
        with st.container(border=True):
            a, b = st.columns([3, 2])
            a.markdown(f"### {r['brand_name']}")
            a.caption(f"{r.get('industry_major', '-')} · {r.get('industry_mid', '-')} · "
                      f"가맹점 {int(r['n_stores']):,}개")
            b.metric("1년 내 악화 가능성", pct(r["pd_1y"]),
                     help="공시 지표로 계산한 확률입니다. 같은 값이 여러 브랜드에 나오는 것은 "
                          "확률 보정이 구간 단위로 이뤄지기 때문이며, 순위는 아래를 보십시오.")
            rank = r.get("pd_rank_pct")
            if pd.notna(rank):
                b.caption(f"전체 중 **상위 {(1 - float(rank)) * 100:.1f}%**")
            reasons = [str(r.get(f"factor{i}")) for i in (1, 2, 3)
                       if isinstance(r.get(f"factor{i}"), str)]
            if reasons:
                a.markdown("**이렇게 본 이유:** " + " · ".join(reasons))

    st.divider()

    # ── 업종별 분포 ──
    st.subheader("업종별 위험 분포")
    st.caption("업종마다 위험이 몰린 정도가 다릅니다. 특정 업종에 여신이 쏠려 있다면 함께 보십시오.")
    key = "industry_mid" if "industry_mid" in df.columns else "industry_major"
    tab = (df.groupby(key)["risk_grade"]
             .value_counts().unstack(fill_value=0).reindex(columns=["High", "Medium", "Low"],
                                                           fill_value=0))
    tab["합계"] = tab.sum(axis=1)
    tab["주의 비율"] = (tab["High"] / tab["합계"]).round(3)
    tab = tab[tab["합계"] >= 10].sort_values("주의 비율", ascending=False)
    tab = tab.rename(columns={"High": "주의", "Medium": "관찰", "Low": "양호"})
    st.dataframe(
        tab.head(15), width="stretch",
        column_config={"주의 비율": st.column_config.ProgressColumn(
            "주의 비율", format="%.1f%%", min_value=0.0, max_value=float(tab["주의 비율"].max() or 1))})

    with st.expander("이 숫자들이 무슨 뜻인가요"):
        st.markdown(WHAT_IS_DETERIORATION)
        st.markdown(
            "- **1년 내 악화 가능성** — 공시 지표로 계산한 확률입니다(모델 용어로는 PD). "
            "실제 부도율과는 다르며, 우선순위를 정하는 용도입니다.\n"
            "- **주의 / 관찰 / 양호** — 전체 브랜드를 위험도 순으로 줄 세워 상위 10% / 그다음 20% / "
            "나머지로 나눈 것입니다. 절대 기준이 아니라 **상대 순위**입니다.\n"
            "- **왜 여러 브랜드의 확률이 똑같이 나오나요** — 모델의 원점수를 실제 발생률에 맞게 "
            "보정할 때 구간(계단) 단위로 맞추기 때문입니다. 같은 구간에 들어간 브랜드는 같은 "
            "확률로 표시되며, 그 안의 우열은 **상위 %** 순위로 구분합니다. 확률을 소수점까지 "
            "믿기보다 '어느 구간에 있는가'로 읽는 것이 맞습니다.\n"
            "- **이렇게 본 이유** — 모델이 그 브랜드를 위험하다고 본 근거를 기여도 순으로 "
            "보여줍니다(모델 용어로는 SHAP)."
        )


# ---------------------------------------------------------------------------
# 화면 ① 브랜드 조회
# ---------------------------------------------------------------------------

def render_lookup(cfg: dict) -> None:
    df, _meta, scope = load_scores(cfg)
    panel_path = Path(cfg["paths"]["processed"]) / "panel_full.parquet"
    st.title("브랜드 조회")
    st.caption("브랜드 이름으로 찾습니다. 통칭으로 검색해도 공시 등록명을 찾아냅니다 "
               "(예: '메가커피' → '메가엠지씨커피(MEGA MGC COFFEE)').")

    q = st.text_input("브랜드 이름", placeholder="메가커피, 교촌, 배스킨라빈스 …")
    cats = ["전체"] + (sorted(df["industry_major"].dropna().unique().tolist())
                      if df is not None else [])
    cat = st.selectbox("업종", cats)

    if df is None:
        st.info("점수가 없습니다. `--step score` 를 먼저 실행하세요.")
        return

    view = df if cat == "전체" else df[df["industry_major"] == cat]
    if not q:
        st.caption(f"{scope} {len(view):,}개 브랜드가 평가되어 있습니다. 이름을 입력하면 조회합니다.")
        return

    hit, sugg = search(view, q)
    if hit.empty:
        st.warning(f"'{q}' 로 평가된 브랜드를 찾지 못했습니다.")
        if sugg:
            st.caption("혹시 이것을 찾으셨나요? " + " · ".join(sugg))
        # 평가 대상이 아닌 것인지, 데이터 자체가 없는 것인지 구분해 알려준다
        if panel_path.exists():
            allb = pd.read_parquet(panel_path)[["brand_name", "industry_major", "year", "n_stores"]]
            h2, _ = search(allb.drop_duplicates("brand_name"), q)
            if not h2.empty:
                st.info(
                    "**공시 데이터에는 있지만 평가 대상이 아닙니다.** 이 도구는 "
                    "'최근 3년 연속 관측 + 일정 규모 이상' 조건을 만족하는 브랜드만 점수화합니다 "
                    "(추세를 계산할 이력이 없으면 예측이 성립하지 않기 때문입니다).\n\n"
                    "공시 기록: " + ", ".join(h2["brand_name"].astype(str).head(5)))
            else:
                st.info(
                    "**공시 데이터 자체에 없습니다.** 이 도구는 공정거래위원회 "
                    "가맹사업 정보공개서를 원천으로 씁니다. 직영으로만 운영하는 브랜드"
                    "(예: 스타벅스코리아)는 가맹사업이 아니어서 공시 대상이 아닙니다.")
        return

    st.success(f"{len(hit)}건 찾았습니다.")
    for _, r in hit.head(10).iterrows():
        _brand_card(r)


def _brand_card(r: pd.Series) -> None:
    with st.container(border=True):
        st.markdown(f"## {r['brand_name']}  {grade_badge(str(r['risk_grade']))}")
        c1, c2, c3 = st.columns(3)
        c1.metric("1년 내 악화 가능성", pct(r["pd_1y"]))
        rank = r.get("pd_rank_pct")
        if pd.notna(rank):
            c1.caption(f"전체 중 상위 {(1 - float(rank)) * 100:.1f}%")
        c2.metric("가맹점 수", f"{int(r['n_stores']):,}개")
        c3.metric("업종", str(r.get("industry_mid") or r.get("industry_major") or "-"))
        reasons = [(str(r.get(f"factor{i}")), r.get(f"factor{i}_shap"))
                   for i in (1, 2, 3) if isinstance(r.get(f"factor{i}"), str)]
        if reasons:
            st.markdown("**이렇게 본 이유** (기여가 큰 순서)")
            for name, val in reasons:
                direction = "위험을 높이는 쪽" if (val or 0) > 0 else "위험을 낮추는 쪽"
                st.markdown(f"- {name} — {direction}")
        st.caption(GRADE_ACTION.get(str(r["risk_grade"]), ""))
