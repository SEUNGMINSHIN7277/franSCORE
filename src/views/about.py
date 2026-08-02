"""서비스 소개 — 처음 보는 사람도, 심사역도 같은 화면에서 이해하게.

두 독자를 한 화면에 담는 방법
    윗부분은 **전제 지식 없이** 읽히게 쓴다 — 무엇을, 왜, 어떻게 쓰는지.
    아랫부분은 접어 둔 상세로 **검증 수치와 한계**를 둔다. 심사역은 펼쳐 보고,
    처음 보는 사람은 안 펼쳐도 흐름이 끊기지 않는다.
    두 층을 섞으면 한쪽은 어렵고 한쪽은 부실해진다.
"""
from __future__ import annotations

import json

import streamlit as st

from src import theme
from src.views import common as C

_STEPS = (
    ("1", "공시를 모은다",
     "공정거래위원회에 등록된 프랜차이즈 브랜드의 가맹점 수·개점·계약종료·평균매출·"
     "면적당매출을 연도별로 모읍니다. 여기에 금융감독원 전자공시의 가맹본부 재무와 "
     "네이버 검색수요·뉴스를 더합니다."),
    ("2", "무엇이 '나빠진 것'인지 정의한다",
     "정의를 우리가 만들지 않습니다. 공시 지표가 **업종 안에서 하위 구간에 들어가는 것**을 "
     "'악화'로 봅니다. 어떤 지표를 어떤 기준으로 보는지 전부 공개하고, 규칙은 "
     "모델을 학습하기 전에 고정해 둡니다."),
    ("3", "다음 해에 나빠질 브랜드를 가려낸다",
     "과거 자료로 학습한 모델이 각 브랜드의 확률을 매깁니다. 그 확률을 **고정된 구간**으로 "
     "나눠 세 등급을 줍니다. 구간은 순위가 아니라 확률 기준이라, 업계 전체가 나빠지면 "
     "낮은 등급이 늘어납니다."),
    ("4", "왜 그런지 문장으로 설명한다",
     "숫자만 주면 쓸 수 없습니다. 34가지 점검 규칙이 그 브랜드의 실제 수치로 소견을 씁니다 — "
     "'계약을 끝내는 가맹점이 많습니다(82개, 58.6%)' 처럼요."),
)


def render() -> None:
    theme.page_header(
        "FranSCORE 는 무엇인가",
        "프랜차이즈 **브랜드**의 사업 안정성을 평가해, 은행이 가맹점 여신을 볼 때 "
        "참고할 수 있게 만든 도구입니다.",
        eyebrow="서비스 소개")

    # ── 한 문단 요약 ──────────────────────────────────────────────
    st.markdown(
        f"<div style='padding:16px 18px;border-radius:{theme.RADIUS_LG};background:{theme.YELLOW_SOFT};"
        f"border:1px solid #F2E3A8;font-size:{theme.FS_LG};line-height:1.75;color:{theme.TEXT}'>"
        f"은행이 가맹점주에게 대출할 때 보는 것은 <b>그 사람의 상환능력</b>입니다. "
        f"그런데 손실은 개인이 아니라 <b>브랜드가 꺾일 때 그 브랜드 가맹점이 한꺼번에</b> "
        f"어려워지는 형태로 옵니다. 차주 심사로는 이 축이 보이지 않습니다. "
        f"FranSCORE 는 그 빠진 축 하나를 채웁니다.</div>",
        unsafe_allow_html=True)

    st.write("")
    st.markdown("### 왜 브랜드를 따로 보아야 하는가")
    imp = _corr_impact()
    c1, c2, c3 = st.columns(3)
    c1.metric("같은 브랜드 안의 상관", _fmt(imp.get("rho_within_brand"), 3),
              help="한 브랜드의 여러 지역 점포가 함께 움직이는 정도. 거시·업종 효과를 "
                   "걷어낸 뒤에도 이만큼 남습니다.")
    c2.metric("브랜드 사이의 상관", _fmt(imp.get("rho_between_brand"), 3),
              help="브랜드끼리는 거의 함께 움직이지 않습니다.")
    c3.metric("독립 가정 시 과소추정", _fmt(imp.get("ul99_multiple"), 2, "배"),
              help="차주를 서로 무관하다고 보면 자본 소요분을 이만큼 낮게 잡습니다.")
    st.caption("위험은 경기가 아니라 **브랜드**에 있습니다. 그래서 처방도 '경기 대응'이 "
               "아니라 '브랜드 단위 관리'가 맞습니다.")

    st.write("")
    st.markdown("### 어떻게 평가하는가")
    for no, title, body in _STEPS:
        with st.container(border=True):
            st.markdown(
                f"<div style='display:flex;gap:14px;align-items:flex-start'>"
                f"<div style='flex:0 0 34px;height:34px;border-radius:{theme.RADIUS_MD};"
                f"background:{theme.YELLOW};color:#26221E;font-weight:800;"
                f"display:flex;align-items:center;justify-content:center'>{no}</div>"
                f"<div><div style='font-weight:700;font-size:{theme.FS_LG};color:{theme.INK};"
                f"margin-bottom:3px'>{title}</div>"
                f"<div style='color:{theme.TEXT};line-height:1.7'>{body}</div></div></div>",
                unsafe_allow_html=True)

    # ── 등급 읽는 법 ──────────────────────────────────────────────
    st.write("")
    st.markdown("### 등급을 읽는 법")
    bands = C.grade_bands(C._mtime(C.out_dir() / "grade_bands.json"))
    if bands.get("pooled"):
        st.markdown(C.grade_legend_html({}), unsafe_allow_html=True)
    st.info("**등급 옆의 '상태'를 함께 보십시오.** 같은 FS3 라도 '건전'이면 아직 "
            "악화 신호가 없는데 모델이 위험을 예측한 것이고, '요주의'면 이미 공시에 "
            "악화가 나타난 브랜드입니다. 뒤쪽은 모델의 성능 근거가 없으니 확률값보다 "
            "**진단 소견**을 보고 판단하셔야 합니다.")

    # ── 화면 안내 ────────────────────────────────────────────────
    st.write("")
    st.markdown("### 화면 안내")
    guide = {
        "FRANSCORE": "전체 현황과 브랜드별 상세. 어떤 브랜드든 눌러 들어가면 진단 소견·"
                     "부문별 점검·공시 추이·본부 재무·검색 수요를 볼 수 있습니다.",
        "점검 큐": "확인해야 할 브랜드를 담당자에게 배정하고 확인 결과를 기록합니다. "
                   "엑셀로 내려받아 결재에 붙일 수 있습니다.",
        "여신 포트폴리오": "브랜드별 여신 쏠림과 예상 손실을 봅니다. 신규 여신을 넣어 "
                          "보면 집중도가 어떻게 변하는지 즉시 계산됩니다.",
        "AI 상담": "자연어로 물어보면 수집된 자료에서 근거를 찾아 답합니다.",
    }
    for k, v in guide.items():
        st.markdown(
            f"<div style='display:flex;gap:12px;padding:8px 0;border-bottom:1px solid "
            f"{theme.BORDER}'><b style='min-width:104px;color:{theme.INK}'>{k}</b>"
            f"<span style='color:{theme.TEXT_SUB};line-height:1.6'>{v}</span></div>",
            unsafe_allow_html=True)

    # ── 심사역용 상세 ────────────────────────────────────────────
    st.write("")
    with st.expander("검증 결과 — 이 평가를 믿어도 되는가"):
        _validation_block()
    with st.expander("한계 — 이 도구가 못 하는 것"):
        st.markdown(
            "1. **부도확률이 아닙니다.** 우리가 세는 것은 공시상 점포 감소·계약종료·"
            "매출 하락이지 차주의 채무불이행이 아닙니다.\n"
            "2. **이미 악화한 브랜드에는 성능 근거가 없습니다.** 모델은 '아직 멀쩡한 "
            "브랜드가 나빠질 확률'로 학습했기 때문입니다.\n"
            "3. **작은 브랜드는 보지 못합니다.** 누적 최대 가맹점 30개 이상·3년 연속 공시가 "
            "조건입니다. 브랜드 수로는 외식 9,333개 중 1,442개(15.5%)뿐이지만, "
            "**가맹점 수로 보면 외식 가맹점 4곳 중 3곳(73.6%)**이 평가 대상 안에 "
            "있습니다 — 못 보는 브랜드가 대부분 아주 작기 때문입니다.\n"
            "4. **가맹본부 재무는 대부분 확인되지 않습니다.** 외부감사 대상이 아닌 본부는 "
            "감사보고서를 내지 않습니다(브랜드 기준 11.4%만 확인).\n"
            "5. **폐업 위약금과 본부 대여금은 공시에 없습니다.** 회수율에 직접 영향을 주는 "
            "두 항목을 우리는 볼 수 없습니다.")
        st.caption("전체 목록과 근거는 docs/MODEL_USE_SPEC.md · docs/METHODOLOGY.md 에 있습니다.")
    with st.expander("이 도구를 쓰면 안 되는 곳"):
        st.markdown(
            "- 자동 여신 승인·거절\n- 여신 한도 산정\n- 금리 산정·차등\n"
            "- 규제자본(IRB) 산출 · IFRS 9 충당금\n"
            "- 개별 가맹점주에 대한 불이익 처분의 단독 근거\n"
            "- 산출물을 '부도확률(PD)' 이라고 부르거나 그렇게 인용하는 것")
        st.caption("2선 리스크 관리의 **점검 우선순위**와 **심사 참고 정보**가 정해진 용도입니다.")

    C.refresh_footer()


def _corr_impact() -> dict:
    """상관 실증 산출물을 읽는다.

    ⚠️ 이 세 수치는 원래 화면에 **박아 놓은 문자열**이었다. 그래서 모형이 다시 돌아
       배수가 5.31 → 7.30 으로 바뀌었을 때 문서만 갱신되고 화면은 옛 값을 계속
       보여 줬다. 같은 제출물 안에서 필요성의 핵심 수치가 두 값으로 갈린 것이다.
       고쳐야 할 것은 그 문자열이 아니라 **문자열을 박아 둔 구조**다 — 산출물에서
       읽으면 다시 어긋날 수가 없다.
    """
    p = C.out_dir() / "correlation_impact.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _fmt(v, nd: int, suffix: str = "") -> str:
    """산출물이 없으면 숫자를 지어내지 않고 '—' 를 보여 준다."""
    return "—" if v is None else f"{float(v):.{nd}f}{suffix}"


def _validation_block() -> None:
    """검증 산출물이 있으면 실측치를, 없으면 없다고 말한다."""
    p = C.out_dir() / "validation" / "summary.json"
    if not p.exists():
        st.caption("검증 실적이 아직 집계되지 않았습니다.")
        return
    try:
        s = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st.caption("검증 산출물을 읽지 못했습니다.")
        return
    d, cal = s.get("discrimination_pooled", {}), s.get("calibration", {})
    st.markdown("**판별력** — 위험한 브랜드를 위로 올리는가")
    c1, c2, c3 = st.columns(3)
    c1.metric("AUC", f"{d.get('auc', 0):.3f}",
              help=f"95% 신뢰구간 [{d.get('auc_lo', 0):.3f}, {d.get('auc_hi', 0):.3f}]. "
                   f"0.5 는 무작위와 같다는 뜻입니다.")
    c2.metric("KS", f"{d.get('ks', 0):.3f}")
    c3.metric("Lift@10%", f"{d.get('lift_at_10', 0):.2f}",
              help="상위 10%에서 실제 악화가 평균의 몇 배로 나오는가.")
    st.caption(f"표본 {d.get('n', 0):,}건 · 실제 악화 {d.get('events', 0)}건")

    st.markdown("**정확성** — 확률의 크기가 실제와 맞는가")
    hl, sz = cal.get("hosmer_lemeshow", {}), cal.get("spiegelhalter", {})
    st.markdown(
        f"- Hosmer-Lemeshow p = **{hl.get('p_value', 0):.3f}** → {hl.get('verdict', '-')}\n"
        f"- Spiegelhalter Z = **{sz.get('z', 0):+.2f}** → {sz.get('verdict', '-')}\n"
        f"- 예측 평균 {cal.get('mean_predicted', 0) * 100:.2f}% vs "
        f"실제 {cal.get('observed', 0) * 100:.2f}%")

    tp = s.get("transition_pairs")
    if tp:
        st.markdown("**안정성** — 등급이 함부로 흔들리지 않는가")
        st.markdown(f"- 등급 이행행렬 {tp}개 코호트 · 표본 이탈률 "
                    f"{s.get('mean_exit_rate', 0) * 100:.1f}%\n"
                    f"- 점수 안정성 지표(PSI) "
                    + " · ".join(f"{r['base_year']}→{r['target_year']} {r['psi']:.4f}"
                                 for r in (s.get("psi", {}).get("score", {})
                                           .get("by_year", []))[:3]))
