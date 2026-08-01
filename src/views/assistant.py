"""AI 상담 — 자연어로 묻고 근거와 함께 답을 받는다."""
from __future__ import annotations

import streamlit as st

from src import chat, theme
from src.views import common as C

_HISTORY = "chat_history"
EXAMPLES = [
    "인생냉면 창업을 고민 중인데 전반적으로 분석해줘",
    "달콤왕가탕후루 가맹점 수 추이를 가져와줘",
    "메가커피와 컴포즈커피 중 어디가 더 안정적이야?",
    "치킨 업종에서 지금 가장 위험한 브랜드는?",
]


def render() -> None:
    theme.page_header(
        "AI 상담",
        "프랜차이즈에 대해 자유롭게 물어보십시오. 공정거래위원회 공시·금융감독원 "
        "감사보고서·뉴스에서 근거를 찾아 답합니다.",
        eyebrow="상담")

    if _HISTORY not in st.session_state:
        st.session_state[_HISTORY] = []
    history = st.session_state[_HISTORY]

    if not history:
        st.markdown("##### 이렇게 물어보실 수 있습니다")
        cols = st.columns(2)
        for i, ex in enumerate(EXAMPLES):
            if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
                _ask(ex)
                st.rerun()
        st.write("")

    # ⚠️ chat_message 의 avatar 는 이모지·이미지 경로·URL 만 받는다. '◆'(U+25C6) 같은
    #    기호 문자를 넣으면 StreamlitAPIException 으로 화면 전체가 죽는다(실측).
    for turn in history:
        with st.chat_message("user" if turn["role"] == "user" else "assistant",
                             avatar="🙋" if turn["role"] == "user" else "🟡"):
            st.markdown(turn["content"])
            if turn.get("evidence"):
                _evidence_block(turn["evidence"], turn.get("idx", 0))
            if turn["role"] == "assistant" and not turn.get("llm_used", True):
                st.caption("답변 생성 모델이 연결되지 않아 수집된 사실만 정리했습니다.")

    q = st.chat_input("무엇이든 물어보십시오")
    if q:
        _ask(q)
        st.rerun()

    if history:
        c1, c2 = st.columns([1, 5])
        if c1.button("대화 지우기"):
            st.session_state[_HISTORY] = []
            st.rerun()
        c2.download_button(
            "대화 내려받기 (.md)",
            "\n\n".join(f"**{'질문' if t['role'] == 'user' else '답변'}**\n\n{t['content']}"
                        for t in history).encode("utf-8-sig"),
            file_name="franscore_상담.md", mime="text/markdown")


def _ask(question: str) -> None:
    history = st.session_state[_HISTORY]
    history.append({"role": "user", "content": question})
    with st.spinner("공시·재무·뉴스에서 근거를 찾는 중…"):
        try:
            res = chat.answer(C.cfg(), question, history[:-1])
        except Exception as exc:                      # 화면이 죽으면 안 된다
            history.append({"role": "assistant",
                            "content": f"답변 중 문제가 발생했습니다: {exc}",
                            "llm_used": False})
            return
    history.append({"role": "assistant", "content": res["text"],
                    "evidence": res.get("evidence") or [],
                    "llm_used": bool(res.get("llm_used")),
                    "idx": len(history)})


def _evidence_block(evidence: list[dict], idx: int) -> None:
    if not evidence:
        return
    with st.expander(f"근거로 쓴 자료 {len(evidence)}건"):
        for i, e in enumerate(evidence):
            kind = str(e.get("출처유형", ""))
            chip = theme.chip(kind or "문서",
                              "Info" if kind == "실시간뉴스" else "Neutral")
            url = str(e.get("url") or "")
            head = f"{chip} <b>{e.get('출처', '')}</b>"
            if e.get("발행"):
                head += (f"<span style='color:{theme.TEXT_MUTED};font-size:.88rem'>"
                         f" · {e['발행']}</span>")
            st.markdown(head, unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-size:.95rem;color:{theme.TEXT_SUB};line-height:1.55;"
                f"margin:2px 0 10px 0'>{str(e.get('내용', ''))[:400]}…</div>",
                unsafe_allow_html=True)
            if url and url.startswith("http"):
                st.markdown(f"[원문 보기]({url})")
            del i
    del idx
