"""LLM 층 실동작 검증 — 모의 Anthropic 클라이언트로 **실제 LLM 코드 경로**를 통과시킨다.

왜 필요한가 (정직성):
    개발 환경에 ANTHROPIC_API_KEY가 없으면 news_llm/memo_llm은 규칙기반 폴백으로만
    동작한다. 폴백만 테스트하면 "LLM을 붙였다"는 주장이 검증되지 않는다(허수아비).
    본 테스트는 anthropic.Anthropic 을 모의 객체로 치환해 **키 없이도 LLM 분기 전체**
    (요청 구성 → structured output 파싱 → 검증 → refusal 처리 → 예외 폴백)를 실행·검증한다.

실행: python -m tests.test_llm_paths      (종료코드 0 = 전부 통과)
"""
from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import load_config  # noqa: E402

PASS, FAIL = "PASS", "FAIL"


# ---------------------------------------------------------------------------
# 모의 Anthropic SDK
# ---------------------------------------------------------------------------

class _Block:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _Resp:
    def __init__(self, text: str, stop_reason: str = "end_turn"):
        self.content = [_Block(text)]
        self.stop_reason = stop_reason
        self.model = "mock-model"


class _Messages:
    def __init__(self, owner):
        self._owner = owner

    def create(self, **kwargs):
        self._owner.calls.append(kwargs)
        if self._owner.mode == "refusal":
            return _Resp("", stop_reason="refusal")
        if self._owner.mode == "raise":
            raise RuntimeError("모의 API 오류 (네트워크 장애 시뮬레이션)")
        if self._owner.mode == "empty":
            r = _Resp("")
            r.content = []
            return r
        return _Resp(self._owner.payload)


class MockAnthropic:
    """anthropic.Anthropic 대체. mode로 정상/거절/예외/빈응답을 시뮬레이션."""

    payload = ""
    mode = "ok"
    calls: list = []

    def __init__(self, *a, **kw):
        self.messages = _Messages(type(self))


def _install_mock(payload: str, mode: str = "ok") -> types.ModuleType:
    MockAnthropic.payload = payload
    MockAnthropic.mode = mode
    MockAnthropic.calls = []
    mod = types.ModuleType("anthropic")
    mod.Anthropic = MockAnthropic  # type: ignore[attr-defined]
    sys.modules["anthropic"] = mod
    return mod


def _tmp_cfg(tmp: Path) -> dict:
    cfg = copy.deepcopy(load_config())
    cfg["paths"] = {"raw": tmp / "raw", "processed": tmp / "processed",
                    "outputs": tmp / "outputs", "demo_outputs": tmp / "outputs" / "_smoke"}
    for p in cfg["paths"].values():
        Path(p).mkdir(parents=True, exist_ok=True)
    (Path(cfg["paths"]["raw"]) / "news").mkdir(parents=True, exist_ok=True)
    return cfg


# ---------------------------------------------------------------------------
# 테스트
# ---------------------------------------------------------------------------

def test_news_llm_path(cfg: dict) -> str:
    """news_llm의 LLM 분기: structured JSON 파싱·enum 검증·refusal·예외 폴백."""
    from src import news_llm

    articles = {"테스트브랜드": [
        {"title": "테스트브랜드 가맹점주協 본사와 분쟁 격화", "link": "http://a",
         "published": "Mon, 01 Jul 2024 00:00:00 GMT", "source": "테스트일보"},
        {"title": "테스트브랜드 운영사 자금난 심화", "link": "http://b",
         "published": "Tue, 02 Jul 2024 00:00:00 GMT", "source": "테스트경제"},
    ]}
    os.environ["ANTHROPIC_API_KEY"] = "mock-key-for-test"
    try:
        # (1) 정상 응답 — LLM 경로가 실제로 사용되고 신호가 파싱되는지
        payload = json.dumps({"signals": [
            {"article_index": 0, "event_type": "본부분쟁", "date": "2024-07-01",
             "evidence_sentence": "가맹점주協 본사와 분쟁 격화", "confidence": "상"},
            {"article_index": 1, "event_type": "재무이슈", "date": "2024-07-02",
             "evidence_sentence": "운영사 자금난 심화", "confidence": "중"},
        ]}, ensure_ascii=False)
        _install_mock(payload, "ok")
        sigs = news_llm.extract_signals(articles, cfg)
        assert sigs, "LLM 경로에서 신호가 반환되지 않음"
        assert MockAnthropic.calls, "client.messages.create 가 호출되지 않음 (LLM 경로 미실행)"
        call = MockAnthropic.calls[0]
        assert call.get("model") == cfg["llm"]["model"], f"모델 미전달: {call.get('model')}"
        assert "temperature" not in call, "Opus 5는 temperature 미지원 — 전달되면 400 오류"
        assert "thinking" not in call, "thinking 파라미터가 전달됨 (미지원)"
        assert any(s.get("llm_used") for s in sigs), "llm_used 플래그가 True로 표기되지 않음"
        types_found = {s.get("event_type") for s in sigs}
        assert "본부분쟁" in types_found, f"사건유형 파싱 실패: {types_found}"
        print(f"    (1) 정상 응답: 신호 {len(sigs)}건, 유형 {sorted(types_found)}, llm_used=True")

        # (2) refusal → 폴백으로 안전 전환
        _install_mock("", "refusal")
        sigs_r = news_llm.extract_signals(articles, cfg)
        assert all(not s.get("llm_used") for s in sigs_r), "refusal인데 llm_used=True로 표기됨"
        print("    (2) refusal 처리: 규칙 폴백 전환 확인 (llm_used=False)")

        # (3) API 예외 → 폴백
        _install_mock("", "raise")
        sigs_e = news_llm.extract_signals(articles, cfg)
        assert sigs_e, "예외 발생 시 폴백이 신호를 반환하지 않음"
        assert all(not s.get("llm_used") for s in sigs_e), "예외인데 llm_used=True"
        print("    (3) API 예외 처리: 폴백 전환 확인")

        # (4) 잘못된 JSON → 폴백 (파서 견고성)
        _install_mock("이것은 JSON이 아님", "ok")
        sigs_b = news_llm.extract_signals(articles, cfg)
        assert sigs_b, "JSON 파싱 실패 시 폴백이 동작하지 않음"
        print("    (4) 비JSON 응답 처리: 폴백 전환 확인")
    finally:
        os.environ.pop("ANTHROPIC_API_KEY", None)
    return PASS


def test_memo_llm_path(cfg: dict) -> str:
    """memo_llm의 LLM 분기: 프롬프트 구성·필수 고지 강제·거절/예외 폴백."""
    from src import memo_llm

    context = {
        "brand_name": "테스트브랜드", "grade": "High", "prob": 0.31,
        "shap_top": [{"feature": "f_chg_contract_end_rate", "feature_kr": "계약종료율",
                      "shap_value": 0.05, "feature_value": 0.2}],
        "panel_metrics": {"가맹점 수": 57, "평균매출(천원)": 115552},
        "news": [{"event_type": "본부분쟁", "evidence_sentence": "분쟁 격화",
                  "source_url": "http://a", "published": "2024-07-01"}],
        "portfolio": {"exposure_ekw": 120.5},
    }
    os.environ["ANTHROPIC_API_KEY"] = "mock-key-for-test"
    try:
        # (1) 정상 — LLM 텍스트가 사용되고 고지문이 보장되는지
        _install_mock("## 심사메모\n브랜드 위험도가 높습니다.", "ok")
        memo = memo_llm.generate_memo(context, cfg)
        assert MockAnthropic.calls, "memo_llm이 LLM을 호출하지 않음"
        call = MockAnthropic.calls[0]
        assert "system" in call and call["system"], "시스템 프롬프트 미전달(근거 제한 불가)"
        assert "temperature" not in call, "Opus 5는 temperature 미지원"
        assert memo_llm.DISCLAIMER in memo, "필수 고지문이 메모에 없음"
        assert "브랜드 위험도가 높습니다" in memo, "LLM 응답 텍스트가 반영되지 않음"
        assert "LLM 생성" in memo, "LLM 사용 각주가 없음"
        # 프롬프트 인젝션 방어 문구가 시스템 프롬프트에 있는지
        assert "지시문" in call["system"] or "명령" in call["system"], \
            "뉴스 제목 프롬프트 인젝션 방어 지시가 시스템 프롬프트에 없음"
        print(f"    (1) 정상 응답: LLM 텍스트 반영 + 고지문 강제 + 인젝션 방어 지시 확인 ({len(memo)}자)")

        # (2) 고지문 누락 응답 → 강제 부착
        _install_mock("고지문 없는 응답", "ok")
        memo2 = memo_llm.generate_memo(context, cfg)
        assert memo_llm.DISCLAIMER in memo2, "고지문 누락 시 강제 부착이 동작하지 않음"
        print("    (2) 고지문 누락 응답: 안전장치로 강제 부착 확인")

        # (3) refusal → 결정적 템플릿 폴백
        _install_mock("", "refusal")
        memo3 = memo_llm.generate_memo(context, cfg)
        assert "LLM 미사용 폴백" in memo3, "refusal 시 폴백 템플릿이 사용되지 않음"
        print("    (3) refusal 처리: 결정적 템플릿 폴백 확인")

        # (4) 폴백 결정성 — 같은 입력 → 같은 출력
        a = memo_llm.generate_memo(context, cfg, force_fallback=True)
        b = memo_llm.generate_memo(context, cfg, force_fallback=True)
        assert a == b, "폴백 메모가 결정적이지 않음 (재현성 위반)"
        print("    (4) 폴백 결정성: 동일 입력 → 동일 출력 확인")
    finally:
        os.environ.pop("ANTHROPIC_API_KEY", None)
    return PASS


def test_rag_retrieval(cfg: dict) -> str:
    """RAG: 코퍼스 구축 → 색인 → 검색이 실제로 관련 문서를 회수하는지."""
    import pandas as pd

    from src import rag

    # 최소 코퍼스 (뉴스 스냅샷 형식)
    news_dir = Path(cfg["paths"]["raw"]) / "news"
    (news_dir / "테스트브랜드.json").write_text(json.dumps({
        "brand": "테스트브랜드",
        "articles": [
            {"title": "테스트브랜드 가맹점 계약종료 급증", "link": "http://x",
             "published": "2024-07-01", "source": "테스트일보"},
            {"title": "테스트브랜드 본사 자금난", "link": "http://y",
             "published": "2024-07-02", "source": "테스트경제"},
        ]}, ensure_ascii=False), encoding="utf-8")
    # 최소 패널 (공시 문서 원천)
    pd.DataFrame([{
        "brand_id": "B1", "brand_name": "테스트브랜드", "industry_major": "외식",
        "industry_mid": "한식", "year": 2023, "n_stores": 57.0, "n_direct": 1.0,
        "avg_sales": 115552.0, "n_contract_end": 5.0, "n_contract_cancel": 2.0,
        "n_new": 3.0, "n_regions": 8.0, "region_hhi": 0.21, "cancel_flag": 0,
        "cancel_type": None, "eligible_t": True,
    }]).to_parquet(Path(cfg["paths"]["processed"]) / "panel.parquet", index=False)

    corpus = rag.build_corpus(cfg)
    assert not corpus.empty, "RAG 코퍼스가 비어 있음"
    assert set(corpus["source_type"]) >= {"news", "disclosure"}, \
        f"코퍼스에 두 문서 유형이 모두 없음: {set(corpus['source_type'])}"
    index = rag.build_index(cfg, corpus)

    hits = index.retrieve("테스트브랜드 계약종료 급증", k=3)
    assert not hits.empty, "검색 결과가 비어 있음"
    assert any("계약종료" in str(t) for t in hits["text"]), \
        f"관련 문서를 회수하지 못함: {hits['text'].tolist()}"

    ev = rag.retrieve_evidence(cfg, "테스트브랜드", ["계약종료율"], index=index)
    assert ev, "retrieve_evidence 결과 없음"
    assert all({"doc_id", "url", "published", "score"} <= set(e) for e in ev), \
        "근거 항목에 출처 추적 필드가 없음"
    print(f"    코퍼스 {len(corpus)}문서 (뉴스+공시), 검색 상위 {len(hits)}건, "
          f"근거 {len(ev)}건 (출처·발행일·유사도 포함)")

    # 개체 매칭 필터: 무관 기사가 코퍼스에 들어오지 않는지
    (news_dir / "속속브랜드.json").write_text(json.dumps({
        "brand": "속속브랜드",
        "articles": [{"title": "세계 곳곳서 사찰 속속 재건", "link": "http://z",
                      "published": "2023-06-29", "source": "무관일보"}]},
        ensure_ascii=False), encoding="utf-8")
    corpus2 = rag.build_corpus(cfg)
    bad = corpus2[(corpus2["source_type"] == "news") & (corpus2["text"].str.contains("사찰"))]
    assert bad.empty, "개체 매칭 필터가 무관 기사를 걸러내지 못함"
    print("    개체 매칭 필터: 무관 기사 코퍼스 유입 차단 확인")
    return PASS


def main() -> int:
    print("=== FranSCORE LLM/RAG 경로 검증 (모의 API — 키 불필요) ===")
    tests = [("news_llm_path", test_news_llm_path),
             ("memo_llm_path", test_memo_llm_path),
             ("rag_retrieval", test_rag_retrieval)]
    results = []
    real_anthropic = sys.modules.get("anthropic")
    with tempfile.TemporaryDirectory(prefix="franscore_llm_") as td:
        for i, (name, fn) in enumerate(tests, 1):
            cfg = _tmp_cfg(Path(td) / name)
            print(f"[{i}/{len(tests)}] {name} ...")
            try:
                results.append((name, fn(cfg)))
                print(f"[{i}/{len(tests)}] {name} ... {PASS}")
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                results.append((name, f"{FAIL} ({exc})"))
                print(f"[{i}/{len(tests)}] {name} ... {FAIL}")
    if real_anthropic is not None:
        sys.modules["anthropic"] = real_anthropic
    else:
        sys.modules.pop("anthropic", None)

    ok = sum(1 for _, r in results if r == PASS)
    print(f"\n=== SUMMARY: {ok}/{len(results)} passed ===")
    for n, r in results:
        print(f"  {r if r == PASS else 'FAIL':5s} {n}" if r == PASS else f"  FAIL  {n}: {r}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
