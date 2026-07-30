"""M4 뉴스 신호 모듈 — Google News RSS 수집 + LLM/규칙기반 구조화 추출.

INTERFACES.md §4 (news_llm.py) 구현.

⚠️ 정직성 원칙:
- 뉴스 신호는 모델 점수에 절대 투입하지 않는다 (config `llm.score_injection: false`).
  산출물(outputs/news_signals.json)은 대시보드 화면 표시 전용이다.
- 수집 원본은 data/raw/news/{safe_brand}.json 에 스냅샷 보존 (재현성).
- ANTHROPIC_API_KEY 가 없거나 API 호출이 실패하면 규칙기반 키워드 폴백을 사용하고
  각 신호에 `llm_used: false` 를 표기한다.

실행: 프로젝트 루트에서 `python -m src.news_llm [브랜드명 ...]`
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
from email.utils import parsedate_to_datetime
from pathlib import Path

import feedparser

from src.common import get_logger, load_config, set_seed

log = get_logger("news_llm")

# 계약 §4 고정 이벤트 유형
EVENT_TYPES = ["본부분쟁", "재무이슈", "집단폐점", "기타", "무관"]

# 규칙기반 폴백 키워드 (계약 명세 그대로, 위에서부터 우선 적용)
_KEYWORD_RULES: list[tuple[str, list[str]]] = [
    ("본부분쟁", ["분쟁", "소송", "갈등"]),
    ("재무이슈", ["부도", "회생", "적자", "압류", "미지급"]),
    ("집단폐점", ["폐점", "철수"]),
]

_RSS_URL = "https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"
_SLEEP_BETWEEN_REQUESTS_SEC = 0.4  # 요청 간 예의상 대기

# LLM 구조화 추출용 JSON 스키마 (additionalProperties/required 필수)
_SIGNAL_SCHEMA = {
    "type": "object",
    "properties": {
        "signals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "article_index": {
                        "type": "integer",
                        "description": "입력 기사 목록에서의 0-기반 인덱스",
                    },
                    "event_type": {"type": "string", "enum": EVENT_TYPES},
                    "evidence_sentence": {
                        "type": "string",
                        "description": "판단 근거가 된 제목 내 문구(입력에 있는 표현만)",
                    },
                    "confidence": {"type": "string", "enum": ["상", "중", "하"]},
                },
                "required": ["article_index", "event_type", "evidence_sentence", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["signals"],
    "additionalProperties": False,
}

_EXTRACT_SYSTEM = (
    "당신은 은행 리스크관리 부서의 뉴스 분류 보조자입니다.\n"
    "규칙:\n"
    "1. 제공된 기사 제목 목록만 근거로 분류한다. 외부 지식으로 사실을 추정하지 않는다.\n"
    "2. event_type 은 다음 중 하나: 본부분쟁(가맹점-본사 분쟁·소송·갈등), "
    "재무이슈(부도·회생·적자·압류·대금 미지급 등), 집단폐점(대규모 폐점·사업 철수), "
    "기타(해당 브랜드 관련이나 위 유형 아님), 무관(해당 브랜드와 무관).\n"
    "3. evidence_sentence 는 제목에 실제로 등장하는 문구만 인용한다.\n"
    "4. confidence 는 제목만으로 판단이 명확하면 상, 개연성 수준이면 중, 불확실하면 하."
)


def _safe_name(name: str) -> str:
    """브랜드명 → 파일명 안전 문자열 (한글/영숫자/밑줄만 유지)."""
    s = re.sub(r"[^\w가-힣]+", "_", str(name)).strip("_")
    return s or "brand"


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).lower()


def _to_date(published: str) -> str:
    """RSS published 문자열 → 'YYYY-MM-DD' (실패 시 빈 문자열)."""
    try:
        return parsedate_to_datetime(published).strftime("%Y-%m-%d")
    except Exception:
        return ""


def _entity_match(brand: str, title: str) -> bool:
    """제목이 명백히 해당 브랜드 기사가 아닌 경우를 걸러내는 완화 매칭.

    가정(로그 명시): 브랜드명 전체가 제목에 포함되거나, 4자 이상 브랜드는
    앞 2자·뒤 2자가 모두 포함되면 관련 기사로 본다
    (예: '메가커피' vs 제목 '메가MGC커피'). 일반명사형 브랜드의 오탐을 줄이기
    위한 휴리스틱이며 완전한 개체 인식이 아니다.
    """
    b, t = _norm(brand), _norm(title)
    if not b or not t:
        return False
    if b in t:
        return True
    if len(b) >= 4 and b[:2] in t and b[-2:] in t:
        return True
    return False


# ---------------------------------------------------------------------------
# 1) 수집
# ---------------------------------------------------------------------------

def fetch_news(brand_names: list[str], cfg: dict) -> dict[str, list]:
    """Google News RSS로 브랜드별 기사 수집 + 원본 스냅샷 저장.

    반환: {brand_name: [{title, link, published, source}, ...]}
    - 브랜드명 + query_suffixes 조합으로 검색, 링크 기준 중복 제거,
      브랜드당 max_articles_per_brand 개 상한.
    - 네트워크/파싱 실패는 로그 후 무시(해당 쿼리 결과 빈 리스트).
    - 스냅샷: {paths.raw}/news/{safe_brand}.json (UTF-8, ensure_ascii=False)
    """
    ncfg = cfg["llm"]["news"]
    max_n = int(ncfg.get("max_articles_per_brand", 8))
    suffixes = list(ncfg.get("query_suffixes", []))
    # 경로는 호출 시점의 cfg['paths'] 로 해석 (데모 러너의 경로 스왑 대응)
    raw_dir = Path(cfg["paths"]["raw"]) / "news"
    raw_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, list] = {}
    for brand in brand_names:
        articles: list[dict] = []
        seen_links: set[str] = set()
        queries = [brand] + [f"{brand} {sfx}" for sfx in suffixes]
        for q in queries:
            if len(articles) >= max_n:
                break
            url = _RSS_URL.format(q=urllib.parse.quote(q))
            try:
                feed = feedparser.parse(url)
                entries = list(feed.entries or [])
                if getattr(feed, "bozo", 0) and not entries:
                    log.warning("RSS 파싱 경고(결과 없음): query=%r", q)
            except Exception as exc:  # 네트워크 등 실패 → 빈 결과
                log.warning("RSS 수집 실패(무시): query=%r err=%s", q, exc)
                entries = []
            for e in entries:
                link = str(getattr(e, "link", "") or "")
                if not link or link in seen_links:
                    continue
                seen_links.add(link)
                source = ""
                try:
                    src = getattr(e, "source", None)
                    if src is not None:
                        source = str(getattr(src, "title", "") or "")
                except Exception:
                    source = ""
                articles.append(
                    {
                        "title": str(getattr(e, "title", "") or ""),
                        "link": link,
                        "published": str(getattr(e, "published", "") or ""),
                        "source": source,
                    }
                )
                if len(articles) >= max_n:
                    break
            time.sleep(_SLEEP_BETWEEN_REQUESTS_SEC)

        snap_path = raw_dir / f"{_safe_name(brand)}.json"
        with open(snap_path, "w", encoding="utf-8") as f:
            json.dump(
                {"brand": brand, "queries": queries, "articles": articles},
                f,
                ensure_ascii=False,
                indent=2,
            )
        log.info("뉴스 수집: brand=%s articles=%d snapshot=%s", brand, len(articles), snap_path)
        out[brand] = articles
    return out


# ---------------------------------------------------------------------------
# 2) 구조화 추출 (LLM / 규칙기반 폴백)
# ---------------------------------------------------------------------------

def _make_signal(brand: str, article: dict, event_type: str, evidence: str,
                 confidence: str, llm_used: bool) -> dict:
    return {
        "brand": brand,
        "event_type": event_type,
        "date": _to_date(article.get("published", "")),
        "evidence_sentence": evidence,
        "confidence": confidence,
        "source_url": article.get("link", ""),
        "published": article.get("published", ""),
        "llm_used": llm_used,
    }


def _extract_with_rules(brand: str, articles: list[dict]) -> list[dict]:
    """규칙기반(키워드) 라이트 신호 — LLM 미사용 폴백. llm_used=false."""
    out = []
    for a in articles:
        title = a.get("title", "")
        event_type, confidence = "기타", "하"
        for etype, keywords in _KEYWORD_RULES:
            if any(kw in title for kw in keywords):
                event_type, confidence = etype, "중"
                break
        out.append(_make_signal(brand, a, event_type, title, confidence, llm_used=False))
    return out


def _extract_with_llm(brand: str, articles: list[dict], cfg: dict) -> list[dict] | None:
    """브랜드당 1회 배치 호출로 구조화 추출. 실패/거절 시 None (폴백 유도)."""
    try:
        import anthropic

        client = anthropic.Anthropic()
        titles_block = "\n".join(
            f"{i}. {a.get('title', '')}" for i, a in enumerate(articles)
        )
        prompt = (
            f"브랜드: {brand}\n\n"
            f"아래는 이 브랜드 관련 검색으로 수집된 기사 제목 목록입니다 (0-기반 인덱스).\n"
            f"각 기사를 event_type 으로 분류해 JSON으로 반환하세요. "
            f"모든 기사에 대해 하나씩 signals 항목을 만드세요.\n\n"
            f"{titles_block}"
        )
        resp = client.messages.create(
            model=cfg["llm"]["model"],
            max_tokens=int(cfg["llm"].get("max_tokens", 4000)),
            system=_EXTRACT_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": _SIGNAL_SCHEMA}},
        )
        if resp.stop_reason == "refusal":
            log.warning("LLM 추출 거절(refusal): brand=%s → 규칙기반 폴백", brand)
            return None
        text = next((b.text for b in resp.content if b.type == "text"), "")
        data = json.loads(text)
        out = []
        for item in data.get("signals", []):
            idx = item.get("article_index")
            if not isinstance(idx, int) or not (0 <= idx < len(articles)):
                log.warning("LLM 결과 인덱스 무시: brand=%s idx=%r", brand, idx)
                continue
            etype = item.get("event_type")
            if etype not in EVENT_TYPES:
                etype = "기타"
            conf = item.get("confidence")
            if conf not in ("상", "중", "하"):
                conf = "하"
            out.append(
                _make_signal(
                    brand,
                    articles[idx],
                    etype,
                    str(item.get("evidence_sentence", "")),
                    conf,
                    llm_used=True,
                )
            )
        return out
    except Exception as exc:
        log.warning("LLM 추출 실패: brand=%s err=%s → 규칙기반 폴백", brand, exc)
        return None


def extract_signals(articles_by_brand: dict[str, list], cfg: dict,
                    force_fallback: bool = False) -> list[dict]:
    """브랜드별 기사 → 구조화 뉴스 신호 목록. outputs/news_signals.json 저장.

    신호 스키마: {brand, event_type, date, evidence_sentence, confidence,
                 source_url, published, llm_used}
    - API 키(env `llm.api_key_env`)가 없거나 호출 실패/거절 시 규칙기반 폴백.
    - force_fallback=True 이면 키가 있어도 폴백 경로를 강제(테스트용).
    ⚠️ 이 신호는 모델 점수에 절대 투입하지 않는다 (화면 표시 전용).
    """
    key_env = cfg["llm"].get("api_key_env", "ANTHROPIC_API_KEY")
    use_llm = (not force_fallback) and bool(os.environ.get(key_env))
    if not use_llm:
        log.info(
            "LLM 미사용 (%s) — 규칙기반 키워드 폴백으로 추출",
            "force_fallback" if force_fallback else f"env {key_env} 없음",
        )

    signals: list[dict] = []
    for brand, articles in articles_by_brand.items():
        kept = [a for a in articles if _entity_match(brand, a.get("title", ""))]
        dropped = len(articles) - len(kept)
        if dropped:
            log.info("개체 매칭 필터: brand=%s 제외 %d건 (제목에 브랜드 부재)", brand, dropped)
        if not kept:
            continue
        brand_signals = _extract_with_llm(brand, kept, cfg) if use_llm else None
        if brand_signals is None:
            brand_signals = _extract_with_rules(brand, kept)
        signals.extend(brand_signals)

    out_path = Path(cfg["paths"]["outputs"]) / "news_signals.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(signals, f, ensure_ascii=False, indent=2)
    # ⚠️ 로그는 '의도(use_llm)'가 아니라 **실제 결과**를 보고해야 한다.
    #    (키가 있어도 refusal·예외·파싱실패로 전량 폴백될 수 있음 — 실측 확인된 결함 수정)
    n_llm = sum(1 for s in signals if s.get("llm_used"))
    log.info(
        "뉴스 신호 저장: %s (신호 %d건 중 LLM 추출 %d건 / 규칙 폴백 %d건, LLM 시도=%s) "
        "— 점수 미반영·화면 표시 전용",
        out_path, len(signals), n_llm, len(signals) - n_llm, use_llm,
    )
    return signals


# ---------------------------------------------------------------------------
# 3) 편의 실행
# ---------------------------------------------------------------------------

def run_news(brand_names: list[str], cfg: dict, force_fallback: bool = False) -> list[dict]:
    """수집 + 추출 + 저장 일괄 실행."""
    set_seed(cfg["seed"])
    articles = fetch_news(brand_names, cfg)
    return extract_signals(articles, cfg, force_fallback=force_fallback)


def _default_brands(cfg: dict, n: int = 10) -> list[str]:
    """패널이 있으면 최신 연도 점포수 상위 브랜드, 없으면 데모 브랜드."""
    panel_path = Path(cfg["paths"]["processed"]) / "panel.parquet"
    if panel_path.exists():
        try:
            import pandas as pd

            panel = pd.read_parquet(panel_path)
            latest = panel[panel["year"] == panel["year"].max()]
            return (
                latest.sort_values("n_stores", ascending=False)["brand_name"]
                .head(n)
                .tolist()
            )
        except Exception as exc:
            log.warning("패널에서 브랜드 목록 로드 실패: %s", exc)
    return ["메가커피", "컴포즈커피", "교촌치킨"]


if __name__ == "__main__":
    import sys

    _cfg = load_config()
    _brands = sys.argv[1:] or _default_brands(_cfg)
    _signals = run_news(_brands, _cfg)
    log.info("run_news 완료: 브랜드 %d개, 신호 %d건", len(_brands), len(_signals))
