"""네이버 오픈API — 검색어트렌드(수요)·뉴스·로고.

왜 검색어트렌드인가
    공시는 1년에 한 번, 그것도 이듬해에 나온다. 가맹점 수가 줄었다는 사실을 은행이
    아는 시점에는 이미 1~2년이 지난 뒤다. 반면 **사람들이 그 브랜드를 검색하는 양**은
    매장을 찾기 전 단계라 매출보다 먼저 움직이고, 월 단위로 즉시 관측된다.

    더 중요한 것은 **카테고리 대비 비교**다. '냉면' 카테고리 검색이 다 같이 줄었다면
    업종 축소지만, 카테고리는 보합인데 특정 브랜드만 빠졌다면 그 브랜드의 문제다.
    이 구분은 공시 데이터만으로는 절대 나오지 않는다.

API (developers.naver.com 애플리케이션 자격증명 — 무료, 일 25,000회)
    검색어트렌드  POST https://openapi.naver.com/v1/datalab/search
    뉴스 검색     GET  https://openapi.naver.com/v1/search/news.json
    헤더 X-Naver-Client-Id / X-Naver-Client-Secret

정직성
    · 키가 없으면 **수집하지 않는다**. 추정값이나 예시값을 만들어 넣지 않는다.
      키 부재는 산출물 메타에 `enabled: false` 로 남고, 진단 규칙은 수요 소견을
      아예 생성하지 않는다(없는 근거로 위험을 말하지 않는다).
    · 데이터랩은 절대 검색량을 주지 않고 기간 내 최대값을 100으로 한 **상대지수**를
      준다. 그래서 이 모듈은 '검색량 N건'이 아니라 항상 **증감률**만 산출한다.
"""
from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.common import get_logger, load_config

log = get_logger("naver")

ID_ENV = "NAVER_CLIENT_ID"
SECRET_ENV = "NAVER_CLIENT_SECRET"

_DATALAB_URL = "https://openapi.naver.com/v1/datalab/search"
_NEWS_URL = "https://openapi.naver.com/v1/search/news.json"

# 데이터랩 제약 (공식 문서): keywordGroups 최대 5개, 그룹당 keywords 최대 20개
MAX_GROUPS = 5
_RETRY_STATUS = {429, 500, 502, 503, 504}
_TIMEOUT = 20.0
_MAX_RETRIES = 4

# 업종 중분류 → 데이터랩에 던질 카테고리 대표어.
# 공시 업종명이 그대로 검색어로는 어색한 경우(예: '기타외국식')를 사람이 쓰는 말로 옮긴다.
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "치킨": ["치킨", "치킨 배달"],
    "커피": ["카페", "커피"],
    "제과제빵": ["베이커리", "빵집"],
    "피자": ["피자"],
    "햄버거": ["햄버거"],
    "한식": ["한식", "백반"],
    "분식": ["분식", "떡볶이"],
    "일식": ["일식", "초밥"],
    "중식": ["중식", "짜장면"],
    "서양식": ["파스타", "스테이크"],
    "주점": ["술집", "호프"],
    "아이스크림/빙수": ["아이스크림", "빙수"],
    "음료(커피 외)": ["음료", "주스"],
    "패스트푸드": ["패스트푸드"],
    "김밥": ["김밥"],
    "냉면": ["냉면"],
    "국수": ["국수", "칼국수"],
    "돼지고기구이": ["삼겹살", "고깃집"],
    "소고기구이": ["소고기", "한우"],
    "편의점": ["편의점"],
    "화장품": ["화장품"],
    "교육": ["학원"],
    "세탁": ["세탁소"],
    "이미용": ["미용실", "헤어샵"],
}

_PAREN = re.compile(r"[(（][^)）]*[)）]")
_NONWORD = re.compile(r"[^0-9A-Za-z가-힣 ]+")


class NaverError(RuntimeError):
    """네이버 API 호출 실패."""


# ---------------------------------------------------------------------------
# 자격증명
# ---------------------------------------------------------------------------

def credentials(cfg: dict | None = None) -> tuple[str, str]:
    c = (cfg or {}).get("naver", {}) if cfg else {}
    cid = os.environ.get(str(c.get("client_id_env", ID_ENV)), "").strip()
    sec = os.environ.get(str(c.get("client_secret_env", SECRET_ENV)), "").strip()
    return cid, sec


def is_enabled(cfg: dict | None = None) -> bool:
    cid, sec = credentials(cfg)
    return bool(cid and sec)


def _headers(cfg: dict | None = None, json_body: bool = False) -> dict:
    cid, sec = credentials(cfg)
    if not (cid and sec):
        raise NaverError(
            f"환경변수 {ID_ENV}/{SECRET_ENV} 미설정 — developers.naver.com 에서 "
            "애플리케이션을 등록하고 '검색'·'데이터랩(검색어트렌드)' API를 추가하세요.")
    h = {"X-Naver-Client-Id": cid, "X-Naver-Client-Secret": sec}
    if json_body:
        h["Content-Type"] = "application/json"
    return h


def _request(method: str, url: str, cfg: dict | None = None, **kw) -> dict:
    """재시도 포함 호출. 4xx(429 제외)는 재시도해도 같으므로 즉시 실패시킨다."""
    last: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            r = requests.request(method, url, timeout=_TIMEOUT, **kw)
            if r.status_code == 200:
                return r.json()
            body = r.text[:300]
            if r.status_code not in _RETRY_STATUS:
                raise NaverError(f"HTTP {r.status_code} (재시도 무의미): {body}")
            last = NaverError(f"HTTP {r.status_code}: {body}")
        except NaverError:
            raise
        except Exception as exc:                      # 네트워크 계층 오류
            last = exc
        if attempt < _MAX_RETRIES:
            time.sleep(1.5 * (2 ** (attempt - 1)))
    raise NaverError(f"네이버 API 호출 실패: {last}")


# ---------------------------------------------------------------------------
# 검색어 정규화
# ---------------------------------------------------------------------------

def search_term(brand_name: str) -> str:
    """공시 등록명 → 사람이 실제로 검색하는 말.

    '메가엠지씨커피(MEGA MGC COFFEE)' 처럼 영문 병기·괄호가 붙은 등록명을 그대로
    던지면 검색량이 0으로 나온다. 괄호와 특수문자를 걷어내고 앞부분만 쓴다.
    """
    t = unicodedata.normalize("NFKC", str(brand_name or "")).strip()
    t = _PAREN.sub("", t)
    t = _NONWORD.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:40]


def category_terms(industry_mid: str, industry_major: str = "") -> list[str]:
    for key, terms in CATEGORY_KEYWORDS.items():
        if key and key in str(industry_mid or ""):
            return terms
    mid = str(industry_mid or "").strip()
    if mid and len(mid) <= 10:
        return [mid]
    return [str(industry_major or "외식").strip()]


# ---------------------------------------------------------------------------
# 검색어트렌드
# ---------------------------------------------------------------------------

def _default_period(months: int = 24) -> tuple[str, str]:
    """데이터랩은 미래 날짜를 거절한다 → 어제까지로 끊는다."""
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=int(months * 30.44))
    return start.isoformat(), end.isoformat()


def datalab_trend(groups: list[tuple[str, list[str]]], cfg: dict | None = None,
                  start: str | None = None, end: str | None = None,
                  time_unit: str = "month") -> dict[str, list[dict]]:
    """검색어트렌드 조회. groups = [(그룹명, [키워드…]), …] (최대 5).

    Returns:
        {그룹명: [{"period": "2025-01-01", "ratio": 45.2}, …]}
    """
    if not groups:
        return {}
    if len(groups) > MAX_GROUPS:
        raise ValueError(f"데이터랩 keywordGroups 는 최대 {MAX_GROUPS}개입니다")
    s, e = (start, end) if (start and end) else _default_period()
    body = {
        "startDate": s, "endDate": e, "timeUnit": time_unit,
        "keywordGroups": [{"groupName": g, "keywords": [k for k in ks if k][:20]}
                          for g, ks in groups if any(ks)],
    }
    if not body["keywordGroups"]:
        return {}
    data = _request("POST", _DATALAB_URL, cfg, headers=_headers(cfg, json_body=True),
                    data=json.dumps(body, ensure_ascii=False).encode("utf-8"))
    return {str(r["title"]): list(r.get("data") or []) for r in (data.get("results") or [])}


def _yoy(points: list[dict]) -> float | None:
    """최근 12개월 평균 vs 그 앞 12개월 평균의 증감률.

    마지막 달 하나만 비교하면 계절성(냉면=여름, 붕어빵=겨울)에 그대로 휘둘린다.
    12개월 창끼리 비교하면 계절 성분이 양쪽에서 상쇄된다.
    """
    vals = [float(p.get("ratio", 0) or 0) for p in points]
    if len(vals) < 18:
        return None
    recent = vals[-12:]
    prior = vals[-24:-12] if len(vals) >= 24 else vals[:-12]
    if not prior:
        return None
    a, b = float(np.mean(prior)), float(np.mean(recent))
    if a <= 0:
        return None
    return b / a - 1.0


def _slope(points: list[dict]) -> float | None:
    """전체 기간 선형 추세 기울기 (평균 대비 월 변화율)."""
    vals = np.array([float(p.get("ratio", 0) or 0) for p in points], dtype=float)
    if len(vals) < 6 or vals.mean() <= 0:
        return None
    x = np.arange(len(vals), dtype=float)
    slope = float(np.polyfit(x, vals, 1)[0])
    return slope / float(vals.mean())


def brand_demand(brand_name: str, industry_mid: str, industry_major: str = "",
                 cfg: dict | None = None) -> dict | None:
    """브랜드 1개의 수요 신호 (브랜드 검색 + 카테고리 검색을 **한 번의 호출**로).

    두 그룹을 한 요청에 담아야 같은 정규화 기준(기간 내 최대=100)에서 비교된다.
    따로 호출하면 각자 100으로 재정규화돼 서로 비교할 수 없다.
    """
    term = search_term(brand_name)
    if not term:
        return None
    cats = category_terms(industry_mid, industry_major)
    cat_label = cats[0]
    s, e = _default_period()
    res = datalab_trend([(term, [term]), (cat_label, cats)], cfg, s, e)
    bpts = res.get(term) or []
    cpts = res.get(cat_label) or []
    if not bpts:
        return None
    out = {
        "term": term, "category": cat_label, "period": f"{s}~{e}",
        "brand_yoy": _yoy(bpts), "brand_slope": _slope(bpts),
        "category_yoy": _yoy(cpts) if cpts else None,
        "series": [{"d": str(p.get("period")), "v": float(p.get("ratio", 0) or 0)}
                   for p in bpts],
        "category_series": [{"d": str(p.get("period")), "v": float(p.get("ratio", 0) or 0)}
                            for p in cpts],
    }
    return out


# ---------------------------------------------------------------------------
# 뉴스·지역
# ---------------------------------------------------------------------------

_TAG = re.compile(r"<[^>]+>")


def _clean(s: str) -> str:
    t = _TAG.sub("", str(s or ""))
    for a, b in (("&quot;", '"'), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                 ("&apos;", "'"), ("&nbsp;", " ")):
        t = t.replace(a, b)
    return t.strip()


def news(query: str, cfg: dict | None = None, display: int = 20,
         sort: str = "date") -> list[dict]:
    """뉴스 검색 — 제목·본문요약·링크·발행일."""
    data = _request("GET", _NEWS_URL, cfg, headers=_headers(cfg),
                    params={"query": query, "display": min(int(display), 100),
                            "sort": sort})
    out = []
    for it in data.get("items", []):
        out.append({
            "title": _clean(it.get("title")),
            "summary": _clean(it.get("description")),
            "url": it.get("originallink") or it.get("link"),
            "published": it.get("pubDate", ""),
        })
    return out


_IMAGE_URL = "https://openapi.naver.com/v1/search/image"

# 로고 후보 질의 — 앞에서부터 시도한다. '로고'만 붙이면 간판·인테리어 사진이
# 섞이므로 CI/BI 같은 정확한 용어를 먼저 넣어 심볼이 걸릴 확률을 높인다.
_LOGO_QUERIES = ("{q} 로고 CI", "{q} 브랜드 로고", "{q} 로고")
# 로고로 보기 어려운 결과를 거른다 (메뉴판·매장 사진·인물)
_LOGO_STOPWORDS = ("메뉴", "인테리어", "매장", "간판", "창업설명회", "채용", "알바")


def _score_logo(item: dict) -> float:
    """이미지 1건의 '로고다움' 점수. 클수록 좋다. 0 이하면 후보 제외.

    정사각형에 가깝고, 너무 크지 않고, 제목에 로고/CI 가 있으면 가점.
    """
    try:
        w, h = float(item.get("sizewidth", 0)), float(item.get("sizeheight", 0))
    except (TypeError, ValueError):
        return 0.0
    if w <= 0 or h <= 0:
        return 0.0
    title = _clean(item.get("title", ""))
    if any(s in title for s in _LOGO_STOPWORDS):
        return 0.0
    ratio = min(w, h) / max(w, h)
    if ratio < 0.55:                            # 가로로 길쭉하면 배너·간판일 확률이 높다
        return 0.0
    score = ratio
    if any(k in title for k in ("로고", "CI", "BI", "심볼", "logo")):
        score += 0.45
    if 80 <= max(w, h) <= 800:                  # 로고 이미지의 통상 크기대
        score += 0.25
    return score


def brand_logo(brand_name: str, cfg: dict | None = None) -> str:
    """브랜드 로고 이미지 URL 1개. 못 찾으면 빈 문자열.

    ⚠️ 이미지 검색 결과는 제3자가 올린 것이라 브랜드 공식 자산이라는 보장이 없다.
       그래서 화면에는 **표시용 썸네일**로만 쓰고, 실패하면 글자 마크로 내려앉는다.
       로고를 못 구했다고 카드가 비어 보이는 일은 없어야 한다.
    """
    q = search_term(brand_name)
    if not q:
        return ""
    for tpl in _LOGO_QUERIES:
        try:
            data = _request("GET", _IMAGE_URL, cfg, headers=_headers(cfg),
                            params={"query": tpl.format(q=q), "display": 20,
                                    "sort": "sim", "filter": "all"})
        except NaverError as exc:
            if "재시도 무의미" in str(exc):      # 인증·권한 오류면 다음 질의도 실패한다
                raise
            return ""
        best, best_score = "", 0.0
        for it in data.get("items", []):
            s = _score_logo(it)
            if s > best_score:
                best, best_score = str(it.get("thumbnail") or it.get("link") or ""), s
        if best and best_score >= 0.9:          # 확신이 설 때만 채택
            return best
    return ""


def collect_logos(cfg: dict | None = None, limit: int | None = None,
                  sleep_sec: float = 0.1) -> dict:
    """점수 코호트 브랜드의 로고를 한꺼번에 찾아 캐시에 저장한다.

    화면에서 브랜드를 볼 때마다 API를 때리면 느리고 한도를 잡아먹는다.
    미리 채워 두면 화면은 캐시만 읽는다. 이미 찾은 브랜드는 건너뛴다.
    """
    cfg = cfg or load_config()
    dest = Path(cfg["_root"]) / "data/raw/naver/logos.json"
    cache: dict = {}
    if dest.exists():
        try:
            cache = json.loads(dest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}

    if not is_enabled(cfg):
        log.warning("네이버 자격증명이 없어 로고를 수집하지 않는다 — "
                    "developers.naver.com 에서 발급 후 %s/%s 설정", ID_ENV, SECRET_ENV)
        return {"enabled": False, "n_cached": len(cache)}

    spath = Path(cfg["paths"]["outputs"]) / "scores_latest.csv"
    if not spath.exists():
        raise FileNotFoundError(f"{spath} 없음 — 먼저 `--step score` 실행")
    scores = pd.read_csv(spath, encoding="utf-8-sig")
    scores = scores.assign(
        _n=pd.to_numeric(scores["n_stores"], errors="coerce").fillna(0))
    todo = scores.sort_values("_n", ascending=False)
    if limit:
        todo = todo.head(int(limit))

    found, missing = 0, 0
    for _, r in todo.iterrows():
        name = str(r["brand_name"])
        if name in cache:
            continue
        try:
            url = brand_logo(name, cfg)
        except NaverError as exc:
            log.warning("로고 조회 중단 (%s): %s", name, exc)
            break
        cache[name] = url or ""
        found += bool(url)
        missing += (not url)
        if (found + missing) % 50 == 0:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                            encoding="utf-8")
            log.info("로고 수집 진행: 확보 %d · 미확보 %d", found, missing)
        time.sleep(sleep_sec)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    n_ok = sum(1 for v in cache.values() if v)
    log.info("로고 수집 완료: 이번 실행 확보 %d · 미확보 %d · 누적 확보 %d/%d (%.0f%%) → %s",
             found, missing, n_ok, len(cache),
             100 * n_ok / max(len(cache), 1), dest.name)
    return {"enabled": True, "n_found": found, "n_missing": missing,
            "n_cached": len(cache), "n_with_logo": n_ok}


# ---------------------------------------------------------------------------
# 배치 수집 (파이프라인 스텝)
# ---------------------------------------------------------------------------

def collect_demand(cfg: dict | None = None, limit: int | None = None,
                   sleep_sec: float = 0.12) -> dict:
    """점수 코호트 상위 브랜드의 검색수요를 모아 outputs/demand_trends.json 으로 저장.

    호출량: 브랜드 1개당 데이터랩 1회. 일 25,000회 한도 안에서 상위 N개를 덮는다.
    이미 수집된 브랜드는 건너뛰므로 여러 날에 나눠 넓힐 수 있다.
    """
    cfg = cfg or load_config()
    out_dir = Path(cfg["paths"]["outputs"])
    dest = out_dir / "demand_trends.json"
    ncfg = cfg.get("naver", {}) or {}
    limit = int(limit if limit is not None else ncfg.get("demand_top_n", 300))

    prev: dict = {}
    if dest.exists():
        try:
            prev = json.loads(dest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prev = {}
    brands: dict = dict(prev.get("brands") or {})

    if not is_enabled(cfg):
        meta = {"enabled": False, "reason": f"{ID_ENV}/{SECRET_ENV} 미설정",
                "n_collected": len(brands), "brands": brands}
        dest.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        log.warning("네이버 자격증명이 없어 검색수요를 수집하지 않는다 — "
                    "developers.naver.com 에서 발급 후 %s/%s 설정", ID_ENV, SECRET_ENV)
        return meta

    spath = out_dir / "scores_latest.csv"
    if not spath.exists():
        raise FileNotFoundError(f"{spath} 없음 — 먼저 `--step score` 실행")
    scores = pd.read_csv(spath, encoding="utf-8-sig")
    # 규모×위험 순으로 훑는다 — 여신 영향이 큰 브랜드부터 덮어야 한도 안에서 이득이 크다
    scores = scores.assign(_pri=(pd.to_numeric(scores["pd_1y"], errors="coerce").fillna(0)
                                 * pd.to_numeric(scores["n_stores"], errors="coerce").fillna(0)))
    todo = scores.sort_values("_pri", ascending=False).head(limit)

    n_new, n_fail = 0, 0
    for _, r in todo.iterrows():
        bid = str(r["brand_id"])
        if brands.get(bid):
            continue
        try:
            d = brand_demand(str(r["brand_name"]), str(r.get("industry_mid") or ""),
                             str(r.get("industry_major") or ""), cfg)
        except NaverError as exc:
            n_fail += 1
            log.warning("수요 조회 실패 (%s): %s", r["brand_name"], exc)
            if "재시도 무의미" in str(exc):
                break                       # 인증·권한 오류면 나머지도 전부 실패한다
            continue
        brands[bid] = d
        if d:
            n_new += 1
        time.sleep(sleep_sec)

    covered = {k: v for k, v in brands.items() if v}
    meta = {
        "enabled": True,
        "period": _default_period()[0] + "~" + _default_period()[1],
        "n_requested": len(todo), "n_collected": len(covered),
        "n_new_this_run": n_new, "n_failed": n_fail,
        "source": "네이버 데이터랩 검색어트렌드 (상대지수, 기간 내 최대=100)",
        "brands": brands,
    }
    dest.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("검색수요 수집: 신규 %d개 · 누적 %d개 · 실패 %d개 → %s",
             n_new, len(covered), n_fail, dest.name)
    return meta


if __name__ == "__main__":
    collect_demand()
