"""브랜드 검색 — 사람이 부르는 이름으로 공시 등록명을 찾는다.

왜 필요한가 (실사용에서 드러난 문제):
    사용자가 "메가커피"를 찾지 못했다. 데이터에는 있다 — 다만 공시 등록명이
    **"메가엠지씨커피(MEGA MGC COFFEE)"** 라서 단순 문자열 일치로는 걸리지 않는다.
    공시명은 법인이 등록한 정식 명칭이라 괄호·영문 병기·법인 접두가 섞여 있고,
    사람이 쓰는 통칭과 자주 다르다. 검색이 이 간극을 메우지 못하면
    "우리 데이터에 그 브랜드가 없다"는 잘못된 결론이 난다.

전략 (앞에서부터 순서대로 시도):
    1. 정규화 부분일치 — 공백·괄호·영문·특수문자를 제거해 비교
       ("메가커피" → "메가커피", "메가엠지씨커피(MEGA MGC COFFEE)" → "메가엠지씨커피")
    2. 통칭 별칭 사전 — 위 규칙으로도 안 걸리는 널리 쓰이는 이름을 명시적으로 연결
    3. 초성/부분 토큰 — 앞 2글자 + 뒤 2글자가 모두 포함되면 후보로 (news_llm 과 같은 규칙)
    4. 유사도 순위 — 그래도 없으면 가장 가까운 이름을 제안한다("혹시 이것을 찾으셨나요")
"""
from __future__ import annotations

import difflib
import re
import unicodedata

import pandas as pd

# 사람이 부르는 통칭 → 공시 등록명에 포함된 핵심 토큰.
# 규칙 기반 정규화로 못 잡는 것만 최소한으로 둔다(사전이 커지면 유지보수가 어렵다).
ALIASES: dict[str, str] = {
    "메가커피": "메가엠지씨",
    "메가엠지씨커피": "메가엠지씨",
    "mgc커피": "메가엠지씨",
    "빽다방커피": "빽다방",
    "베스킨라빈스": "배스킨라빈스",
    "베라": "배스킨라빈스",
    "던킨도너츠": "던킨",
    "비비큐": "bbq",
    "비에이치씨": "bhc",
    "씨유": "cu",
    "지에스25": "gs25",
    "쥐에스25": "gs25",
    "파바": "파리바게뜨",
    "뚜쥬": "뚜레쥬르",
    "스벅": "스타벅스",
    "맥날": "맥도날드",
    "롯데리아": "롯데리아",
    "버거킹": "burger king",
}

_STRIP = re.compile(r"[\s()（）\[\]{}·・,.\-_/&'\"]+")


def normalize(s: str) -> str:
    """검색 비교용 정규화: NFKC → 소문자 → 구분기호 제거."""
    t = unicodedata.normalize("NFKC", str(s or "")).lower()
    return _STRIP.sub("", t)


def _korean_only(s: str) -> str:
    """한글만 남긴다 — 영문 병기가 붙은 공시명과 통칭을 맞추기 위한 보조 키."""
    return re.sub(r"[^가-힣]", "", str(s or ""))


def suggest(names: list[str], query: str, limit: int = 8) -> list[str]:
    """연관검색어 — **한 글자만 쳐도** 후보를 돌려준다.

    왜 search() 와 따로인가
        search() 는 '조회 결과'를 만드는 함수라 매칭이 없으면 빈 결과를 준다.
        자동완성은 타이핑 도중에 계속 불리므로 '아직 다 안 친 상태'가 정상이고,
        접두 일치를 최우선으로 정렬해 사람이 고를 수 있게 해야 한다.

    정렬: 접두 일치 → 포함 위치가 앞선 것 → 이름이 짧은 것.
    짧은 이름을 앞에 두는 이유는 '메가엠지씨커피(MEGA MGC COFFEE)' 같은 긴 등록명보다
    통칭에 가까운 이름이 사용자가 찾던 것일 확률이 높기 때문이다.
    """
    q = normalize(query)
    if not q:
        return []
    keys = {q}
    for alias, token in ALIASES.items():
        if q in alias or alias in q:
            keys.add(normalize(token))

    scored: list[tuple[int, int, int, str]] = []
    seen: set[str] = set()
    for name in names:
        s = str(name)
        if s in seen:
            continue
        n = normalize(s)
        k_hit = _korean_only(s)
        pos = -1
        for k in keys:
            for hay in (n, k_hit):
                p = hay.find(k) if k else -1
                if p >= 0 and (pos < 0 or p < pos):
                    pos = p
        if pos < 0:
            continue
        seen.add(s)
        scored.append((0 if pos == 0 else 1, pos, len(s), s))
    scored.sort()
    return [s for *_, s in scored[:limit]]


def search(df: pd.DataFrame, query: str, col: str = "brand_name",
           limit: int = 50) -> tuple[pd.DataFrame, list[str]]:
    """브랜드 검색.

    반환: (매칭된 행, 제안 목록)
    - 매칭이 있으면 제안은 빈 리스트.
    - 매칭이 없으면 가장 가까운 이름 몇 개를 제안한다(오타·통칭 대응).
    """
    q = normalize(query)
    if not q:
        return df.head(0), []

    names = df[col].astype(str)
    norm = names.map(normalize)
    kor = names.map(_korean_only)

    # 1) 별칭 → 핵심 토큰으로 치환해서도 시도
    keys = {q}
    if q in ALIASES:
        keys.add(normalize(ALIASES[q]))
    for alias, token in ALIASES.items():
        if q in alias or alias in q:
            keys.add(normalize(token))

    mask = pd.Series(False, index=df.index)
    for k in keys:
        mask |= norm.str.contains(re.escape(k), na=False)
        mask |= kor.str.contains(re.escape(_korean_only(k)), na=False) if _korean_only(k) else False

    # 2) 앞2·뒤2 토큰 규칙 (예: '메가커피' vs '메가엠지씨커피')
    if not mask.any() and len(q) >= 4:
        head, tail = re.escape(q[:2]), re.escape(q[-2:])
        mask |= norm.str.contains(head, na=False) & norm.str.contains(tail, na=False)

    hit = df[mask]
    if len(hit):
        return hit.head(limit), []

    # 3) 근접 제안
    sugg = difflib.get_close_matches(q, norm.dropna().unique().tolist(), n=5, cutoff=0.5)
    proposals = names[norm.isin(sugg)].drop_duplicates().tolist()[:5]
    return hit, proposals
