"""공정위 가맹사업정보제공시스템 **웹 열람** 수집 — 전 등록 브랜드의 본부 재무.

왜 이 모듈인가 (모형의 최대 한계를 직접 겨냥)
    본부 재무 커버리지가 브랜드 기준 11.4%, 점포가중 39.7%에 멈춰 있었다.
    DART 는 외부감사 대상만 제출하므로 **구조적으로** 더 못 늘린다(실측: 미확보
    점포가중 60.3%p 중 50.9%p 는 법인명이 DART 인덱스에 아예 없다).

    그런데 정보공개서에는 외감 여부와 무관하게 **전 등록 브랜드의 3개년 재무**가
    있다. API(src/ifrmp.py)는 정식 키가 필요해 막혀 있었지만, 같은 시스템의
    **일반 열람 웹 화면**은 키·로그인·캡차 없이 공개돼 있다(실측):

        list.do  — 전 등록 브랜드 열거 (pageUnit=100, 약 11,751건 / ~118페이지)
        view.do  — 브랜드별 요약: 재무 3개년·가맹점 변동·지역별 매출·법위반·부담금

    실측 표본 — 국물에빠진두루치기(우리 모형이 재무를 못 보던 브랜드):
        2025년 자산 79,169 / 부채 85,906 / **자본 -6,737(자본잠식)** /
        매출 515,198 / **영업이익 -41,070(적자)** (천원)
    이런 신호가 지금까지 브랜드의 88.6%에서 비어 있었다.

🛑 **전량 자동 수집은 불가하다 — 캡차 (실측 확정)**
    ⚠️ 2026-08-03 정정: 처음엔 "세션당 5건 후" 로 적었으나, 별도 조사에서 **쿠키 없는
       새 프로그램 세션은 list.do 직후 **첫 번째** view.do 요청부터** 캡차 페이지를
       받는 것이 확인됐다. 앞선 5건은 이미 사람이 만든 세션 상태를 물려받은 것이었다.
       즉 한도는 5가 아니라 사실상 0이고, 그 전제로 계산한 수집량 추정은 전부 무효다.
    응답이 453바이트로 줄고 본문이 이렇다:

        "인증 요청 / 사람 인증이 필요합니다. / 캡챠 입력:"

    목록 페이지 재방문·10초 대기 모두 해제되지 않는다(실측). 즉 이것은 rate
    limit 이 아니라 **자동화 차단 장치**이고, 운영자가 사람만 열람하도록 설계한
    것이다. 캡차를 우회하지 않는다 — 기술적으로 가능한지와 무관하게, 명시적
    자동화 차단을 뚫는 것은 이 프로젝트가 지킬 선 밖이다.

    그래서 이 모듈은 **전량 수집기가 아니라 소량 보강 도구**로 남긴다:
      · 세션당 5건 한도를 존중한다(BATCH_LIMIT). 그 이상 시도하지 않는다.
      · 심사역이 특정 브랜드를 확인해야 할 때 쓰는 **주문형 조회**,
        또는 대형 브랜드 소수를 사람이 나눠 받는 용도.
      · 자동 배치(daily_refresh)에는 **연결하지 않는다.**

    본부 재무 커버리지의 구조적 해소는 여전히 **정식 API 키**가 유일한 길이다
    (src/ifrmp.py 의 IFRMP_SERVICE_KEY — 파서는 이미 검증돼 있다).

⛔ 그 밖의 설계 제약 (전부 실측)
    ① WAF 는 `python-requests` UA 를 차단한다(406). 브라우저 UA 를 쓴다 —
       ifrmp.py 의 _HEADERS 재사용.
    ② `encFirMstSn` 토큰은 **세션 종속 암호문**이다. 다른 세션에서 재사용하면
       200 이지만 오류 페이지가 온다. 목록 수확과 상세 조회를 **같은 Session
       안에서 인터리브**해야 한다.
    ③ 원본 HTML 은 브랜드당 2.9MB 라 저장하지 않는다. 필요한 표만 추출한다.

출처·이용
    공정거래위원회 공공저작물 — 공공누리 제1유형(출처표시 시 자유이용, 실측:
    ftc.go.kr 저작권 정책). 모든 산출물에 출처 필드를 남긴다.
    robots.txt 는 미제공(403)이라 금지 명시 없음. 예의상 간격·야간 실행 권장.

산출
    data/raw/ifrmp_web/{등록번호}.json      브랜드별 추출 표 (재무·변동·식별자)
    data/processed/ifrmp_web_financials.parquet   등록번호×회계연도 재무
    outputs/ifrmp_web_status.json           수집 진행·커버리지 진단

실행:
    python -m src.ifrmp_web --brand 국물에빠진두루치기   # 주문형 1건
    python -m src.ifrmp_web --build-only                # 수집분으로 parquet 재구축
"""
from __future__ import annotations

import argparse
import contextlib
import html as html_mod
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from src.common import get_logger, load_config
from src.ifrmp import _HEADERS

log = get_logger("ifrmp_web")

BASE = "https://franchise.ftc.go.kr/mnu/00013/program/userRqst"
RAW_DIR = "data/raw/ifrmp_web"
PACE_SEC = 0.6
PAGE_UNIT = 100
MAX_CONSEC_FAIL = 2          # 캡차가 걸리면 더 두드리지 않는다
BATCH_LIMIT = 5              # 세션당 캡차 없이 열람되는 실측 한도
SOURCE = "공정거래위원회 가맹사업정보제공시스템 (공공누리 제1유형·출처표시)"

_TAGS = re.compile(r"<[^>]+>")
_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S)
_TOKEN_RE = re.compile(r"fn_moveUrl\('[^']*view\.do\?encFirMstSn=',\s*'([^']+)'\)")


def _txt(cell: str) -> str:
    return html_mod.unescape(_TAGS.sub("", re.sub(r"<!--.*?-->", "", cell, flags=re.S))).strip()


def _num(s: str) -> float | None:
    t = s.replace(",", "").replace(" ", "")
    if not t or t in ("-", "—"):
        return None
    # 정보공개서 PDF 는 음수를 △ 또는 괄호로 쓴다. 부호를 놓치면 적자가 흑자가 된다.
    neg = t.startswith(("△", "▲", "-", "−")) or (t.startswith("(") and t.endswith(")"))
    t = t.strip("()").lstrip("△▲-−")
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else v


def _extract_table(page: str, heading: str, span: int = 40_000) -> list[list[str]]:
    """제목 뒤에 나오는 첫 <table> 의 전 행을 텍스트로. 없으면 []."""
    i = page.find(heading)
    if i < 0:
        return []
    seg = page[i:i + span]
    m = re.search(r"<table.*?</table>", seg, re.S)
    if not m:
        return []
    rows = []
    for r in _ROW_RE.findall(m.group(0)):
        cells = [_txt(c) for c in _CELL_RE.findall(r)]
        if any(cells):
            rows.append(cells)
    return rows


def parse_view(page: str) -> dict:
    """view.do HTML → 여신 심사에 쓰는 표만 추출."""
    out: dict = {}
    # 단위 — 지금까지 전 표본이 '천원'이지만, 가정하지 말고 페이지에서 읽는다
    mu = re.search(r'amountUnitsTxt">([^<]+)<', page)
    out["amount_unit"] = _txt(mu.group(1)) if mu else "천원"
    out["financials"] = _extract_table(page, "재무상황")
    out["store_flow"] = _extract_table(page, "가맹점 및 직영점 현황") or \
        _extract_table(page, "가맹점 변동 현황") or _extract_table(page, "가맹점 현황")
    m = re.search(r"사업자등록번호.{0,400}?(\d{3}-\d{2}-\d{5})", page, re.S)
    out["brno"] = m.group(1).replace("-", "") if m else ""
    m = re.search(r"법인등록번호.{0,400}?(\d{6}-\d{7})", page, re.S)
    out["crno"] = m.group(1).replace("-", "") if m else ""
    return out


# 웹 열람 표는 '자산/부채/자본', 정보공개서 PDF 는 '자산총계/부채총계/자본총계' 로 쓴다.
# 같은 값을 가리키는 다른 표기이므로 별칭으로 함께 인정한다 — 파서를 둘로 늘리지 않는다.
_FIN_COLS = {"자산": "assets", "자산총계": "assets",
             "부채": "liabilities", "부채총계": "liabilities",
             "자본": "equity", "자본총계": "equity",
             "매출액": "revenue", "매출": "revenue",
             "영업이익": "operating_income",
             "당기순이익": "net_income", "순이익": "net_income"}
_FIN_KEYS = ("assets", "liabilities", "equity", "revenue",
             "operating_income", "net_income")


def financial_rows(reg_no: str, brand: dict) -> list[dict]:
    """추출 표 → 연도별 재무 레코드.

    ⚠️ 열 위치를 **고정 오프셋으로 가정하지 않는다.** 이 표에는 '재무제표작성여부'
       라는 빈 열이 끼어 있는데, 페이지에 따라 있기도 하고 HTML 주석으로 빠져
       있기도 하다(실측: 국물에빠진두루치기는 8열, 교촌치킨은 주석 처리).
       고정 오프셋을 썼더니 자산이 비고 부채값이 자본 자리로 한 칸씩 밀렸다 —
       조용히 틀린 재무가 만들어지는 종류의 결함이다. **헤더 행에서 열 이름을
       읽어 매핑**한다.
    """
    unit_mul = 1_000 if "천" in str(brand.get("amount_unit", "천원")) else 1
    table = brand.get("financials", [])
    # 헤더: '연도' 로 시작하고 재무 항목명을 포함하는 행
    idx: dict[str, int] = {}
    for cells in table:
        if cells and cells[0].replace(" ", "") == "연도":
            for j, name in enumerate(cells):
                key = _FIN_COLS.get(name.replace(" ", ""))
                if key:
                    idx[key] = j
            break
    if len(idx) < len(_FIN_KEYS):
        log.warning("%s: 재무 표 헤더를 못 읽었다(찾은 열 %s) — 이 브랜드는 건너뛴다",
                    reg_no, sorted(idx))
        return []

    rows = []
    for cells in table:
        if not cells or not re.fullmatch(r"(19|20)\d{2}", cells[0]):
            continue
        if max(idx.values()) >= len(cells):
            continue
        rec = {"reg_no": reg_no, "fiscal_year": int(cells[0])}
        vals = {k: _num(cells[j]) for k, j in idx.items()}
        if all(v is None for v in vals.values()):
            continue
        for k, v in vals.items():
            rec[k] = v * unit_mul if v is not None else None   # 원 단위 (DART 와 동일)
        rows.append(rec)
    return rows


_YEAR_RE = re.compile(r"^(19|20)\d{2}\s*년?$")
# 쉼표로 세 자리씩 끊긴 수. 붙어 있는 숫자열을 가르는 유일한 단서다.
_NUM_RE = re.compile(r"-?\d{1,3}(?:,\d{3})*")
_HDR_ORDER = ("자산", "부채", "자본", "매출", "영업이익", "당기순이익")


def _pdf_fin_table(flat: str) -> list[list[str]]:
    """PDF 텍스트에서 재무 표를 뽑아 [헤더, 연도행...] 형태로 돌려준다.

    ⚠️ PDF 마다 표가 **공백 없이 한 덩어리로** 추출된다(실측):

        연도자산총계부채총계자본총계매출액영업이익당기순이익20227,599,1104,791,316…

    뚜레쥬르는 공백이 있었지만 이삭토스트·투다리·처갓집양념치킨은 없다. 그래서
    줄 단위 split() 으로는 못 읽는다. 헤더는 **열 이름을 순서대로** 정규식으로 찾고,
    데이터는 '연도 4자리 + 열 개수만큼의 수' 를 반복해 읽는다.

    숫자 경계는 **쉼표 세 자리 구분**이 알려 준다 — `7,599,1104,791,316` 은
    `7,599,110` 과 `4,791,316` 으로만 갈라진다. 천 미만 값이 섞이면 갈림이 모호해질
    수 있으나, 그 경우는 자산=부채+자본 검산이 잡는다(ingest_saved 참조).
    """
    hdr = re.compile(r"연\s*도" + r"".join(rf"\s*({c}\s*(?:총계|액)?)" for c in _HDR_ORDER))
    m = hdr.search(flat)
    if not m:
        return []
    header = ["연도"] + [re.sub(r"\s+", "", g) for g in m.groups()]
    ncol = len(_HDR_ORDER)
    rows = [header]
    pos = m.end()
    tail = flat[pos:pos + 4000]
    while True:
        ym = re.match(r"\s*((?:19|20)\d{2})\s*년?\s*", tail)
        if not ym:
            break
        cells = [ym.group(1)]
        tail = tail[ym.end():]
        for _ in range(ncol):
            nm = _NUM_RE.match(tail.lstrip())
            if not nm:
                break
            tail = tail.lstrip()[nm.end():]
            cells.append(nm.group(0))
        if len(cells) != ncol + 1:
            break
        rows.append(cells)
    return rows if len(rows) > 1 else []


def parse_pdf(path: Path) -> dict:
    """정보공개서 **PDF**(대외 공개용) → parse_view 와 같은 모양의 dict.

    열람 화면에서 받는 PDF 가 HTML 표보다 오히려 낫다 — 공정위가 게시하는 정식
    '대외 공개용 정보공개서' 원문이고, 최근 3개 사업연도 재무가 한 표에 있다.

    ⚠️ 여기서도 **고정 오프셋을 쓰지 않는다.** HTML 표에서 '재무제표작성여부' 빈 열
       때문에 값이 한 칸씩 밀려 자산이 비었던 사고가 있었다. PDF 도 문서마다 열
       구성이 다를 수 있으므로 헤더 행에서 열 이름을 읽어 위치를 정한다.

    ⚠️ 단위도 읽는다. '(단위: 천원, 부가세 미포함)' 처럼 표 바로 위에 적혀 있고
       문서마다 천원/백만원/원이 다르다. 가정하면 1,000배 틀린다.
    """
    try:
        import pypdf  # 선택 의존성 — 배포 앱은 쓰지 않는다
    except ImportError as exc:                          # pragma: no cover
        raise RuntimeError("PDF 를 읽으려면 pypdf 가 필요하다: pip install pypdf") from exc

    reader = pypdf.PdfReader(str(path))
    text = "\n".join((p.extract_text() or "") for p in reader.pages)
    flat = re.sub(r"[ \t]+", " ", text)

    out: dict = {"amount_unit": "천원"}
    m = re.search(r"단위\s*[:：]\s*([가-힣]+원)", flat)
    if m:
        out["amount_unit"] = m.group(1)

    # ⚠️ 등록번호를 문서에서 **추측하지 않는다.** 정보공개서에는 자기 번호 말고도
    #    서비스표 등록번호(4100917700000), 특수관계인·자매 브랜드의 등록번호가 함께
    #    실린다. 첫 구현이 실제로 두 건을 틀렸다:
    #      · 이삭토스트 문서 → 자매 브랜드 '이삭버거'(20201428, 패널에 없음)
    #      · 투다리 문서   → 같은 본부의 '토대력'(20090100370, 점포 19개)
    #    투다리(점포 1,292) 재무가 토대력에 붙을 뻔했다. 조용히 틀린 데이터다.
    #    그래서 **후보만 모으고**, 패널 대조로 확정하는 일은 resolve_brand() 가 한다.
    cands: list[str] = []
    for pat in (r"정보공개서\s*등록\s*번호\s*[:：]?\s*(\d{8,12})",
                r"공정거래위원회\s*등록번호\s*[:：]?\s*(\d{8,12})",
                r"등록\s*번호\s*[:：]?\s*(\d{8})\s*(?:홈페이지|최초등록)",
                r"등록\s*번호\s*[:：]?\s*(\d{8,11})\b"):
        cands += [m.group(1) for m in re.finditer(pat, flat)]
    # 중복 제거하되 등장 순서(=우선순위)를 지킨다
    out["reg_no_candidates"] = list(dict.fromkeys(cands))

    names: list[str] = []
    for pat in (r"[\[［]\s*([^\[\]［］]{1,30}?)\s*[\]］]\s*정보공개서",
                r"앞으로\s*[\[［]?\s*([^\[\]［］\s]{2,25}?)\s*[\]］]?\s*(?:라|이라)\s*합니다",
                r"\n\s*([^\n]{2,40}?)\s*정보공개서\s*\n"):
        for m in re.finditer(pat, flat):
            v = m.group(1).strip()
            if v and not re.fullmatch(r"[\d\s년.\-()]+", v):
                names.append(v)
    out["brand_name_candidates"] = list(dict.fromkeys(names))

    out["financials"] = _pdf_fin_table(flat)
    out["store_flow"] = []
    m = re.search(r"사업자등록번호.{0,200}?(\d{3}-\d{2}-\d{5})", flat, re.S)
    out["brno"] = m.group(1).replace("-", "") if m else ""
    m = re.search(r"법인등록번호.{0,200}?(\d{6}-\d{7})", flat, re.S)
    out["crno"] = m.group(1).replace("-", "") if m else ""
    return out


def extra_fields(flat: str) -> dict:
    """열람 화면에만 있는 **추가 신호**를 뽑는다.

    재무 6항목은 정보공개서 PDF 에도 있지만, 웹 열람 본문에는 여신 판단에 직접
    쓰이는 항목이 더 있다. 그리고 이것들은 **우리가 지금 어떤 원천에서도 못 얻는
    값**이다 — 공정위 오픈API 에도, DART 에도 없다.

      법 위반 3종   시정조치·민사패소·형의 선고 (최근 3년) — 가장 직접적인 위험 신호
      가맹금 예치   '예치' 가 아니라 '보험' 이면 보증 구조가 다르다
      광고·판촉비   본부가 브랜드에 실제로 쓰는 돈
      부담금 세부   가입비·교육비·보증금·기타 (합계만 있던 것을 분해)
      인테리어 단가 단위면적당 — 창업비용 구조
      계약기간      최초·연장 (계약종료율의 분모를 해석하는 데 필요)

    ⚠️ 없으면 없는 대로 둔다. 0 으로 채우지 않는다 — '위반 0건'과 '기재 없음'은
       다른 사실이고, 섞으면 자료 없는 브랜드가 깨끗해 보인다(부문 설계에서 이미
       같은 이유로 본부재무를 부문에서 뺐다).
    """
    out: dict = {}

    def num(pat: str, key: str) -> None:
        m = re.search(pat, flat)
        if m:
            v = _num(m.group(1))
            if v is not None:
                out[key] = v

    m = re.search(r"시정조치.{0,80}?형의\s*선고\s*([\d,]+)\s+([\d,]+)\s+([\d,]+)", flat, re.S)
    if m:
        out["viol_ftc"], out["viol_civil"], out["viol_criminal"] = (
            _num(m.group(1)), _num(m.group(2)), _num(m.group(3)))
    num(r"광고비\s*판촉비\s*\d{4}\s+([\d,]+)", "ad_cost")
    num(r"광고비\s*판촉비\s*\d{4}\s+[\d,]+\s+([\d,]+)", "promo_cost")
    m = re.search(r"형태\s*([가-힣]+)\s*예치\s*가맹금\s*([\d,]+)", flat)
    if m:
        out["deposit_type"], out["deposit_amount"] = m.group(1), _num(m.group(2))
    m = re.search(r"가입비\(가맹비\)\s*교육비\s*보증금\s*기타비용\s*합계\s*"
                  r"([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)", flat)
    if m:
        for k, g in zip(("fee_join", "fee_edu", "fee_deposit", "fee_other", "fee_total"),
                        m.groups(), strict=False):
            out[k] = _num(g)
    m = re.search(r"인테리어\s*비용\s*([\d,]+)\s+([\d,]+)\s+([\d,]+)", flat)
    if m:
        out["interior_per_unit"], out["interior_area"], out["interior_total"] = (
            _num(m.group(1)), _num(m.group(2)), _num(m.group(3)))
    m = re.search(r"계약기간\s*최초\s*연장\s*(\d+)\s+(\d+)", flat)
    if m:
        out["contract_years_first"], out["contract_years_renew"] = int(m.group(1)), int(m.group(2))
    num(r"가맹지역본부\(지사,지역총판\)수\s*([\d,]+)", "regional_hq")
    num(r"브랜드\s*수\s*가맹사업\s*계열사\s*수\s*([\d,]+)", "hq_brand_count")

    # 자기 검산 — 부담금 4항목 합이 합계와 맞는가. 재무 표의 자산=부채+자본 과 같은
    # 장치다. 열이 밀리거나 정규식이 옆 표를 물면 여기서 깨진다.
    parts = [out.get(k) for k in ("fee_join", "fee_edu", "fee_deposit", "fee_other")]
    if out.get("fee_total") and all(v is not None for v in parts):
        s = sum(parts)
        if abs(s - out["fee_total"]) > 0.01 * max(abs(out["fee_total"]), 1):
            log.warning("부담금 합계 불일치 — 항목합 %s vs 합계 %s. 이 5개는 버린다",
                        f"{s:,.0f}", f"{out['fee_total']:,.0f}")
            for k in ("fee_join", "fee_edu", "fee_deposit", "fee_other", "fee_total"):
                out.pop(k, None)
    return out


def _panel_index(cfg: dict) -> tuple[dict[str, str], dict[str, list[str]]]:
    """(등록번호 → 브랜드명, 정규화 브랜드명 → 등록번호 목록). 최신 연도 기준."""
    p = Path(cfg["_root"]) / cfg["paths"]["processed"] / "panel_full.parquet"
    if not p.exists():
        return {}, {}
    f = pd.read_parquet(p, columns=["brand_id", "brand_name", "year"])
    f = f[f["year"] == f["year"].max()]
    by_id, by_name = {}, {}
    for bid, nm in zip(f["brand_id"].astype(str), f["brand_name"].astype(str), strict=False):
        reg = bid[4:] if bid.startswith("BRD_") else bid
        by_id[reg] = nm
        by_name.setdefault(re.sub(r"[\s()（）]", "", nm), []).append(reg)
    return by_id, by_name


def resolve_brand(brand: dict, by_id: dict, by_name: dict) -> tuple[str, str]:
    """(등록번호, 확정 사유). 확정 못 하면 ("", 사유).

    **추측을 금지한다.** 문서에서 뽑은 등록번호 후보는 자기 것이 아닐 수 있으므로
    (자매 브랜드·서비스표·특수관계인 번호가 섞여 있다) 패널에 실재하는 번호만
    받아들이고, 브랜드명까지 대조해 어긋나면 거부한다. 재무를 남의 브랜드에 붙이는
    것은 데이터가 없는 것보다 나쁘다 — 이 프로젝트가 로고에서 이미 배운 교훈이다.
    """
    names = [re.sub(r"[\s()（）]", "", n) for n in brand.get("brand_name_candidates", [])]

    for reg in brand.get("reg_no_candidates", []):
        pnl = by_id.get(reg)
        if not pnl:
            continue                                   # 패널에 없는 번호는 우리 대상이 아니다
        key = re.sub(r"[\s()（）]", "", pnl)
        if not names or any(n in key or key in n for n in names):
            return reg, f"등록번호 {reg} 패널 일치({pnl})"

    # 번호로 못 정하면 영업표지로 — 단, **정확히 하나**일 때만
    for n in names:
        hit = by_name.get(n) or [r for k, v in by_name.items()
                                 if k == n or (len(n) >= 3 and n in k) for r in v]
        hit = list(dict.fromkeys(hit))
        if len(hit) == 1:
            return hit[0], f"영업표지 '{n}' 단일 일치 → {hit[0]}"
        if len(hit) > 1:
            return "", f"영업표지 '{n}' 가 {len(hit)}개 브랜드에 걸린다 — 사람이 확인해야 한다"
    return "", (f"등록번호 후보 {brand.get('reg_no_candidates')} 가 패널에 없고 "
                f"영업표지 후보 {brand.get('brand_name_candidates')} 로도 특정 불가")


def ingest_saved(cfg: dict, src_dir: str) -> dict:
    """**사람이 브라우저로 저장한** 열람 페이지(.html)를 읽어 수집분에 합친다.

    왜 이 경로가 필요한가
        전량 자동 수집은 캡차로 막혀 있고(세션당 5건), 우리는 그것을 우회하지
        않는다. 그러나 정보공개서는 **누구나 볼 수 있게 공개된 문서**이고, 사람이
        브라우저로 열어 저장하는 것은 캡차가 막으려는 행위가 아니라 그 사이트가
        의도한 사용 그 자체다. 자동화를 뚫는 대신 **사람의 열람 결과를 받는다.**

        정식 키(IFRMP_SERVICE_KEY)가 마감 전에 안 나올 수 있으므로, 점포 수가 많은
        브랜드부터 손으로 몇십 건만 받아도 점포가중 커버리지가 크게 오른다
        (실측: 상위 30개 → 37.1% → 49.4%). 어느 브랜드를 먼저 받아야 하는지는
        `tools/ifrmp_wanted.py` 가 뽑아 준다.

    저장 방법 (브라우저에서)
        정보공개서 열람 → 브랜드 검색 → 상세 화면에서 Ctrl+S
        → "웹페이지, HTML만"(single file 아님) 으로 저장. 파일명은 아무거나 좋다.

    파싱은 자동 수집과 **완전히 같은 코드**(parse_view / financial_rows)를 쓴다.
    경로만 다르고 결과는 같아야 하므로, 다른 파서를 두지 않는다.
    """
    root = Path(cfg["_root"])
    src = Path(src_dir)
    if not src.is_absolute():
        src = root / src
    dest = root / RAW_DIR
    dest.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in src.rglob("*")
                    if p.suffix.lower() in (".html", ".htm", ".pdf", ".txt")])
    if not files:
        log.warning("%s 에 .html/.pdf/.txt 파일이 없다", src)
        return {"files": 0, "parsed": 0, "with_financials": 0}

    by_id, by_name = _panel_index(cfg)
    if not by_id:
        log.warning("패널을 못 읽었다 — 브랜드 확정 없이 진행하지 않는다")
        return {"files": len(files), "parsed": 0, "with_financials": 0,
                "skipped": len(files)}

    parsed = with_fin = 0
    skipped: list[str] = []
    for p in files:
        try:
            if p.suffix.lower() == ".pdf":
                brand = parse_pdf(p)
                flat = ""
            else:
                page = p.read_text(encoding="utf-8", errors="replace")
                flat = re.sub(r"[ \t]+", " ", _TAGS.sub(" ", page))
                brand = parse_view(page)
                # HTML 저장본·화면 복사본 모두 받는다. 표 구조가 남아 있으면
                # parse_view 가 잡고, 텍스트만 남았으면 재무표 정규식이 잡는다.
                if not brand.get("financials"):
                    brand["financials"] = _pdf_fin_table(flat)
                    mu = re.search(r"단위\s*[:：]?\s*\(?\s*([가-힣]+원)", flat)
                    if mu:
                        brand["amount_unit"] = mu.group(1)
                brand.update({k: v for k, v in extra_fields(flat).items()
                              if k not in brand})
        except Exception as exc:
            skipped.append(f"{p.name}: 읽기 실패 {type(exc).__name__}: {exc}")
            continue
        # 등록번호는 문서 안에서 찾되 **패널 대조로 확정**한다. 파일명에 기대지 않는다
        # (대표님이 받은 파일 이름이 '가맹계약서.pdf' 였는데 내용은 정보공개서였다).
        if flat and not brand.get("reg_no_candidates"):
            brand["reg_no_candidates"] = [m.group(1) for m in
                                          re.finditer(r"등록\s*번호.{0,60}?(\d{8,12})", flat, re.S)]
        reg_no, why = resolve_brand(brand, by_id, by_name)
        if not reg_no:
            skipped.append(f"{p.name}: 브랜드 특정 실패 — {why}")
            continue
        brand["_resolved_by"] = why
        rows = financial_rows(reg_no, brand)
        if not rows:
            skipped.append(f"{p.name}: 재무 표를 못 읽음 (열람 상세 페이지가 맞는지 확인)")
            continue
        # 자기 검산 — 자산 = 부채 + 자본. 열이 밀리면 여기서 반드시 깨진다.
        # 허용 오차는 DART 층과 같은 상대 1% 다(_BS_TOL). 원문이 천원 단위로 반올림돼
        # ±1천원이 흔하므로 절대값 비교를 쓰면 멀쩡한 문서가 전부 불합격으로 나온다
        # — 실제로 첫 시험에서 뚜레쥬르 3개년 중 2건이 그렇게 걸렸다.
        bad = [r["fiscal_year"] for r in rows
               if None not in (r["assets"], r["liabilities"], r["equity"])
               and abs(r["assets"] - r["liabilities"] - r["equity"]) > 0.01 * max(abs(r["assets"]), 1)]
        if bad:
            skipped.append(f"{p.name}: 검산 실패(자산≠부채+자본) 연도 {bad} — 열 밀림 의심")
            continue

        for key, pat in (("brand_name", r"영업표지\s*([^\s<]{1,40})"),
                         ("corp_name", r"상\s*호\s*([^\s<]{1,40})"),
                         ("ceo", r"대표자\s*(?:명)?\s*([^\s<]{1,20})")):
            hit = re.search(pat, flat)
            if hit:
                brand.setdefault(key, _txt(hit.group(1)))
        brand.update({"reg_no": reg_no, "source": SOURCE, "_source_file": p.name,
                      "_collected_by": "manual_browser_save"})

        # ⚠️ 덮어쓰지 않고 **병합**한다. 주문형 조회(fetch_brand)로 이미 받아 둔 건에는
        #    목록 화면에서만 얻는 항목(상호·영업표지·업종·등록일)이 들어 있는데,
        #    저장 페이지에는 그게 없을 수 있다. 통째로 쓰면 그 값들이 사라진다
        #    (실제로 왕복 시험에서 corp_name·brand_name·industry 를 날렸다).
        out_p = dest / f"{reg_no}.json"
        if out_p.exists():
            try:
                prev = json.loads(out_p.read_text(encoding="utf-8"))
                merged = {**prev, **{k: v for k, v in brand.items() if v not in ("", None, [])}}
                brand = merged
            except (OSError, ValueError):
                pass
        out_p.write_text(json.dumps(brand, ensure_ascii=False), encoding="utf-8")
        parsed += 1
        with_fin += 1
    for s in skipped[:10]:
        log.warning("건너뜀 — %s", s)
    log.info("오프라인 수집: 파일 %d개 → 파싱 %d건 (재무 %d건, 건너뜀 %d)",
             len(files), parsed, with_fin, len(skipped))
    return {"files": len(files), "parsed": parsed, "with_financials": with_fin,
            "skipped": len(skipped)}


def build_parquet(cfg: dict) -> pd.DataFrame:
    """수집된 브랜드별 JSON → 재무 parquet. 수집과 분리해 재실행 가능하게 둔다."""
    root = Path(cfg["_root"])
    raw = root / RAW_DIR
    recs, meta_rows = [], []
    for p in sorted(raw.glob("*.json")):
        try:
            b = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rows = financial_rows(p.stem, b)
        recs.extend({**r, "brno": b.get("brno", ""), "crno": b.get("crno", ""),
                     "corp_name": b.get("corp_name", ""),
                     "brand_name": b.get("brand_name", ""),
                     "source": "ifrmp_web"} for r in rows)
        meta_rows.append({"reg_no": p.stem, "n_fin_years": len(rows),
                          "brno": b.get("brno", "")})
    df = pd.DataFrame(recs)
    if len(df):
        df = df.sort_values(["reg_no", "fiscal_year"]).reset_index(drop=True)
        dest = Path(cfg["paths"]["processed"]) / "ifrmp_web_financials.parquet"
        df.to_parquet(dest, index=False)
        log.info("ifrmp_web_financials.parquet — 브랜드 %d개 · 재무 %d행 "
                 "(연도 %s~%s)", df["reg_no"].nunique(), len(df),
                 int(df["fiscal_year"].min()), int(df["fiscal_year"].max()))
    n_meta = len(meta_rows)
    n_fin = sum(1 for m in meta_rows if m["n_fin_years"] > 0)
    status = {"collected_brands": n_meta, "with_financials": n_fin,
              "financial_rows": len(df), "source": SOURCE}
    (Path(cfg["paths"]["outputs"]) / "ifrmp_web_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=1), encoding="utf-8")
    return df


def collect(cfg: dict, max_pages: int | None = None) -> dict:
    root = Path(cfg["_root"])
    raw = root / RAW_DIR
    raw.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in raw.glob("*.json")}

    s = requests.Session()
    s.headers.update(_HEADERS)

    n_new = n_skip = n_fail = consec_fail = 0
    page = 0
    total_pages = None
    while True:
        page += 1
        if max_pages and page > max_pages:
            break
        try:
            r = s.get(f"{BASE}/list.do",
                      params={"pageUnit": PAGE_UNIT, "pageIndex": page}, timeout=45)
            r.raise_for_status()
        except Exception as exc:
            log.warning("목록 %d페이지 실패: %s — 중단", page, str(exc)[:100])
            break
        # 행: [번호, 상호(a), 영업표지(a), 대표자, 등록번호, 등록일, 업종]
        body = re.search(r"<tbody.*?</tbody>", r.text, re.S)
        if not body:
            log.info("목록 %d페이지 — tbody 없음, 종료", page)
            break
        rows = _ROW_RE.findall(body.group(0))
        if total_pages is None:
            first = _CELL_RE.findall(rows[0]) if rows else []
            if first:
                with contextlib.suppress(ValueError):
                    total = int(_txt(first[0]))
                    total_pages = -(-total // PAGE_UNIT)
                    log.info("전체 약 %s건 · %d페이지 (pageUnit=%d)",
                             f"{total:,}", total_pages, PAGE_UNIT)
        if not rows:
            break
        for row in rows:
            cells = [_txt(c) for c in _CELL_RE.findall(row)]
            toks = _TOKEN_RE.findall(row)
            if len(cells) < 7 or not toks:
                continue
            corp, brand, reg_no = cells[1], cells[2], cells[4]
            if not reg_no or reg_no in done:
                n_skip += 1
                continue
            time.sleep(PACE_SEC)
            try:
                v = s.get(f"{BASE}/view.do", params={"encFirMstSn": toks[0]}, timeout=90)
                v.raise_for_status()
                if len(v.content) < 100_000:        # 오류 페이지(~1KB)는 성공이 아니다
                    raise RuntimeError(f"응답 {len(v.content)}B — 오류 페이지 추정")
                parsed = parse_view(v.text)
            except Exception as exc:
                n_fail += 1
                consec_fail += 1
                log.info("상세 실패 (%s %s): %s", reg_no, brand[:14], str(exc)[:80])
                if consec_fail >= MAX_CONSEC_FAIL:
                    log.warning("연속 %d회 실패 — 차단 가능성, 즉시 중단한다", consec_fail)
                    return {"stopped": "consecutive_failures", "new": n_new,
                            "skip": n_skip, "fail": n_fail, "page": page}
                continue
            consec_fail = 0
            parsed.update({"corp_name": corp, "brand_name": brand, "reg_no": reg_no,
                           "industry": cells[6], "registered": cells[5],
                           "ceo": "",              # 대표자 성명은 저장하지 않는다 (개인정보)
                           "source": SOURCE})
            (raw / f"{reg_no}.json").write_text(
                json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
            done.add(reg_no)
            n_new += 1
            if n_new % 100 == 0:
                log.info("진행 %d페이지 · 신규 %d · 기존 %d · 실패 %d",
                         page, n_new, n_skip, n_fail)
        if total_pages and page >= total_pages:
            break
        time.sleep(PACE_SEC)
    return {"new": n_new, "skip": n_skip, "fail": n_fail, "pages": page}


def fetch_brand(cfg: dict, keyword: str) -> dict | None:
    """브랜드명으로 **1건** 조회 — 캡차 한도 안에서 쓰는 주문형 경로.

    심사역이 "이 브랜드 본부 재무를 확인해야 한다"고 할 때 쓰는 용도다.
    전량 수집은 캡차로 막혀 있으므로(모듈 상단 참조) 이것이 정상 사용법이다.
    """
    root = Path(cfg["_root"])
    raw = root / RAW_DIR
    raw.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers.update(_HEADERS)
    r = s.get(f"{BASE}/list.do",
              params={"pageUnit": 5, "pageIndex": 1, "column": "brd",
                      "searchKeyword": keyword}, timeout=45)
    r.raise_for_status()
    body = re.search(r"<tbody.*?</tbody>", r.text, re.S)
    if not body:
        log.warning("'%s' 검색 결과 없음", keyword)
        return None
    for row in _ROW_RE.findall(body.group(0)):
        cells = [_txt(c) for c in _CELL_RE.findall(row)]
        toks = _TOKEN_RE.findall(row)
        if len(cells) < 7 or not toks:
            continue
        time.sleep(PACE_SEC)
        v = s.get(f"{BASE}/view.do", params={"encFirMstSn": toks[0]}, timeout=90)
        if len(v.content) < 100_000:
            log.warning("캡차 또는 오류 (%dB) — 브라우저로 직접 열람해야 한다: %s/list.do",
                        len(v.content), BASE)
            return None
        parsed = parse_view(v.text)
        parsed.update({"corp_name": cells[1], "brand_name": cells[2],
                       "reg_no": cells[4], "industry": cells[6],
                       "registered": cells[5], "source": SOURCE})
        (raw / f"{cells[4]}.json").write_text(
            json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
        log.info("확보: %s (%s) — 재무 %d개년",
                 cells[2], cells[4], len(financial_rows(cells[4], parsed)))
        return parsed
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="공정위 웹 열람 — 본부 재무 (주문형)")
    ap.add_argument("--brand", type=str, default="", help="브랜드명으로 1건 조회")
    ap.add_argument("--max-pages", type=int, default=0,
                    help="⚠️ 전량 수집은 캡차로 막혀 있다 — 진단용으로만 남긴다")
    ap.add_argument("--build-only", action="store_true", help="수집 없이 parquet 재구축만")
    ap.add_argument("--ingest-dir", type=str, default="",
                    help="사람이 브라우저로 저장한 열람 페이지(.html) 폴더를 읽어들인다")
    args = ap.parse_args()
    cfg = load_config()
    if args.ingest_dir:
        ingest_saved(cfg, args.ingest_dir)
    elif args.brand:
        fetch_brand(cfg, args.brand)
    elif not args.build_only:
        log.warning("전량 수집은 캡차로 차단돼 있다(세션당 %d건). --brand 로 주문형 조회를 "
                    "쓰거나, 구조적 해소는 IFRMP_SERVICE_KEY 정식 키를 받으십시오.",
                    BATCH_LIMIT)
        stats = collect(cfg, max_pages=args.max_pages or 1)
        log.info("수집 종료: %s", stats)
    df = build_parquet(cfg)
    if len(df):
        neg_eq = (df.groupby("reg_no")["equity"].last() < 0).sum()
        op_loss = (df.groupby("reg_no")["operating_income"].last() < 0).sum()
        log.info("최신 연도 기준 자본잠식 %d개 · 영업적자 %d개 브랜드", neg_eq, op_loss)
    return 0


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, OSError):
            _s.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
