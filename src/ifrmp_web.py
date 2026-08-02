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
    파일럿을 돌려 보니 **한 세션에서 5건을 열람한 직후 캡차가 걸린다.**
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
    try:
        return float(t)
    except ValueError:
        return None


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


_FIN_COLS = {"자산": "assets", "부채": "liabilities", "자본": "equity",
             "매출액": "revenue", "영업이익": "operating_income",
             "당기순이익": "net_income"}


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
    if len(idx) < len(_FIN_COLS):
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

    files = sorted([p for p in src.rglob("*") if p.suffix.lower() in (".html", ".htm")])
    if not files:
        log.warning("%s 에 .html 파일이 없다", src)
        return {"files": 0, "parsed": 0, "with_financials": 0}

    parsed = with_fin = 0
    skipped: list[str] = []
    for p in files:
        try:
            page = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            skipped.append(f"{p.name}: 읽기 실패 {exc}")
            continue
        brand = parse_view(page)
        flat = _TAGS.sub(" ", page)
        # 등록번호를 페이지에서 찾는다 — 파일명에 기대지 않는다(사람이 붙인 이름이다)
        m = re.search(r"등록번호.{0,200}?(\d{8,12})", flat, re.S)
        reg_no = m.group(1) if m else p.stem
        rows = financial_rows(reg_no, brand)
        if not rows:
            skipped.append(f"{p.name}: 재무 표를 못 읽음 (열람 상세 페이지가 맞는지 확인)")
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
