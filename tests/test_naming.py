"""명칭 규약 회귀 방지 — 산출물을 PD(부도확률)라고 부르지 않는다.

왜 테스트로 두는가 (실제로 있던 모순)
    두 문서가 정반대를 말하고 있었다.

        docs/MODEL_USE_SPEC.md  "'부도확률'·'PD'로 표기하거나 그렇게 인용" → **금지 용도**
        docs/MODEL_USE_SPEC.md  "어떤 산출물도 PD 가 아니다."
        docs/OPERATIONS.md      "PD로 쓰되 신용 부도율과는 다르다."   ← 정면 충돌

    그리고 공개 산출물 `outputs/scores_latest.csv` 의 컬럼명이 `pd_1y`·`pd_raw`·
    `pd_calibrated_step`·`pd_rank_pct` 였다. 은행 분석가가 이 CSV 를 열면 명세가
    금지한 바로 그 오독을 한다 — 그것도 우리가 이름으로 유도한 오독이다.

    라벨은 부도가 아니라 **공시 기반 구조악화 전환**이다. 실제 여신 부도·연체
    라벨을 한 건도 보지 못했으므로 PD 라고 부를 근거가 없다. 이름은 주장이다.

무엇을 막는가
    A. 옛 `pd_*` 식별자의 부활 (컬럼명·변수명·설정키·문서 표기 전부)
    B. 산출물을 PD 라고 부르는 문장 (EL 공식 포함)

무엇을 막지 않는가
    바젤 IRB 의 PD 개념을 **참조**하는 것은 정당하다 — 확률 하한 관행을 준용한
    근거를 밝히거나, 규제자본 산출에 쓸 수 없는 이유를 설명할 때가 그렇다.
    금지 대상은 '우리 숫자가 PD 다'라는 주장뿐이다.

실행: python tests/test_naming.py
"""
from __future__ import annotations

import contextlib
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

for _s in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, OSError):
        _s.reconfigure(encoding="utf-8", errors="replace")

EXTS = {".py", ".md", ".yaml", ".yml"}
SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", "data", "outputs"}
SKIP_FILES = {"test_naming.py"}          # 이 파일은 금지 표현을 예시로 담고 있다

# A. 되살아나면 안 되는 옛 식별자
OLD_TOKENS = re.compile(
    r"\bpd_(1y|raw|calibrated_step|rank_pct|min|max|floor|cap|col|column|"
    r"component|cut|definition|multiplier|pctile|stressed|sum|within_calibrator_range)\b")

# B. 산출물을 PD 라고 부르는 문장. 개념 참조와 구별되도록 **주장 형태**만 잡는다.
CLAIM_PATTERNS = [
    (re.compile(r"PD\s*로\s*쓰"),               "'PD로 쓰' — 산출물을 PD로 사용하라는 지시"),
    (re.compile(r"EL\s*=\s*exposure\s*×\s*PD"), "EL 공식에서 우리 확률을 PD 라고 표기"),
    (re.compile(r"EL_i\s*=\s*exposure_i\s*×\s*PD_i"), "EL 공식에서 우리 확률을 PD 라고 표기"),
    (re.compile(r"스트레스\s*PD"),               "'스트레스 PD' — 우리 확률에 PD 명칭 사용"),
    (re.compile(r"PD\s*대용"),                   "'PD 대용' — 대용이라도 PD 명칭을 앞세운다"),
    (re.compile(r"PD\s*컬럼"),                   "'PD 컬럼' — 산출물 컬럼을 PD 로 지칭"),
    (re.compile(r"PD\s*상[·・]?\s*하한\s*적용"),  "'PD 상·하한 적용' — 우리 확률에 PD 명칭 사용"),
]

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    RESULTS.append((ok, name, detail))


def _files():
    for p in _ROOT.rglob("*"):
        if not p.is_file() or p.suffix not in EXTS:
            continue
        if any(part in SKIP_DIRS for part in p.parts) or p.name in SKIP_FILES:
            continue
        yield p


def test_no_old_pd_tokens() -> None:
    hits = []
    for p in _files():
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for m in OLD_TOKENS.finditer(line):
                hits.append(f"{p.relative_to(_ROOT)}:{i} {m.group(0)}")
    check(not hits, "옛 pd_* 식별자 없음", f"{len(hits)}건" + (f" 예: {hits[0]}" if hits else ""))
    for h in hits[:10]:
        print(f"        {h}")


def test_no_pd_claim() -> None:
    hits = []
    for p in _files():
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for pat, why in CLAIM_PATTERNS:
                if pat.search(line):
                    hits.append(f"{p.relative_to(_ROOT)}:{i} — {why}")
    check(not hits, "산출물을 PD 라 부르는 문장 없음",
          f"{len(hits)}건" + (f" 예: {hits[0]}" if hits else ""))
    for h in hits[:10]:
        print(f"        {h}")


def test_published_columns() -> None:
    """공개 산출물 컬럼명이 실제로 바뀌었는지 — 코드만 고치고 CSV 를 안 돌리면 무의미하다."""
    p = _ROOT / "outputs" / "scores_latest.csv"
    if not p.exists():
        check(True, "산출물 컬럼명", "scores_latest.csv 없음 — 건너뜀")
        return
    header = p.read_text(encoding="utf-8-sig").splitlines()[0]
    cols = [c.strip() for c in header.split(",")]
    bad = [c for c in cols if c.startswith("pd_")]
    need = {"deterioration_1y", "deterioration_step", "deterioration_rank_pct", "score_raw"}
    missing = sorted(need - set(cols))
    check(not bad and not missing, "산출물 컬럼명이 새 규약",
          f"잔존 {bad} · 누락 {missing}" if (bad or missing) else f"{len(cols)}개 컬럼 확인")


def test_spec_consistency() -> None:
    """사용 명세와 운영 문서가 같은 말을 하는가 — 이 둘이 어긋난 것이 사건의 시작이었다."""
    spec = (_ROOT / "docs" / "MODEL_USE_SPEC.md")
    ops = (_ROOT / "docs" / "OPERATIONS.md")
    if not (spec.exists() and ops.exists()):
        check(True, "명세·운영 문서 정합", "문서 없음 — 건너뜀")
        return
    s, o = spec.read_text(encoding="utf-8"), ops.read_text(encoding="utf-8")
    spec_forbids = "PD" in s and "가 아니다" in s
    ops_allows = bool(re.search(r"PD\s*로\s*(쓰|사용)", o))
    check(spec_forbids and not ops_allows, "명세·운영 문서가 같은 기준",
          "운영 문서가 PD 사용을 허용" if ops_allows else "두 문서 모두 PD 표기 금지")


def main() -> int:
    for fn in (test_no_old_pd_tokens, test_no_pd_claim,
               test_published_columns, test_spec_consistency):
        try:
            fn()
        except Exception as exc:
            check(False, fn.__name__, f"예외: {exc}")
    fails = [r for r in RESULTS if not r[0]]
    for ok, name, detail in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:34s} {detail}")
    print(f"=== {len(RESULTS) - len(fails)}/{len(RESULTS)} passed ===")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
