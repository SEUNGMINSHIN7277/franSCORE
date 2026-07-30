"""API 키 진단 도구 — 키 발급 직후 실행해 바로 쓸 수 있는지 확인한다.

사용법 (프로젝트 루트에서):
    python check_keys.py            # 둘 다 점검
    python check_keys.py --data     # data.go.kr 키만
    python check_keys.py --llm      # Anthropic 키만

무엇을 알려주나:
  · 키가 설정돼 있는지, 어떤 형태인지(Encoding/Decoding 판별)
  · 6개 데이터셋 각각에 활용신청이 승인됐는지 (하나씩 실제 호출)
  · 실패 시 원인을 구분해서 안내 (미등록 / 아직 미활성 / 이중 인코딩 / 트래픽 초과)
  · Anthropic 키가 실제로 호출 가능한지 (최소 토큰 1회 호출)
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.parse

import requests

from src.collect import SERVICES
from src.common import load_config

DATASET_IDS = {
    "brand_frcs_stats": "15110241",
    "brand_overview": "15109828",
    "industry_openclose": "15157660",
    "brand_master": "15125467",
    "brand_region_direct": "15125490",
    "brand_cancel": "15125518",
}

OK, NG, WARN = "  [OK]  ", "  [실패]", "  [주의]"


def _classify(body: dict | None, raw_text: str, status: int) -> tuple[bool, str]:
    """응답 → (성공여부, 사람이 읽을 원인 설명)."""
    txt = (raw_text or "")[:400]
    if status == 401 or "Unauthorized" in txt:
        # apis.data.go.kr 게이트웨이는 잘못되거나 아직 동기화 안 된 키에 XML이 아니라 401을 준다
        return False, ("인증 실패(401). 키 오타이거나 **활용신청 후 동기화 대기 중**입니다. "
                       "포털 게이트웨이는 보통 10~30분 뒤 반영됩니다. 재발급 금지.")
    if body is None:
        if "SERVICE_KEY_IS_NOT_REGISTERED" in txt:
            return False, ("키 미등록(코드 30). 활용신청 직후라면 **인증키 동기화 대기**일 수 "
                           "있습니다(포털 게이트웨이 10~30분). 키를 재발급하지 마세요 — "
                           "재발급하면 기존 키가 폐기되고 동기화가 처음부터 다시 시작됩니다.")
        if "SERVICE_ACCESS_DENIED" in txt:
            return False, ("코드 20 — 이 데이터셋에 **활용신청을 하지 않았습니다**. "
                           "키는 계정당 1개로 모든 API 공용이지만, 데이터셋별 활용신청은 따로 해야 합니다.")
        if "LIMITED_NUMBER_OF_SERVICE_REQUESTS" in txt:
            return False, "일일 트래픽 초과(개발계정 10,000건/일). 내일 다시 시도하세요."
        if "SERVICE_ACCESS_DENIED" in txt:
            return False, "접근 거부. 활용신청 승인 상태를 확인하세요."
        return False, f"JSON 파싱 실패(응답 앞부분: {txt[:120]!r})"
    code = str(body.get("resultCode", body.get("response", {}).get("header", {}).get("resultCode", "")))
    msg = str(body.get("resultMsg", ""))
    if code in ("00", "0"):
        return True, "정상"
    if code == "30" or "NOT_REGISTERED" in msg:
        return False, "활용신청 미승인 또는 키 미등록(resultCode=30)."
    if code == "22":
        return False, "일일 트래픽 초과(resultCode=22)."
    if code in ("10", "11"):
        return False, f"파라미터 오류(resultCode={code}) — 코드 문제일 수 있으니 알려주세요."
    return False, f"resultCode={code} msg={msg}"


def check_data_key(cfg: dict) -> bool:
    print("=" * 72)
    print("data.go.kr (DATA_GO_KR_KEY) 점검")
    print("=" * 72)
    key = os.environ.get("DATA_GO_KR_KEY", "").strip()
    if not key:
        print(f"{NG} 환경변수 DATA_GO_KR_KEY 가 설정돼 있지 않습니다.")
        print("       PowerShell 영구 설정:")
        print('         [Environment]::SetEnvironmentVariable("DATA_GO_KR_KEY","<키>","User")')
        print("       (설정 후 터미널을 새로 열어야 반영됩니다)")
        return False

    print(f"{OK} 키 감지 (길이 {len(key)})")
    # 키 형식 판별.
    #  · 2025-08-21 이후 발급분: 특수문자 없는 영문·숫자 → 인코딩/디코딩 구분이 폐지됨.
    #    그대로 쓰면 되고, requests의 자동 인코딩도 무해(변환할 문자가 없음).
    #  · 그 이전 발급분(레거시): 원본 base64라 + / = 포함 → 이 프로젝트는 requests가
    #    자동 인코딩하므로 **Decoding(원본) 키**를 써야 한다.
    if "%" in key and not any(c in key for c in "+/="):
        print(f"{WARN} **이미 URL 인코딩된 키(Encoding)로 보입니다** (% 포함).")
        print("       이 프로젝트는 requests가 자동 인코딩하므로 이중 인코딩이 되어 실패합니다.")
        print("       레거시 키라면 'Decoding(원본)' 값을 쓰세요.")
    elif any(c in key for c in "+/="):
        print(f"{OK} 레거시 Decoding(원본) 키 형태 (+ / = 포함) — 이 프로젝트에 맞습니다.")
    elif key.isalnum():
        print(f"{OK} 신형 키 형태 (영문·숫자만) — 2025-08-21 이후 발급분.")
        print("       인코딩/디코딩 구분이 폐지되어 그대로 사용하면 됩니다.")
    else:
        print(f"{WARN} 형식을 단정할 수 없습니다. 실패하면 원본(Decoding) 값으로 바꿔보세요.")

    print()
    print("데이터셋별 활용신청 승인 여부 (각 1건씩 실제 호출):")
    all_ok = True
    for name, spec in SERVICES.items():
        ds = DATASET_IDS.get(name, "?")
        params = {"serviceKey": key, "pageNo": 1, "numOfRows": 1,
                  "resultType": "json", spec["year_param"]: 2023}
        try:
            r = requests.get(spec["url"], params=params, timeout=20)
            try:
                body = r.json()
            except ValueError:
                body = None
            ok, why = _classify(body, r.text, r.status_code)
        except requests.RequestException as e:
            ok, why = False, f"네트워크 오류: {type(e).__name__}"
        total = ""
        if ok and isinstance(body, dict):
            total = f" (totalCount={body.get('totalCount', '?')})"
        print(f"{OK if ok else NG} {ds}  {name:22s}{total}")
        if not ok:
            print(f"           → {why}")
            print(f"           → 신청 페이지: https://www.data.go.kr/data/{ds}/openapi.do")
            all_ok = False

    print()
    if all_ok:
        print("  ✅ 6개 데이터셋 전부 사용 가능 — `python run_pipeline.py --step collect` 실행하면")
        print("     본인 키로 원본 스냅샷을 재수집합니다.")
    else:
        print("  ⚠️ 일부 데이터셋이 아직 사용 불가합니다. 위 신청 페이지에서 활용신청 후")
        print("     (자동승인이라도 반영에 시간이 걸릴 수 있음) 다시 실행하세요.")
    return all_ok


def check_llm_key(cfg: dict) -> bool:
    print()
    print("=" * 72)
    print("Anthropic (ANTHROPIC_API_KEY) 점검")
    print("=" * 72)
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        print(f"{NG} 환경변수 ANTHROPIC_API_KEY 가 설정돼 있지 않습니다.")
        print("       PowerShell 영구 설정:")
        print('         [Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY","<키>","User")')
        print("       미설정 상태에서도 파이프라인은 규칙 폴백으로 정상 동작합니다.")
        return False
    masked = key[:12] + "..." + key[-4:] if len(key) > 20 else "(짧음)"
    print(f"{OK} 키 감지: {masked}")
    if not key.startswith("sk-ant-"):
        print(f"{WARN} 일반적인 Anthropic 키는 'sk-ant-' 로 시작합니다. 형식을 확인하세요.")

    model = cfg["llm"]["model"]
    try:
        import anthropic
    except ImportError:
        print(f"{NG} anthropic 패키지 미설치 — pip install -r requirements.txt")
        return False
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model, max_tokens=16,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        print(f"{OK} 호출 성공 (model={model}, 응답={text!r})")
        u = resp.usage
        print(f"       토큰: 입력 {u.input_tokens} / 출력 {u.output_tokens}")
        print()
        print("  ✅ LLM 층 활성화 가능 — 다음 실행 시 실제 Claude가 사용됩니다:")
        print("     python run_pipeline.py --step news     (뉴스 신호 구조화 추출)")
        print("     streamlit run src/app.py               (심사메모 실제 생성)")
        return True
    except Exception as e:  # noqa: BLE001
        name = type(e).__name__
        print(f"{NG} 호출 실패: {name}")
        hint = {
            "AuthenticationError": "키가 유효하지 않습니다. Console에서 다시 발급하세요.",
            "PermissionDeniedError": "이 키로 해당 모델에 접근 권한이 없습니다.",
            "NotFoundError": f"모델 ID '{model}' 를 찾을 수 없습니다. config.yaml의 llm.model 확인.",
            "RateLimitError": "요청 한도 초과. 잠시 후 재시도하세요.",
            "APIConnectionError": "네트워크 연결 실패.",
        }.get(name)
        if hint:
            print(f"       → {hint}")
        else:
            print(f"       → {str(e)[:300]}")
        if "credit" in str(e).lower() or "billing" in str(e).lower():
            print("       → **크레딧 부족**일 수 있습니다. Console > Billing 에서 크레딧을 충전하세요.")
            print("          (Claude Pro/Max 구독은 API 크레딧에 포함되지 않습니다)")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="FranSCORE API 키 진단")
    ap.add_argument("--data", action="store_true", help="data.go.kr 키만 점검")
    ap.add_argument("--llm", action="store_true", help="Anthropic 키만 점검")
    args = ap.parse_args()
    both = not (args.data or args.llm)
    cfg = load_config()
    ok = True
    if both or args.data:
        ok &= check_data_key(cfg)
    if both or args.llm:
        ok &= check_llm_key(cfg)
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
