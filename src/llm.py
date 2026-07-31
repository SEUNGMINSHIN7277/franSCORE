"""공용 LLM 클라이언트 — Google Gemini (Generative Language API v1beta).

INTERFACES.md §4 의 LLM 호출 계약을 이 한 곳에 모은다. news_llm(구조화 추출)과
memo_llm(심사메모 생성)은 모두 이 모듈만 통해 모델을 호출한다.

왜 공식 SDK가 아니라 얇은 REST 클라이언트인가:
  1. 이 저장소는 이미 requests 기반 재시도 계층(src/collect.py)을 쓴다 — 실패·백오프
     정책을 한 가지 방식으로 통일해 리뷰 지점을 줄인다.
  2. 제출본은 고정(freeze)되어야 한다. SDK 마이너 버전이 응답 객체 형태를 바꾸면
     재현이 깨지므로, 변하지 않는 HTTP 계약(v1beta)에 직접 붙는다.
  3. 사고(thought) 파트 필터링·finishReason 분기·안전정책 차단 같은 **실패 경로를
     명시적으로** 다뤄야 한다. 폴백 판단 근거가 코드에 드러나야 감사 가능하다.
  의존성도 늘지 않는다(requests는 이미 필수).

API 계약 (2026-07 실호출 검증):
    POST {api_base}/models/{model}:generateContent
    헤더  X-goog-api-key: <API 키>, Content-Type: application/json
    본문  {systemInstruction, contents, generationConfig, safetySettings}
    응답  {candidates:[{content:{parts:[{text|thought|thoughtSignature}]},finishReason}],
           promptFeedback?, usageMetadata}

정직성 원칙:
  · 키는 환경변수(config `llm.api_key_env`)에서만 읽고 로그·스냅샷에 남기지 않는다.
  · 호출 실패·정책 차단은 삼키지 않고 예외로 올린다. 폴백 여부는 **호출자**가 정하고
    산출물에 `llm_used=false` 로 표기한다 (LLM을 썼다고 거짓 표기하지 않기 위해).
"""
from __future__ import annotations

import json
import os
import time

import requests

from src.common import get_logger

log = get_logger("llm")

# 안전정책상 차단된 응답 — 같은 입력으로 재시도해도 동일하므로 즉시 폴백시킨다.
_BLOCKED_FINISH = {
    "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY",
}
# 일시적 장애로 보고 지수 백오프 재시도할 HTTP 상태
_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}

# responseSchema 는 OpenAPI 3.0 스키마의 **부분집합**만 받는다.
# 아래 키워드는 JSON Schema 방언이라 그대로 보내면 400이 나므로 재귀적으로 제거한다.
_SCHEMA_DROP = {
    "additionalProperties", "$schema", "$id", "$defs", "definitions",
    "patternProperties", "unevaluatedProperties",
}

# 안전 필터 임계값.
#   이 도구의 입력은 '가맹점주 소송', '본사 부도', '집단 폐점' 같은 **부정적 기업 뉴스**다.
#   기본 임계값은 이런 텍스트를 유해로 오탐해 분류가 통째로 차단되는 일이 있어,
#   고위험만 차단(BLOCK_ONLY_HIGH)하도록 낮춘다. 차단이 나더라도 예외로 올려
#   규칙기반 폴백으로 안전하게 내려앉으므로 무방비가 아니다.
_SAFETY_SETTINGS = [
    {"category": c, "threshold": "BLOCK_ONLY_HIGH"}
    for c in (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
]

_DEFAULT_BASE = "https://generativelanguage.googleapis.com/v1beta"


class LLMError(RuntimeError):
    """LLM 호출 실패 — 호출자는 폴백 경로로 전환한다."""


class LLMFatal(LLMError):
    """재시도해도 결과가 같은 실패 (설정 오류·토큰 절단 등) — 백오프 없이 즉시 폴백.

    일시적 장애(네트워크·429·5xx)와 구분해야 무의미한 재시도로 시간·쿼터를 낭비하지 않는다.
    """


class LLMRefused(LLMFatal):
    """모델이 안전정책으로 응답을 거절/차단 — 재시도해도 동일하므로 즉시 폴백."""


# ---------------------------------------------------------------------------
# 설정·키
# ---------------------------------------------------------------------------

def api_key_env(cfg: dict) -> str:
    return str(cfg.get("llm", {}).get("api_key_env", "GEMINI_API_KEY"))


def api_key(cfg: dict) -> str:
    """환경변수에서 API 키를 읽는다 (없으면 빈 문자열)."""
    return os.environ.get(api_key_env(cfg), "").strip()


def is_enabled(cfg: dict) -> bool:
    """LLM 경로를 쓸 수 있는지 — 키 존재 여부만 본다(호출 가능 여부는 호출 시 판정)."""
    return bool(api_key(cfg))


def model_name(cfg: dict) -> str:
    return str(cfg["llm"]["model"])


def _mask(text: str, key: str) -> str:
    """예외 메시지 등에 키가 섞여 들어갈 가능성을 차단."""
    return text.replace(key, "***KEY***") if key else text


# ---------------------------------------------------------------------------
# 스키마 변환
# ---------------------------------------------------------------------------

def to_gemini_schema(schema: dict) -> dict:
    """JSON Schema → Gemini responseSchema (지원 부분집합).

    - 미지원 키워드 제거 (`additionalProperties` 등)
    - object 에 `propertyOrdering` 부여 → 필드 순서가 호출마다 흔들리지 않게 고정
      (Gemini 공식 권고. 구조화 출력 안정성이 눈에 띄게 좋아진다.)
    """
    if not isinstance(schema, dict):
        return schema
    out: dict = {}
    for k, v in schema.items():
        if k in _SCHEMA_DROP:
            continue
        if k == "properties" and isinstance(v, dict):
            out[k] = {pk: to_gemini_schema(pv) for pk, pv in v.items()}
        elif k == "items":
            out[k] = to_gemini_schema(v) if isinstance(v, dict) else v
        elif k in ("anyOf", "oneOf", "allOf") and isinstance(v, list):
            out[k] = [to_gemini_schema(x) for x in v]
        else:
            out[k] = v
    if out.get("type") == "object" and isinstance(out.get("properties"), dict):
        out.setdefault("propertyOrdering", list(out["properties"].keys()))
    return out


# ---------------------------------------------------------------------------
# 호출
# ---------------------------------------------------------------------------

def _http_post(url: str, headers: dict, body: dict, timeout: float):
    """HTTP 경계 — 테스트는 이 함수 하나만 대체해 전체 LLM 코드 경로를 실행한다."""
    return requests.post(url, headers=headers, json=body, timeout=timeout)


def _candidate_text(cand: dict) -> str:
    """후보 응답에서 사용자에게 줄 텍스트만 추출.

    ⚠️ 사고(thinking) 모델은 parts 에 `thought: true` 파트를 섞어 보낸다. 이를 걸러내지
       않으면 메모에 모델의 내부 추론이 그대로 노출된다.
    """
    parts = (cand.get("content") or {}).get("parts") or []
    return "".join(
        str(p.get("text", "")) for p in parts
        if isinstance(p, dict) and not p.get("thought") and "text" in p
    )


def generate(cfg: dict, *, system: str, user: str,
             schema: dict | None = None, max_tokens: int | None = None) -> tuple[str, dict]:
    """Gemini 1회 호출 → (텍스트, 메타).

    Args:
        system: 시스템 지시문(systemInstruction) — 근거 제한·인젝션 방어 규칙.
        user:   사용자 메시지 본문.
        schema: JSON Schema. 주면 responseMimeType=application/json + responseSchema 로
                **구조화 출력**을 강제한다(모델이 자유 텍스트를 섞을 수 없음).
        max_tokens: 미지정 시 config `llm.max_tokens`.

    Returns:
        (text, meta) — meta = {model, finish_reason, usage, attempts}

    Raises:
        LLMRefused: 안전정책 차단(프롬프트 또는 응답).
        LLMError:   키 미설정, HTTP 오류, 응답 비어있음, 토큰 초과 절단 등.
    """
    lcfg = cfg["llm"]
    key = api_key(cfg)
    if not key:
        raise LLMError(f"환경변수 {api_key_env(cfg)} 미설정")

    model = model_name(cfg)
    base = str(lcfg.get("api_base") or _DEFAULT_BASE).rstrip("/")
    url = f"{base}/models/{model}:generateContent"

    gen_cfg: dict = {"maxOutputTokens": int(max_tokens or lcfg.get("max_tokens", 8000))}
    if lcfg.get("temperature") is not None:
        gen_cfg["temperature"] = float(lcfg["temperature"])
    if lcfg.get("thinking"):  # (선택) 모델별 사고 예산 파라미터를 그대로 전달
        gen_cfg["thinkingConfig"] = dict(lcfg["thinking"])
    if schema is not None:
        gen_cfg["responseMimeType"] = "application/json"
        gen_cfg["responseSchema"] = to_gemini_schema(schema)

    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": gen_cfg,
        "safetySettings": _SAFETY_SETTINGS,
    }
    headers = {"Content-Type": "application/json", "X-goog-api-key": key}
    timeout = float(lcfg.get("timeout_sec", 90))
    max_retries = max(1, int(lcfg.get("max_retries", 4)))
    backoff = float(lcfg.get("retry_backoff_sec", 2.0))

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = _http_post(url, headers, body, timeout)
            status = int(getattr(resp, "status_code", 0))
            if status != 200:
                text = _mask(str(getattr(resp, "text", ""))[:500], key)
                if status in _RETRY_STATUS:
                    raise LLMError(f"HTTP {status} (일시적): {text}")
                # 400/403 등 설정·권한 오류는 재시도해도 같다 — 즉시 실패시켜 원인을 드러낸다.
                raise LLMFatal(f"HTTP {status} (재시도 무의미): {text}")

            data = resp.json()
            block = (data.get("promptFeedback") or {}).get("blockReason")
            if block:
                raise LLMRefused(f"프롬프트 차단(blockReason={block})")

            cands = data.get("candidates") or []
            if not cands:
                raise LLMError("응답에 candidates 없음")
            cand = cands[0]
            finish = str(cand.get("finishReason") or "")
            if finish in _BLOCKED_FINISH:
                raise LLMRefused(f"응답 차단(finishReason={finish})")

            text = _candidate_text(cand)
            if finish == "MAX_TOKENS":
                # 잘린 응답은 JSON이든 메모든 신뢰할 수 없다 → 폴백이 정직하다.
                # 같은 입력·같은 예산이면 다시 잘리므로 재시도하지 않는다(LLMFatal).
                raise LLMFatal(
                    f"토큰 한도 초과로 응답 절단(maxOutputTokens={gen_cfg['maxOutputTokens']}, "
                    f"수신 {len(text)}자). config llm.max_tokens 를 올리거나 입력을 줄이세요."
                )
            if not text.strip():
                raise LLMError(f"응답 텍스트 비어있음(finishReason={finish})")

            meta = {
                "model": str(data.get("modelVersion") or model),
                "finish_reason": finish,
                "usage": data.get("usageMetadata") or {},
                "attempts": attempt,
            }
            log.info(
                "LLM 호출 성공: model=%s finish=%s 입력%s/출력%s토큰 시도%d회",
                meta["model"], finish or "-",
                meta["usage"].get("promptTokenCount", "?"),
                meta["usage"].get("candidatesTokenCount", "?"), attempt,
            )
            return text, meta

        except LLMFatal:
            raise  # 재시도 무의미(차단·설정오류·절단) — 그대로 올려 호출자가 폴백하게 한다
        except Exception as exc:
            last_err = exc
            if attempt >= max_retries:
                break
            wait = backoff * (2 ** (attempt - 1))
            log.warning(
                "LLM 호출 실패 (%d/%d): %s → %.1fs 후 재시도",
                attempt, max_retries, _mask(str(exc)[:300], key), wait,
            )
            time.sleep(wait)

    raise LLMError(f"LLM 호출 최종 실패 (model={model}): {_mask(str(last_err)[:300], key)}") from last_err


def generate_json(cfg: dict, *, system: str, user: str, schema: dict,
                  max_tokens: int | None = None) -> tuple[dict, dict]:
    """generate() + JSON 파싱. 스키마 강제 출력이라도 파싱 실패는 예외로 올린다."""
    text, meta = generate(cfg, system=system, user=user, schema=schema, max_tokens=max_tokens)
    try:
        data = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise LLMError(f"구조화 응답 JSON 파싱 실패: {str(text)[:200]!r}") from exc
    if not isinstance(data, dict):
        raise LLMError(f"구조화 응답이 객체가 아님: {type(data).__name__}")
    return data, meta
