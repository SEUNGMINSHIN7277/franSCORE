"""M1 수집: 공정위 가맹정보 오픈API → data/raw/ 원본 스냅샷.

데이터 소스 (조사로 확정, 2026-07-30):
- brand_frcs_stats  : 15110241 FftcBrandFrcsStatsService/getBrandFrcsStats
                      브랜드×연도 가맹점수·신규등록·계약종료/해지·명의변경·평균매출 (패널 본체)
- brand_overview    : 15109828 FftcBrandBrandStatsService/getBrandBrandStats
                      가맹사업개시일·임원수·직원수 (라벨 차원 밖 보조 피처)
- industry_openclose: 15157660 FftcIndutyFrcsCntOpclStatsService/getIndutyFrcsCntOpclStats
                      업종×연도 개폐점률 (업종 기준선 전용)

커버리지: yr 2017~2025 (yr=정보공개서 기준연도, 수치는 통상 직전 회계연도 실적).
스냅샷 원칙: 원본 JSON을 data/raw/에 그대로 보존 (재현성·정직성), 존재 시 재호출 생략.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.common import get_logger, load_config

log = get_logger("collect")

SERVICES: dict[str, dict] = {
    "brand_frcs_stats": {
        "url": "https://apis.data.go.kr/1130000/FftcBrandFrcsStatsService/getBrandFrcsStats",
        "year_param": "yr",
    },
    "brand_overview": {
        "url": "https://apis.data.go.kr/1130000/FftcBrandBrandStatsService/getBrandBrandStats",
        "year_param": "yr",
    },
    "industry_openclose": {
        "url": "https://apis.data.go.kr/1130000/FftcIndutyFrcsCntOpclStatsService/getIndutyFrcsCntOpclStats",
        "year_param": "jngBizCrtrYr",
    },
}

YEARS = list(range(2017, 2026))

# 공정위 가맹사업정보제공시스템(franchise.ftc.go.kr)의 공개 미리보기 페이지
# (https://franchise.ftc.go.kr/openApi.do?service=FftcBrandFrcsStatsService)에
# 내장·공개되어 있는 데모 인증키. 비밀키가 아니라 공정위가 공개 배포한 값이지만
# 회전될 수 있으므로, 제출/운영 파이프라인은 반드시 본인 data.go.kr 키(DATA_GO_KR_KEY)를 사용할 것.
_PUBLIC_PREVIEW_KEY = (
    "B7y6mnXwYiqT8J24zK9IAyF7EGDjLktqZMwR/0h8yg6Tq6Scw/j5naZkQIInZ6b2iM+6Ndvk/FrG4qoTK3gK1Q=="
)


def get_service_key(cfg: dict) -> str:
    key = os.environ.get(cfg["collect"]["service_key_env"], "").strip()
    if key:
        log.info("serviceKey: 환경변수 %s 사용", cfg["collect"]["service_key_env"])
        return key
    log.warning(
        "serviceKey 미설정 — 공정위 공개 미리보기 데모키로 대체합니다. "
        "제출 전 data.go.kr에서 본인 키를 발급받아 %s 에 설정하세요 (개발단계 자동승인).",
        cfg["collect"]["service_key_env"],
    )
    return _PUBLIC_PREVIEW_KEY


def _fetch_page(url: str, params: dict, cfg: dict) -> dict:
    """단일 페이지 호출 (재시도·백오프 포함). 응답 body(dict)를 반환."""
    c = cfg["collect"]
    last_err: Exception | None = None
    for attempt in range(1, c["max_retries"] + 1):
        try:
            r = requests.get(url, params=params, timeout=c["timeout_sec"])
            if r.status_code == 429:
                raise requests.HTTPError("429 Too Many Requests")
            r.raise_for_status()
            body = r.json()
            # 응답 형태: 평면형 {resultCode,...,items:[...]} 또는 중첩형 {response:{header,body}}
            if "response" in body:
                header = body["response"].get("header", {})
                inner = body["response"].get("body", {})
                code = header.get("resultCode", inner.get("resultCode"))
                merged = {**header, **inner}
            else:
                merged = body
                code = body.get("resultCode")
            if str(code) not in ("00", "0"):
                raise RuntimeError(f"API resultCode={code} msg={merged.get('resultMsg')}")
            return merged
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = c["retry_backoff_sec"] * (2 ** (attempt - 1))
            log.warning("호출 실패 (%d/%d): %s → %.1fs 후 재시도", attempt, c["max_retries"], e, wait)
            time.sleep(wait)
    raise RuntimeError(f"API 호출 최종 실패: {url} params(연도)={params.get('yr') or params.get('jngBizCrtrYr')}") from last_err


def _extract_items(body: dict) -> list[dict]:
    items = body.get("items", [])
    if isinstance(items, dict):  # XML 스타일 {items: {item: [...]}}
        items = items.get("item", [])
    if isinstance(items, dict):  # 단건
        items = [items]
    return items or []


def fetch_service_year(service: str, year: int, key: str, cfg: dict) -> dict:
    """서비스×연도 전체 페이지 수집 → {"totalCount", "items"} 반환."""
    spec = SERVICES[service]
    page_size = 3000
    all_items: list[dict] = []
    page = 1
    total = None
    while True:
        params = {
            "serviceKey": key,
            "pageNo": page,
            "numOfRows": page_size,
            "resultType": "json",
            spec["year_param"]: year,
        }
        body = _fetch_page(spec["url"], params, cfg)
        total = int(body.get("totalCount", 0))
        items = _extract_items(body)
        all_items.extend(items)
        log.info("%s yr=%d page=%d: +%d건 (누적 %d / 총 %d)", service, year, page, len(items), len(all_items), total)
        if len(all_items) >= total or not items:
            break
        page += 1
        time.sleep(0.3)
    return {"totalCount": total, "items": all_items}


def collect_all(cfg: dict, services: list[str] | None = None, years: list[int] | None = None) -> dict[str, list[Path]]:
    """전 서비스×연도 수집. 스냅샷 존재 시 생략(강제 재수집은 파일 삭제 후 실행)."""
    key = get_service_key(cfg)
    raw_dir: Path = cfg["paths"]["raw"]
    out: dict[str, list[Path]] = {}
    for service in services or list(SERVICES):
        paths: list[Path] = []
        for year in years or YEARS:
            snap = raw_dir / f"{service}_{year}.json"
            if snap.exists():
                log.info("스냅샷 존재 → 생략: %s", snap.name)
                paths.append(snap)
                continue
            try:
                data = fetch_service_year(service, year, key, cfg)
            except RuntimeError as e:
                log.warning("%s yr=%d 수집 실패 → 건너뜀: %s", service, year, e)
                continue
            snapshot = {
                "service": service,
                "dataset_url": SERVICES[service]["url"],  # 키는 저장하지 않음
                "year_param": SERVICES[service]["year_param"],
                "year": year,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "totalCount": data["totalCount"],
                "items": data["items"],
            }
            snap.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
            log.info("저장: %s (%d건)", snap.name, len(data["items"]))
            paths.append(snap)
        out[service] = paths
    return out


def load_snapshots(cfg: dict, service: str) -> list[dict]:
    """저장된 스냅샷 로드 → item dict 리스트 (연도 오름차순, 각 item에 _yr 부여)."""
    raw_dir: Path = cfg["paths"]["raw"]
    rows: list[dict] = []
    for snap in sorted(raw_dir.glob(f"{service}_*.json")):
        data = json.loads(snap.read_text(encoding="utf-8"))
        for it in data["items"]:
            it["_yr"] = data["year"]
        rows.extend(data["items"])
    return rows


if __name__ == "__main__":
    cfg = load_config()
    collect_all(cfg)
