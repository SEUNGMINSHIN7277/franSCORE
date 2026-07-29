"""M1 엔티티 정합: 브랜드명·법인명 정규화 → 연도 간 브랜드 연결 → 마스터.

배경: 공정위 브랜드 API에는 고유 브랜드 ID가 없다(조사로 확인).
연결 키 전략(정합 규칙 — 문서화 의무, 명세 2.2):
  1) 브랜드명·법인명을 정규화(NFKC, 공백/특수문자 제거, ㈜/주식회사 제거, 소문자화).
  2) 1차 그룹 = (정규화 브랜드명, 업종대분류). 연도가 달라도 같은 그룹이면 같은 브랜드로 연결
     (본부 법인 변경·리브랜딩 소폭 변형은 같은 브랜드로 취급, 변경 이력 로깅).
  3) 동일 그룹 안에서 "같은 연도"에 서로 다른 법인이 공존하면 동명이브랜드(동명이업종 포함)로
     판단하고 (브랜드명#법인명)으로 분리한다.
  4) 연결 불가/모호 잔여는 버리고 개수를 로깅한다.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

from src.common import get_logger, load_config
from src.collect import load_snapshots

log = get_logger("entity")

_CORP_PREFIX = re.compile(r"(주식회사|\(주\)|㈜|\(유\)|유한회사|농업회사법인|\(재\)|재단법인|\(사\)|사단법인)")
_NON_ALNUM = re.compile(r"[^0-9a-z가-힣]")


def normalize_name(s: str | None) -> str:
    """브랜드/법인명 정규화: NFKC → 법인접두 제거 → 특수문자·공백 제거 → 소문자."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = unicodedata.normalize("NFKC", str(s)).strip().lower()
    s = _CORP_PREFIX.sub("", s)
    s = _NON_ALNUM.sub("", s)
    return s


def build_master(cfg: dict) -> pd.DataFrame:
    """brand_frcs_stats 스냅샷 → brand_id 부여된 원천 행 단위 매핑 DataFrame.

    반환 컬럼: [_yr, brand_name, company_name, industry_major, industry_mid,
               norm_brand, norm_corp, brand_id]
    """
    rows = load_snapshots(cfg, "brand_frcs_stats")
    if not rows:
        raise FileNotFoundError("brand_frcs_stats 스냅샷 없음 — 먼저 `python run_pipeline.py --step collect`")
    df = pd.DataFrame(rows)
    n0 = len(df)

    df["brand_name"] = df["brandNm"].astype(str).str.strip()
    df["company_name"] = df["corpNm"].astype(str).str.strip()
    df["industry_major"] = df["indutyLclasNm"].astype(str).str.strip()
    df["industry_mid"] = df["indutyMlsfcNm"].astype(str).str.strip()
    df["norm_brand"] = df["brand_name"].map(normalize_name)
    df["norm_corp"] = df["company_name"].map(normalize_name)

    # 정규화 후 브랜드명이 비는 행은 연결 불가 → 제거(로깅)
    bad = df["norm_brand"] == ""
    if bad.any():
        log.warning("브랜드명 정규화 불가 %d행 제거", int(bad.sum()))
    df = df[~bad].copy()

    # (norm_brand, industry_major) 그룹에서 같은 연도에 법인 2개 이상 → 동명이브랜드로 분리
    grp_year_corps = df.groupby(["norm_brand", "industry_major", "_yr"])["norm_corp"].nunique()
    collide = grp_year_corps[grp_year_corps > 1].reset_index()[["norm_brand", "industry_major"]].drop_duplicates()
    collide_set = set(map(tuple, collide.values))
    log.info("동명 충돌 그룹(같은 연도 복수 법인): %d개 → 법인명으로 분리", len(collide_set))

    def _brand_id(r) -> str:
        key = (r["norm_brand"], r["industry_major"])
        if key in collide_set:
            return f"{r['norm_brand']}#{r['norm_corp']}@{r['industry_major']}"
        return f"{r['norm_brand']}@{r['industry_major']}"

    df["brand_id"] = df.apply(_brand_id, axis=1)

    # 본부(법인) 변경 이력 로깅 — 같은 brand_id에 복수 법인 (연도 간 변경)
    corp_changes = df.groupby("brand_id")["norm_corp"].nunique()
    n_changed = int((corp_changes > 1).sum())
    log.info("정합 요약: 원천 %d행 → 유효 %d행, 고유 브랜드 %d개, 본부변경 이력 브랜드 %d개",
             n0, len(df), df["brand_id"].nunique(), n_changed)
    return df


if __name__ == "__main__":
    cfg = load_config()
    m = build_master(cfg)
    print(m[["brand_id", "brand_name", "company_name", "industry_major", "_yr"]].head(20).to_string())
