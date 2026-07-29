"""FranSCORE 공통 유틸: 설정 로드·로깅·시드·합성 스모크 패널.

모든 모듈이 이 파일만 통해 설정/경로에 접근한다 (단일 진실 원천).
"""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent

_CFG_CACHE: dict | None = None


def load_config(path: str | Path | None = None) -> dict:
    """config.yaml 로드. cfg['_root']에 프로젝트 루트 Path를 담는다."""
    global _CFG_CACHE
    if path is None and _CFG_CACHE is not None:
        return _CFG_CACHE
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = ROOT
    for key, rel in cfg["paths"].items():
        p = ROOT / rel
        p.mkdir(parents=True, exist_ok=True)
        cfg["paths"][key] = p
    if path is None:
        _CFG_CACHE = cfg
    return cfg


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"franscore.{name}")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        try:
            fh = logging.FileHandler(ROOT / "outputs" / "pipeline.log", encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except OSError:
            pass
        logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# 합성 스모크 패널 — ⚠️ 파이프라인 배선 점검(스모크테스트) 전용.
# 제출 지표·데모 수치에 절대 사용하지 않는다 (outputs/_smoke 격리).
# ---------------------------------------------------------------------------

_INDUSTRIES = {
    "외식": ["치킨", "커피", "한식", "분식", "피자", "주점"],
    "서비스": ["교육", "세탁", "이미용"],
    "도소매": ["편의점", "화장품"],
}


def make_synthetic_panel(cfg: dict, n_brands: int = 300, years: tuple[int, int] = (2019, 2024)) -> pd.DataFrame:
    """INTERFACES.md §1 스키마를 따르는 합성 brand×year 패널.

    일부 브랜드는 특정 연도에 '구조악화 국면'으로 전환(성장 둔화→점포 순감·매출 하락·
    계약종료 급증)하도록 설계해 라벨·모델 코드의 배선을 점검할 수 있게 한다.
    """
    rng = np.random.default_rng(cfg["seed"])
    majors = list(_INDUSTRIES.keys())
    rows = []
    y0, y1 = years
    for i in range(n_brands):
        major = majors[rng.choice(len(majors), p=[0.7, 0.2, 0.1])]
        mid = _INDUSTRIES[major][rng.integers(len(_INDUSTRIES[major]))]
        brand_id = f"SYN{i:04d}"
        start = int(rng.integers(y0, y0 + 3))
        n_stores = float(rng.integers(15, 400))
        n_direct = max(0.0, round(n_stores * rng.uniform(0.0, 0.15)))
        avg_sales = float(rng.uniform(150_000, 700_000))  # 천원
        base_growth = rng.normal(0.06, 0.08)
        # 악화 전환 연도 (약 35% 브랜드, 관측 중반 이후)
        turn_year = int(rng.integers(start + 2, y1 + 1)) if rng.random() < 0.35 else None
        for year in range(start, y1 + 1):
            deteriorated = turn_year is not None and year >= turn_year
            g = rng.normal(-0.12, 0.06) if deteriorated else rng.normal(base_growth, 0.05)
            sg = rng.normal(-0.10, 0.05) if deteriorated else rng.normal(0.03, 0.05)
            end_rate = rng.uniform(0.12, 0.30) if deteriorated else rng.uniform(0.02, 0.10)
            prev_stores = n_stores
            n_stores = max(3.0, round(n_stores * (1 + g)))
            n_end = round(prev_stores * end_rate * rng.uniform(0.6, 1.0))
            n_cancel = round(prev_stores * end_rate * rng.uniform(0.0, 0.4))
            n_new = max(0.0, round(n_stores - prev_stores + n_end + n_cancel))
            avg_sales = max(30_000.0, avg_sales * (1 + sg))
            rows.append({
                "brand_id": brand_id,
                "brand_name": f"합성브랜드{i:04d}",
                "company_name": f"합성본부{i % 120:03d}",
                "industry_major": major,
                "industry_mid": mid,
                "year": year,
                "n_stores": n_stores,
                "n_direct": n_direct,
                "n_new": float(n_new),
                "n_contract_end": float(n_end),
                "n_contract_cancel": float(n_cancel),
                "n_name_change": float(rng.integers(0, max(2, int(prev_stores * 0.05)))),
                "avg_sales": avg_sales if rng.random() > 0.05 else np.nan,
                "avg_sales_per_area": avg_sales / rng.uniform(8, 25),
            })
    df = pd.DataFrame(rows).sort_values(["brand_id", "year"]).reset_index(drop=True)
    return df


def industry_group_col(panel: pd.DataFrame, min_group: int = 30) -> pd.Series:
    """업종그룹 결정: 해당 연도 industry_mid 그룹 크기 ≥ min_group이면 mid, 아니면 major.

    라벨·피처의 업종 내 분위수 계산에 공용으로 사용 (INTERFACES.md §1).
    """
    sizes = panel.groupby(["year", "industry_mid"])["brand_id"].transform("size")
    return panel["industry_mid"].where(sizes >= min_group, panel["industry_major"])
