"""진짜 RAG — 검색 기반 근거 인용 (명세 COULD "진짜 RAG").

왜 필요한가:
    심사메모가 "근거만 인용"하려면 근거가 **검색 가능한 문서**여야 한다. 단순히
    프롬프트에 수치를 붙여 넣는 것은 RAG가 아니다. 본 모듈은 실제 문서 코퍼스를
    구축하고 색인·검색하여, 메모가 인용한 문장이 어느 문서(출처 URL·발행일)의
    몇 번째 청크에서 왔는지 추적 가능하게 만든다.

코퍼스 구성 (2종, 모두 실데이터):
    ① 뉴스 문서   — data/raw/news/*.json 스냅샷 (제목·출처·발행일·링크)
    ② 공시 문서   — 브랜드×연도 패널을 자연어 문장으로 렌더링
                    ("2023년 A브랜드: 가맹점 57개(전년비 -42.1%), 평균매출 1.16억원,
                      계약종료율 12.3%, 직영비중 1.2%, 진출 8개 시도")
                    → 모형이 본 수치와 메모가 인용하는 근거가 **동일 원천**임을 보장

검색 방식 (재현성 우선):
    TF-IDF + 코사인 유사도. 한국어 형태소 분석기 없이도 동작하도록 **문자 n-gram
    (char_wb 2~4)** 과 단어 토큰을 결합한다. 외부 임베딩 API에 의존하지 않으므로
    키 없이 완전 재현 가능하고, seed 고정과 무관하게 결정적이다.
    (임베딩 기반 확장은 로드맵 — 본 프로토타입 범위에서는 재현성을 우선한다.)

산출물:
    data/processed/rag_corpus.parquet   문서 코퍼스 (doc_id·brand_id·text·출처)
    outputs/rag_index.joblib            벡터라이저 + 문서 행렬
    outputs/rag_stats.json              코퍼스 통계 (문서 수·유형별 분포)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.common import get_logger, load_config

try:  # 뉴스 개체 매칭 규칙 재사용 (단일 정의 원칙)
    from src.news_llm import _entity_match
except Exception:  # noqa: BLE001
    _entity_match = None  # type: ignore[assignment]

log = get_logger("rag")

CORPUS_FILE = "rag_corpus.parquet"
INDEX_FILE = "rag_index.joblib"
STATS_FILE = "rag_stats.json"

_WS = re.compile(r"\s+")


def _clean(s: str) -> str:
    return _WS.sub(" ", str(s or "")).strip()


# ---------------------------------------------------------------------------
# 코퍼스 구축
# ---------------------------------------------------------------------------

def _news_documents(cfg: dict) -> list[dict]:
    """뉴스 스냅샷 → 문서. 브랜드명으로 brand 태깅(파일명 = 브랜드명 기반)."""
    news_dir = Path(cfg["paths"]["raw"]) / "news"
    docs: list[dict] = []
    if not news_dir.exists():
        return docs
    for fp in sorted(news_dir.glob("*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        brand = _clean(data.get("brand") or fp.stem)
        for i, art in enumerate(data.get("articles") or []):
            title = _clean(art.get("title"))
            if not title:
                continue
            # 개체 매칭 필터: 일반명사·부분문자 오염 제거 (news_llm과 동일 규칙 재사용).
            # 없으면 문자 n-gram 검색이 무관 기사를 근거로 끌어올 수 있다(실측 오탐 확인).
            if _entity_match is not None and not _entity_match(brand, title):
                continue
            docs.append({
                "doc_id": f"news:{fp.stem}:{i}",
                "brand_name": brand,
                "source_type": "news",
                "text": title,
                "url": _clean(art.get("link")),
                "published": _clean(art.get("published")),
                "source_name": _clean(art.get("source")),
                "year": np.nan,
            })
    return docs


def _disclosure_documents(cfg: dict, max_rows: int | None = None) -> list[dict]:
    """브랜드×연도 패널 → 자연어 공시 문서 (모형 입력과 동일 원천)."""
    panel_path = Path(cfg["paths"]["processed"]) / "panel.parquet"
    if not panel_path.exists():
        return []
    p = pd.read_parquet(panel_path)
    if "eligible_t" in p.columns:                      # 표본 자격 행 우선
        p = pd.concat([p[p["eligible_t"]], p[~p["eligible_t"]]])
    if max_rows:
        p = p.head(max_rows)
    docs: list[dict] = []
    for r in p.itertuples():
        parts = [f"{int(r.year)}년 {r.brand_name} ({r.industry_major}/{r.industry_mid}) 공시"]
        if pd.notna(r.n_stores):
            parts.append(f"가맹점 {int(r.n_stores)}개")
        if pd.notna(getattr(r, "n_direct", np.nan)):
            parts.append(f"직영점 {int(r.n_direct)}개")
        if pd.notna(r.avg_sales):
            parts.append(f"가맹점 평균매출 {r.avg_sales / 1e4:.2f}억원")
        end = (0 if pd.isna(r.n_contract_end) else r.n_contract_end) + \
              (0 if pd.isna(r.n_contract_cancel) else r.n_contract_cancel)
        parts.append(f"계약종료·해지 {int(end)}건")
        if pd.notna(r.n_new):
            parts.append(f"신규개점 {int(r.n_new)}개")
        nreg = getattr(r, "n_regions", np.nan)
        if pd.notna(nreg):
            parts.append(f"진출 {int(nreg)}개 시도")
        hhi = getattr(r, "region_hhi", np.nan)
        if pd.notna(hhi):
            parts.append(f"지역집중도(HHI) {hhi:.3f}")
        if getattr(r, "cancel_flag", 0) == 1:
            parts.append(f"브랜드 등록취소({getattr(r, 'cancel_type', '') or '유형미상'})")
        docs.append({
            "doc_id": f"disc:{r.brand_id}:{int(r.year)}",
            "brand_name": r.brand_name,
            "source_type": "disclosure",
            "text": ", ".join(parts),
            "url": "공정거래위원회 가맹사업 정보공개 오픈API (15110241/15125490)",
            "published": f"{int(r.year)}",
            "source_name": "공정거래위원회",
            "year": int(r.year),
            "brand_id": r.brand_id,
        })
    return docs


def build_corpus(cfg: dict, max_disclosure_rows: int | None = 20000) -> pd.DataFrame:
    """뉴스 + 공시 문서 코퍼스 구축 및 저장."""
    docs = _news_documents(cfg) + _disclosure_documents(cfg, max_disclosure_rows)
    if not docs:
        log.warning("RAG 코퍼스 문서 0건 — 뉴스/패널 산출물 확인 필요")
        return pd.DataFrame()
    corpus = pd.DataFrame(docs)
    if "brand_id" not in corpus.columns:
        corpus["brand_id"] = pd.NA
    corpus = corpus.drop_duplicates(subset=["doc_id"]).reset_index(drop=True)
    out = Path(cfg["paths"]["processed"]) / CORPUS_FILE
    corpus.to_parquet(out, index=False)
    stats = {
        "n_documents": int(len(corpus)),
        "by_source_type": corpus["source_type"].value_counts().to_dict(),
        "n_brands_covered": int(corpus["brand_name"].nunique()),
        "retrieval": "TF-IDF (word 1-2gram + char_wb 2-4gram) + cosine similarity",
        "note": "외부 임베딩 API 미사용 — 키 없이 완전 재현 가능",
    }
    (Path(cfg["paths"]["outputs"]) / STATS_FILE).write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("RAG 코퍼스: %d문서 (%s), 브랜드 %d개 → %s",
             len(corpus), stats["by_source_type"], stats["n_brands_covered"], out.name)
    return corpus


# ---------------------------------------------------------------------------
# 색인 / 검색
# ---------------------------------------------------------------------------

class RagIndex:
    """TF-IDF 색인 + 브랜드 필터 검색."""

    def __init__(self, corpus: pd.DataFrame):
        self.corpus = corpus.reset_index(drop=True)
        texts = self.corpus["text"].fillna("").tolist()
        # 한국어: 형태소 분석기 없이도 동작하도록 단어 + 문자 n-gram 결합
        self.vec_word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                        min_df=2, sublinear_tf=True)
        self.mat_word = self.vec_word.fit_transform(texts)
        self.mat_char = self.vec_char.fit_transform(texts)

    def _score(self, query: str) -> np.ndarray:
        qw = self.vec_word.transform([query])
        qc = self.vec_char.transform([query])
        sw = cosine_similarity(qw, self.mat_word).ravel()
        sc = cosine_similarity(qc, self.mat_char).ravel()
        return 0.5 * sw + 0.5 * sc

    def retrieve(self, query: str, k: int = 5, brand_name: str | None = None,
                 source_type: str | None = None, min_score: float = 0.02) -> pd.DataFrame:
        """질의에 대한 상위 k 문서. brand_name/source_type으로 사전 필터 가능."""
        s = self._score(query)
        mask = np.ones(len(self.corpus), dtype=bool)
        if brand_name:
            mask &= (self.corpus["brand_name"].astype(str) == str(brand_name)).to_numpy()
        if source_type:
            mask &= (self.corpus["source_type"].astype(str) == str(source_type)).to_numpy()
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return self.corpus.head(0).assign(score=[])
        sub = s[idx]
        order = np.argsort(-sub)[:k]
        pick = idx[order]
        out = self.corpus.iloc[pick].copy()
        out["score"] = s[pick]
        return out[out["score"] >= min_score].reset_index(drop=True)


def build_index(cfg: dict, corpus: pd.DataFrame | None = None) -> RagIndex:
    if corpus is None:
        cp = Path(cfg["paths"]["processed"]) / CORPUS_FILE
        corpus = pd.read_parquet(cp) if cp.exists() else build_corpus(cfg)
    if corpus is None or corpus.empty:
        raise ValueError("RAG 코퍼스가 비어 색인을 만들 수 없습니다.")
    index = RagIndex(corpus)
    joblib.dump(index, Path(cfg["paths"]["outputs"]) / INDEX_FILE)
    log.info("RAG 색인 저장: %s (문서 %d, word차원 %d, char차원 %d)", INDEX_FILE,
             len(corpus), index.mat_word.shape[1], index.mat_char.shape[1])
    return index


def load_index(cfg: dict) -> RagIndex | None:
    fp = Path(cfg["paths"]["outputs"]) / INDEX_FILE
    if not fp.exists():
        return None
    try:
        return joblib.load(fp)
    except Exception as e:  # noqa: BLE001
        log.warning("RAG 색인 로드 실패 (%s) — 재구축 필요", e)
        return None


def retrieve_evidence(cfg: dict, brand_name: str, risk_factors: list[str],
                      k_per_query: int = 3, index: RagIndex | None = None) -> list[dict]:
    """브랜드 + SHAP 위험요인을 질의로 근거 문서 검색 → 인용 가능한 리스트.

    반환 각 항목: {doc_id, source_type, text, url, published, source_name, score, query}
    """
    idx = index or load_index(cfg)
    if idx is None:
        log.warning("RAG 색인 없음 — 근거 검색 생략 (build_index 먼저 실행)")
        return []
    seen: set[str] = set()
    out: list[dict] = []
    queries = [f"{brand_name} 가맹점 계약종료 폐점 위험"] + \
              [f"{brand_name} {rf}" for rf in (risk_factors or [])[:4]]
    for q in queries:
        # 브랜드 한정 검색 → 결과 없으면 전체 코퍼스로 완화
        hits = idx.retrieve(q, k=k_per_query, brand_name=brand_name)
        if hits.empty:
            hits = idx.retrieve(q, k=k_per_query)
        for r in hits.itertuples():
            if r.doc_id in seen:
                continue
            seen.add(r.doc_id)
            out.append({"doc_id": r.doc_id, "source_type": r.source_type,
                        "text": r.text, "url": r.url, "published": r.published,
                        "source_name": r.source_name, "score": round(float(r.score), 4),
                        "query": q})
    out.sort(key=lambda d: -d["score"])
    return out


def main(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    corpus = build_corpus(cfg)
    if corpus.empty:
        return
    index = build_index(cfg, corpus)
    # 데모: 첫 뉴스 브랜드로 검색 예시 출력
    sample_brand = corpus.loc[corpus["source_type"] == "news", "brand_name"]
    if len(sample_brand):
        b = sample_brand.iloc[0]
        ev = retrieve_evidence(cfg, b, ["계약종료율 급증", "가맹점 순감"], index=index)
        print(f"\n=== RAG 검색 예시: {b} ===")
        for e in ev[:5]:
            print(f"  [{e['source_type']} {e['score']:.3f}] {e['text'][:80]}")
            print(f"     ↳ {e['source_name']} {e['published']}")


if __name__ == "__main__":
    main()
