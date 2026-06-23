"""Hybrid retrieval: dense FAISS + sparse BM25 fused via Reciprocal Rank Fusion.

LangChain's ``EnsembleRetriever`` already implements weighted RRF across child
retrievers, which is exactly the dense+sparse fusion the PRD specifies. A
standalone ``reciprocal_rank_fusion`` helper is also exposed for manual merges.
"""

from __future__ import annotations

from collections import defaultdict

try:  # langchain >= 1.x relocated these to langchain_classic
    from langchain.retrievers import EnsembleRetriever
except ImportError:  # pragma: no cover - version shim
    from langchain_classic.retrievers import EnsembleRetriever

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.config import get_settings


def build_hybrid_retriever(
    dense: BaseRetriever,
    sparse: BaseRetriever,
    dense_weight: float | None = None,
    bm25_weight: float | None = None,
) -> EnsembleRetriever:
    """Combine a dense and a sparse retriever with RRF and configured weights."""
    settings = get_settings()
    dw = settings.ensemble_dense_weight if dense_weight is None else dense_weight
    bw = settings.ensemble_bm25_weight if bm25_weight is None else bm25_weight
    return EnsembleRetriever(retrievers=[dense, sparse], weights=[dw, bw])


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]], k: int = 60, top_n: int = 30
) -> list[Document]:
    """Merge several ranked result lists into one via RRF.

    score(d) = Σ 1 / (k + rank_i(d)). Documents are keyed by (source, offset).
    """
    scores: dict[tuple, float] = defaultdict(float)
    by_key: dict[tuple, Document] = {}
    for results in ranked_lists:
        for rank, doc in enumerate(results):
            key = (
                doc.metadata.get("document_id", ""),
                doc.metadata.get("char_offset", doc.page_content[:64]),
            )
            scores[key] += 1.0 / (k + rank + 1)
            by_key.setdefault(key, doc)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [by_key[key] for key, _ in ordered[:top_n]]
