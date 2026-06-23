"""Reranking: local cross-encoder by default, hosted Cohere/Voyage optional.

A bi-encoder (FAISS) scores query and chunk independently — fast but imprecise.
A reranker reads (query, chunk) jointly and emits a sharper relevance score, so
reranking top-N candidates to top-k meaningfully cuts hallucination. The local
cross-encoder needs no key and is the default; ``reranker_provider`` can switch
to Cohere or Voyage, each falling back to the cross-encoder when its key or SDK
is missing (local-first never hard-fails on a hosted dependency).
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.documents import Document

from app.config import get_settings

logger = logging.getLogger("docmind")

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache
def _get_cross_encoder():  # type: ignore[no-untyped-def]
    from sentence_transformers import CrossEncoder

    return CrossEncoder(_MODEL_NAME)


def _sigmoid(x: float) -> float:
    """Squash an unbounded cross-encoder logit into a 0..1 relevance probability."""
    import math

    return 1.0 / (1.0 + math.exp(-x))


def _cross_encoder_rerank(
    query: str, candidates: list[Document], top_k: int
) -> list[tuple[Document, float]]:
    encoder = _get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in candidates]
    scores = encoder.predict(pairs)
    ranked = sorted(
        zip(candidates, scores, strict=False), key=lambda x: float(x[1]), reverse=True
    )
    # Cross-encoder outputs raw logits (roughly -11..+11); map to a 0..1
    # probability so relevance scores and confidence thresholds are meaningful.
    return [(doc, _sigmoid(float(score))) for doc, score in ranked[:top_k]]


def _cohere_rerank(
    query: str, candidates: list[Document], top_k: int
) -> list[tuple[Document, float]]:
    import cohere

    client = cohere.Client(get_settings().cohere_api_key)
    docs = [d.page_content for d in candidates]
    resp = client.rerank(query=query, documents=docs, top_n=top_k, model="rerank-english-v3.0")
    return [(candidates[r.index], float(r.relevance_score)) for r in resp.results]


def _voyage_rerank(
    query: str, candidates: list[Document], top_k: int
) -> list[tuple[Document, float]]:
    import voyageai

    client = voyageai.Client(api_key=get_settings().voyage_api_key)
    docs = [d.page_content for d in candidates]
    resp = client.rerank(query, docs, model="rerank-2", top_k=top_k)
    return [(candidates[r.index], float(r.relevance_score)) for r in resp.results]


def rerank(
    query: str, candidates: list[Document], top_k: int = 5
) -> list[tuple[Document, float]]:
    """Return the top-``k`` candidates as (document, score) pairs, best first.

    Uses the configured ``reranker_provider``; any hosted provider that is
    selected but unconfigured/unavailable falls back to the local cross-encoder.
    """
    if not candidates:
        return []
    settings = get_settings()
    provider = settings.reranker_provider
    try:
        if provider == "cohere" and settings.cohere_api_key:
            return _cohere_rerank(query, candidates, top_k)
        if provider == "voyage" and settings.voyage_api_key:
            return _voyage_rerank(query, candidates, top_k)
    except Exception as exc:  # noqa: BLE001 — hosted reranker failed → local fallback
        logger.warning("%s reranker failed, falling back to cross-encoder: %s", provider, exc)
    return _cross_encoder_rerank(query, candidates, top_k)


def deduplicate(docs: list[Document]) -> list[Document]:
    """Drop near-identical chunks (same document_id + char_offset, or exact text)."""
    seen: set = set()
    unique: list[Document] = []
    for doc in docs:
        key = (
            doc.metadata.get("document_id", ""),
            doc.metadata.get("char_offset", doc.page_content[:128]),
        )
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique
