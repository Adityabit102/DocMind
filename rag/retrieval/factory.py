"""Assemble the full retriever stack from configuration.

Composition order (each layer optional, driven by ``Settings`` / per-query
overrides):

    base (similarity | mmr | hybrid[FAISS+BM25])
      → multi-query expansion (LLM)
      → contextual compression (LLM)

Reranking and CRAG/Self-RAG are applied by the generation chain, not here, since
they need the query text and produce scores the chain reports back to the UI.
A ``filter_documents`` helper provides pre-/post-retrieval metadata scoping.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever

from app.config import Settings, get_settings
from rag.ingestion.indexer import IndexStore
from rag.retrieval.bm25_retriever import build_bm25_retriever
from rag.retrieval.compressor import build_compression_retriever
from rag.retrieval.faiss_retriever import build_faiss_retriever
from rag.retrieval.hybrid import build_hybrid_retriever
from rag.retrieval.query_expansion import build_multi_query_retriever


@dataclass
class RetrievalOverrides:
    """Per-query overrides; ``None`` fields fall back to ``Settings``."""

    mode: str | None = None
    top_k: int | None = None
    enable_query_expansion: bool | None = None
    enable_hyde: bool | None = None
    enable_context_compression: bool | None = None
    document_ids: list[str] | None = None
    tags: list[str] | None = None
    page_range: tuple[int, int] | None = None


def _resolve(value, default):  # type: ignore[no-untyped-def]
    return default if value is None else value


def build_retriever(
    store: IndexStore,
    llm: BaseLanguageModel | None = None,
    overrides: RetrievalOverrides | None = None,
    settings: Settings | None = None,
) -> BaseRetriever:
    """Build a retriever for the current index honouring config + overrides."""
    settings = settings or get_settings()
    overrides = overrides or RetrievalOverrides()
    if store.vectorstore is None:
        raise ValueError("Index is empty — ingest documents before querying.")

    mode = _resolve(overrides.mode, settings.retrieval_mode)
    k = _resolve(overrides.top_k, settings.retrieval_k)

    if mode == "hybrid":
        dense = build_faiss_retriever(store.vectorstore, "similarity", k)
        sparse = build_bm25_retriever(store.chunks, k)
        retriever: BaseRetriever = build_hybrid_retriever(dense, sparse)
    else:  # similarity | mmr
        retriever = build_faiss_retriever(store.vectorstore, mode, k)

    if _resolve(overrides.enable_query_expansion, settings.enable_query_expansion) and llm:
        retriever = build_multi_query_retriever(retriever, llm)

    if _resolve(overrides.enable_context_compression, settings.enable_context_compression) and llm:
        retriever = build_compression_retriever(retriever, llm)

    return retriever


def filter_documents(
    docs: list[Document], overrides: RetrievalOverrides
) -> list[Document]:
    """Apply metadata scoping (document ids, tags, page range) to results."""
    out = docs
    if overrides.document_ids:
        wanted = set(overrides.document_ids)
        out = [d for d in out if d.metadata.get("document_id") in wanted]
    if overrides.tags:
        tagset = set(overrides.tags)
        out = [d for d in out if tagset & set(d.metadata.get("tags", []))]
    if overrides.page_range:
        lo, hi = overrides.page_range
        out = [d for d in out if lo <= d.metadata.get("page_number", 0) <= hi]
    return out
