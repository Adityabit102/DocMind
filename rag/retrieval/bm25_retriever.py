"""BM25 sparse retriever wrapper (keyword/lexical matching)."""

from __future__ import annotations

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.config import get_settings


def build_bm25_retriever(chunks: list[Document], k: int | None = None) -> BM25Retriever:
    """Build a BM25 retriever from the in-memory chunk corpus.

    BM25 indexes raw tokens, so it complements dense retrieval on exact terms,
    identifiers, and rare keywords that embeddings tend to blur.
    """
    k = k or get_settings().retrieval_k
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = k
    return retriever
