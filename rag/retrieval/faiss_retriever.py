"""FAISS dense retriever wrapper supporting similarity and MMR search."""

from __future__ import annotations

from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from app.config import get_settings


def build_faiss_retriever(
    vectorstore: VectorStore,
    mode: str = "similarity",
    k: int | None = None,
    lambda_mult: float = 0.5,
) -> BaseRetriever:
    """Return a FAISS retriever in ``similarity`` or ``mmr`` mode.

    MMR (Maximal Marginal Relevance) balances relevance against diversity via
    ``lambda_mult`` (1.0 = pure relevance, 0.0 = max diversity).
    """
    k = k or get_settings().retrieval_k
    if mode == "mmr":
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 2, "lambda_mult": lambda_mult},
        )
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
