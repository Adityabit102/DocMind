"""Embedding-model factory with local-first auto-fallback.

Selecting ``openai`` without an API key silently degrades to the local
HuggingFace MiniLM model so indexing never hard-fails on a missing secret.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from app.config import Settings, get_settings

# HuggingFace dimensions, used for FAISS index sizing decisions if needed.
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "all-MiniLM-L6-v2": 384,
    "BAAI/bge-large-en-v1.5": 1024,
}


def _build_openai(model: str, api_key: str) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=model, api_key=api_key)


def _build_huggingface(model: str) -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_embeddings(settings: Settings | None = None) -> Embeddings:
    """Return an ``Embeddings`` instance honouring the effective provider."""
    settings = settings or get_settings()
    provider = settings.effective_embedding_provider
    if provider == "openai":
        return _build_openai(settings.embedding_model, settings.openai_api_key)
    # huggingface (local) — use a sane local default if an OpenAI model name leaked through.
    model = settings.embedding_model
    if model.startswith("text-embedding"):
        model = "all-MiniLM-L6-v2"
    return _build_huggingface(model)


@lru_cache
def get_embeddings() -> Embeddings:
    """Process-wide cached embeddings instance."""
    return build_embeddings()


def embedding_dimension(model: str) -> int:
    """Best-effort dimensionality lookup (defaults to 384)."""
    return EMBEDDING_DIMENSIONS.get(model, 384)
