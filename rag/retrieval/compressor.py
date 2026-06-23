"""Contextual compression: strip irrelevant sentences from retrieved chunks.

``LLMChainExtractor`` runs each chunk through the LLM with the query, keeping only
the relevant sentences. This shrinks token count and raises signal-to-noise before
the chunks reach the generation prompt.
"""

from __future__ import annotations

from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever


def build_compression_retriever(
    base_retriever: BaseRetriever, llm: BaseLanguageModel
) -> BaseRetriever:
    """Wrap a retriever with an LLM-based contextual compressor."""
    try:
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import LLMChainExtractor
    except ImportError:  # pragma: no cover - langchain >= 1.x shim
        from langchain_classic.retrievers import ContextualCompressionRetriever
        from langchain_classic.retrievers.document_compressors import LLMChainExtractor

    compressor = LLMChainExtractor.from_llm(llm)
    return ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
