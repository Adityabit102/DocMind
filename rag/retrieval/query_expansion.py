"""Pre-retrieval query transforms: multi-query expansion, HyDE, rewriting."""

from __future__ import annotations

import os

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "prompts")


def _load_prompt(name: str) -> str:
    with open(os.path.join(_PROMPTS_DIR, name), encoding="utf-8") as fh:
        return fh.read()


def build_multi_query_retriever(
    base_retriever: BaseRetriever, llm: BaseLanguageModel
) -> BaseRetriever:
    """Wrap a retriever so the LLM generates diverse query variants (higher recall)."""
    try:
        from langchain.retrievers.multi_query import MultiQueryRetriever
    except ImportError:  # pragma: no cover - langchain >= 1.x shim
        from langchain_classic.retrievers.multi_query import MultiQueryRetriever

    return MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)


def rewrite_query(query: str, llm: BaseLanguageModel, chat_history: str = "") -> str:
    """Rewrite a vague/short query into an explicit, self-contained search query."""
    template = PromptTemplate.from_template(_load_prompt("query_rewrite.txt"))
    chain = template | llm | StrOutputParser()
    return chain.invoke({"question": query, "chat_history": chat_history}).strip()


def hyde_query(query: str, llm: BaseLanguageModel) -> str:
    """Generate a hypothetical answer passage to embed in place of the raw query."""
    template = PromptTemplate.from_template(_load_prompt("hyde.txt"))
    chain = template | llm | StrOutputParser()
    return chain.invoke({"question": query}).strip()


def hyde_retrieve(
    query: str, llm: BaseLanguageModel, retriever: BaseRetriever
) -> list[Document]:
    """Run HyDE: synthesize a passage, then retrieve against it."""
    return retriever.invoke(hyde_query(query, llm))
