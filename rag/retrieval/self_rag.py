"""Self-RAG reflection: after generating an answer, check it is grounded.

A short LLM critique asks whether every claim in the answer is supported by the
retrieved context. If not, the caller can re-retrieve / regenerate once. Kept
deliberately small (one extra LLM call) to bound latency and cost.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

_REFLECT_TEMPLATE = (
    "You are grading whether an answer is fully grounded in the given context.\n"
    "Reply with exactly one word: GROUNDED if every claim is supported by the "
    "context, or UNGROUNDED if any claim is not.\n\n"
    "Context:\n{context}\n\nAnswer:\n{answer}\n\nVerdict:"
)


@dataclass
class ReflectionVerdict:
    grounded: bool
    raw: str


def reflect(
    answer: str, docs: list[Document], llm: BaseLanguageModel
) -> ReflectionVerdict:
    """Return whether ``answer`` is grounded in ``docs`` per an LLM self-critique."""
    if not answer.strip() or not docs:
        return ReflectionVerdict(grounded=False, raw="empty")
    context = "\n\n".join(d.page_content for d in docs)
    chain = PromptTemplate.from_template(_REFLECT_TEMPLATE) | llm | StrOutputParser()
    verdict = chain.invoke({"context": context, "answer": answer}).strip().upper()
    return ReflectionVerdict(grounded=verdict.startswith("GROUNDED"), raw=verdict)
