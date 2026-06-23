"""Corrective RAG (CRAG): a lightweight relevance gate over retrieved chunks.

Each chunk is scored for relevance to the query. If the best score falls below a
threshold, the retrieval is judged weak and the caller is signalled to refine the
query and re-retrieve. Uses the cross-encoder when available, otherwise a cheap
lexical-overlap heuristic so it degrades gracefully with no model present.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class CRAGVerdict:
    """Outcome of a corrective-RAG relevance check."""

    relevant: bool
    top_score: float
    action: str  # "use" | "refine"


def _lexical_overlap(query: str, text: str) -> float:
    q = {w.lower() for w in query.split() if len(w) > 2}
    if not q:
        return 0.0
    t = {w.lower() for w in text.split()}
    return len(q & t) / len(q)


def evaluate_relevance(
    query: str, docs: list[Document], threshold: float = 0.2
) -> CRAGVerdict:
    """Grade retrieval quality and recommend whether to use or refine."""
    if not docs:
        return CRAGVerdict(relevant=False, top_score=0.0, action="refine")

    try:
        from rag.retrieval.reranker import rerank

        # rerank already returns a 0..1 relevance probability per chunk.
        scored = rerank(query, docs, top_k=1)
        top = float(scored[0][1]) if scored else 0.0
    except Exception:  # noqa: BLE001 — no model available → lexical fallback
        top = max(_lexical_overlap(query, d.page_content) for d in docs)

    relevant = top >= threshold
    return CRAGVerdict(relevant=relevant, top_score=top, action="use" if relevant else "refine")
