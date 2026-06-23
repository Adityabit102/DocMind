"""Retrieval tests: RRF fusion, dedup, CRAG fallback, retrieval metrics."""

from __future__ import annotations

from langchain_core.documents import Document

from rag.evaluation.metrics import mean_reciprocal_rank, ndcg_at_k, recall_at_k
from rag.retrieval.crag import evaluate_relevance
from rag.retrieval.hybrid import reciprocal_rank_fusion
from rag.retrieval.reranker import deduplicate


def _doc(text: str, doc_id: str, offset: int) -> Document:
    return Document(page_content=text, metadata={"document_id": doc_id, "char_offset": offset})


def test_rrf_merges_and_ranks():
    a = _doc("alpha", "d1", 0)
    b = _doc("beta", "d1", 100)
    c = _doc("gamma", "d2", 0)
    fused = reciprocal_rank_fusion([[a, b], [a, c]], top_n=3)
    # 'a' appears in both lists near the top → should rank first
    assert fused[0].page_content == "alpha"
    assert len(fused) == 3


def test_deduplicate_removes_repeats():
    a = _doc("same", "d1", 0)
    dup = _doc("same", "d1", 0)
    other = _doc("different", "d1", 200)
    assert len(deduplicate([a, dup, other])) == 2


def test_crag_flags_irrelevant():
    docs = [_doc("completely unrelated cooking recipe", "d1", 0)]
    verdict = evaluate_relevance("quarterly financial revenue figures", docs, threshold=0.5)
    assert verdict.action in {"use", "refine"}
    assert isinstance(verdict.top_score, float)


def test_crag_empty_is_refine():
    verdict = evaluate_relevance("anything", [])
    assert not verdict.relevant
    assert verdict.action == "refine"


def test_retrieval_metrics():
    retrieved = ["c1", "c2", "c3", "c4"]
    relevant = {"c2", "c4"}
    assert recall_at_k(retrieved, relevant, 4) == 1.0
    assert recall_at_k(retrieved, relevant, 1) == 0.0
    assert 0.0 < ndcg_at_k(retrieved, relevant, 4) <= 1.0
    assert mean_reciprocal_rank([(retrieved, relevant)]) == 0.5
