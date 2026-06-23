"""Classic retrieval-quality metrics: Recall@k, NDCG, MRR.

Pure-Python/numpy implementations operating on ranked lists of chunk ids with a
set of relevant ids — no external service required.
"""

from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items present in the top-``k`` retrieved."""
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return len(top & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant item (0 if none retrieved)."""
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(rankings: list[tuple[list[str], set[str]]]) -> float:
    """Average reciprocal rank across multiple (retrieved, relevant) queries."""
    if not rankings:
        return 0.0
    return sum(reciprocal_rank(r, rel) for r, rel in rankings) / len(rankings)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain at ``k`` (binary relevance)."""
    dcg = 0.0
    for i, item in enumerate(retrieved[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
