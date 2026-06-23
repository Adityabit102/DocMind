"""Retrieval A/B harness: compare two retrieval configs on a shared question set.

Runs the same synthetic questions through the RAG chain under each configuration
and scores both arms with the dependency-free grounding metrics (mean grounding,
USR, confidence, latency). Lets you answer "does HyDE / a bigger top-k / MMR
actually help on *my* corpus?" without standing up a full RAGAS run.
"""

from __future__ import annotations

from app.models.evaluation import ABArmConfig, ABArmScore, ABTestResult
from rag.evaluation.grounding import lexical_usr
from rag.ingestion.indexer import IndexStore
from rag.retrieval.factory import RetrievalOverrides


def _overrides(cfg: ABArmConfig) -> RetrievalOverrides:
    return RetrievalOverrides(
        mode=cfg.retrieval_mode,
        top_k=cfg.top_k,
        enable_query_expansion=cfg.enable_query_expansion,
        enable_hyde=cfg.enable_hyde,
        enable_context_compression=cfg.enable_context_compression,
    )


def _score_arm(cfg: ABArmConfig, questions: list[str], store: IndexStore) -> ABArmScore:
    from rag.generation.chain import RAGChain
    from rag.generation.llm_factory import build_llm

    chain = RAGChain(store, build_llm(streaming=False))
    overrides = _overrides(cfg)
    groundings, usrs, confidences, latencies = [], [], [], []
    for q in questions:
        result = chain.answer(q, overrides)
        contexts = [s["text"] for s in result.sources]
        if result.grounding_score is not None:
            groundings.append(result.grounding_score)
        usrs.append(lexical_usr(result.answer, contexts))
        confidences.append(result.confidence_score)
        latencies.append(result.latency_ms)

    def _mean(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    return ABArmScore(
        label=cfg.label,
        mean_grounding=_mean(groundings),
        mean_usr=_mean(usrs),
        mean_confidence=_mean(confidences),
        mean_latency_ms=_mean([float(x) for x in latencies]),
    )


def run_ab_test(
    store: IndexStore, test_size: int, arm_a: ABArmConfig, arm_b: ABArmConfig
) -> ABTestResult:
    """Score both arms over a shared synthetic question set and diff them."""
    from rag.evaluation.testset_gen import generate_testset

    rows = generate_testset(store, test_size)
    questions = [r["question"] for r in rows]
    score_a = _score_arm(arm_a, questions, store)
    score_b = _score_arm(arm_b, questions, store)

    deltas = {
        "grounding": round(score_b.mean_grounding - score_a.mean_grounding, 4),
        "usr": round(score_b.mean_usr - score_a.mean_usr, 4),
        "confidence": round(score_b.mean_confidence - score_a.mean_confidence, 4),
        "latency_ms": round(score_b.mean_latency_ms - score_a.mean_latency_ms, 4),
    }
    # Winner by grounding first (higher is better), then lower USR as tie-break.
    if deltas["grounding"] > 0.01 or (abs(deltas["grounding"]) <= 0.01 and deltas["usr"] < -0.01):
        winner = arm_b.label
    elif deltas["grounding"] < -0.01 or deltas["usr"] > 0.01:
        winner = arm_a.label
    else:
        winner = "tie"

    return ABTestResult(
        question_count=len(questions),
        arm_a=score_a,
        arm_b=score_b,
        winner=winner,
        deltas=deltas,
    )
