"""Lightweight background scheduler for periodic RAGAS evaluation.

A daemon ``threading.Timer`` loop — no extra dependency — that re-runs the
evaluation suite every ``scheduled_eval_interval_hours`` when
``enable_scheduled_eval`` is set. Results land in ``evaluation/results/`` like
any manual run, so the Evaluation page picks up the latest automatically.
Best-effort: a failed run is logged and the next tick is still scheduled.
"""

from __future__ import annotations

import logging
import threading

from app.config import get_settings

logger = logging.getLogger("docmind")

_timer: threading.Timer | None = None


def _run_once() -> None:
    """Generate a test set and evaluate the current index, persisting results."""
    from app.dependencies import get_state
    from rag.evaluation.ragas_eval import evaluate_dataset
    from rag.evaluation.testset_gen import generate_testset
    from rag.generation.chain import RAGChain
    from rag.generation.llm_factory import build_llm

    state = get_state()
    if state.store.is_empty:
        logger.info("Scheduled eval skipped — index is empty")
        return
    rows = generate_testset(state.store, get_settings().ragas_test_size)
    if not rows:
        logger.info("Scheduled eval skipped — could not build a test set")
        return
    chain = RAGChain(state.store, build_llm(streaming=False))
    from rag.retrieval.factory import RetrievalOverrides

    overrides = RetrievalOverrides(enable_query_expansion=False, top_k=5)
    questions, answers, contexts, ground_truths = [], [], [], []
    for row in rows:
        result = chain.answer(row["question"], overrides)
        questions.append(row["question"])
        answers.append(result.answer)
        contexts.append([s["text"] for s in result.sources] or [row["ground_truth"]])
        ground_truths.append(row["ground_truth"])
    evaluate_dataset(questions, answers, contexts, ground_truths)
    logger.info("Scheduled evaluation completed (%d questions)", len(questions))


def _tick() -> None:
    try:
        _run_once()
    except Exception as exc:  # noqa: BLE001 — never let a run kill the loop
        logger.warning("Scheduled evaluation failed: %s", exc)
    finally:
        _schedule_next()


def _schedule_next() -> None:
    global _timer
    interval = get_settings().scheduled_eval_interval_hours * 3600
    _timer = threading.Timer(interval, _tick)
    _timer.daemon = True
    _timer.start()


def start_scheduler() -> bool:
    """Start the periodic evaluation loop if enabled. Returns True if started."""
    if not get_settings().enable_scheduled_eval:
        return False
    _schedule_next()
    logger.info(
        "Scheduled evaluation enabled (every %d h)",
        get_settings().scheduled_eval_interval_hours,
    )
    return True


def stop_scheduler() -> None:
    """Cancel the pending timer (called on shutdown)."""
    global _timer
    if _timer is not None:
        _timer.cancel()
        _timer = None
