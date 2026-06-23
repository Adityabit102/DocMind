"""Evaluation endpoints: run RAGAS, fetch results, export CSV, drift status."""

from __future__ import annotations

import csv
import io

from fastapi import APIRouter, Depends, HTTPException
from starlette.responses import StreamingResponse

from app.dependencies import AppState, get_state
from app.models.evaluation import (
    ABTestRequest,
    ABTestResult,
    DriftStatus,
    EvaluationRequest,
    EvaluationResult,
)

router = APIRouter(prefix="/api/v1/evaluate", tags=["evaluation"])


@router.post("", response_model=EvaluationResult)
def run_evaluation(
    req: EvaluationRequest, state: AppState = Depends(get_state)
) -> EvaluationResult:
    """Generate a synthetic test set and run RAGAS over the current index."""
    if state.store.is_empty:
        raise HTTPException(status_code=409, detail="No documents indexed yet")
    from rag.evaluation.ragas_eval import evaluate_dataset
    from rag.evaluation.testset_gen import generate_testset
    from rag.generation.chain import RAGChain
    from rag.generation.llm_factory import build_llm

    rows = generate_testset(state.store, req.test_size)
    if not rows:
        raise HTTPException(status_code=409, detail="Could not generate a test set")
    # Each question runs the full RAG chain (a live LLM call), so cap the count
    # to keep a UI-triggered run from taking many minutes on a local model.
    rows = rows[: min(req.test_size, 10)]

    chain = RAGChain(state.store, build_llm(streaming=False))
    # Lean retrieval for eval: skip LLM query-expansion (several extra calls per
    # question) so a run finishes in a reasonable time on a local model.
    from rag.retrieval.factory import RetrievalOverrides

    overrides = RetrievalOverrides(enable_query_expansion=False, top_k=5)
    questions, answers, contexts, ground_truths = [], [], [], []
    for row in rows:
        result = chain.answer(row["question"], overrides)
        questions.append(row["question"])
        answers.append(result.answer)
        contexts.append([s["text"] for s in result.sources] or [row["ground_truth"]])
        ground_truths.append(row["ground_truth"])
    return evaluate_dataset(questions, answers, contexts, ground_truths)


@router.get("/results", response_model=EvaluationResult)
def get_results() -> EvaluationResult:
    from rag.evaluation.ragas_eval import load_latest_result

    result = load_latest_result()
    if result is None:
        raise HTTPException(status_code=404, detail="No evaluation has been run yet")
    return result


@router.get("/export")
def export_csv() -> StreamingResponse:
    """Download the latest per-question evaluation breakdown as CSV."""
    from rag.evaluation.ragas_eval import load_latest_result

    result = load_latest_result()
    if result is None:
        raise HTTPException(status_code=404, detail="No evaluation has been run yet")

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["question", "answer", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    )
    for q in result.per_question:
        writer.writerow(
            [q.question, q.answer, q.faithfulness, q.answer_relevancy, q.context_precision, q.context_recall]
        )
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=evaluation.csv"},
    )


@router.post("/ab", response_model=ABTestResult)
def run_ab(req: ABTestRequest, state: AppState = Depends(get_state)) -> ABTestResult:
    """Compare two retrieval configurations on a shared synthetic question set."""
    if state.store.is_empty:
        raise HTTPException(status_code=409, detail="No documents indexed yet")
    from rag.evaluation.ab_test import run_ab_test

    result = run_ab_test(state.store, req.test_size, req.arm_a, req.arm_b)
    if result.question_count == 0:
        raise HTTPException(status_code=409, detail="Could not generate a test set")
    return result


@router.get("/drift", response_model=DriftStatus)
def drift_status(state: AppState = Depends(get_state)) -> DriftStatus:
    """Return current embedding-distribution drift status."""
    from rag.evaluation.drift import detect_drift
    from rag.ingestion.embedder import get_embeddings

    chunks = state.store.chunks
    if len(chunks) < 4:
        return DriftStatus(message="Not enough data to assess drift")
    embeddings = get_embeddings()
    texts = [c.page_content for c in chunks]
    half = len(texts) // 2
    reference = embeddings.embed_documents(texts[:half])
    current = embeddings.embed_documents(texts[half:])
    return detect_drift(reference, current)
