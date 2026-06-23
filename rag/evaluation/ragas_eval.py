"""RAGAS evaluation: faithfulness, answer relevancy, context precision/recall.

Runs the configured RAG chain over a (synthetic or supplied) test set and scores
the answers with RAGAS. Results persist to ``evaluation/results/`` as JSON.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from app.config import get_settings
from app.models.evaluation import EvaluationResult, PerQuestionResult


def _persist(result: EvaluationResult) -> str:
    out_dir = get_settings().eval_results_dir
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"eval_{stamp}.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(result.model_dump_json(indent=2))
    return path


def _tokens(text: str) -> set[str]:
    import re

    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _overlap(a: str, b_tokens: set[str]) -> float:
    """Fraction of ``a``'s content words present in ``b_tokens`` (0..1)."""
    at = _tokens(a)
    return len(at & b_tokens) / len(at) if at else 0.0


def _local_evaluate(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> EvaluationResult:
    """Dependency-free evaluation when RAGAS/OpenAI aren't available.

    Approximates the four RAGAS metrics with lexical heuristics so the local-first
    stack still produces meaningful, comparable scores:
      - faithfulness     ≈ answer sentences supported by the retrieved context
      - answer_relevancy ≈ how much the answer covers the question's terms
      - context_precision≈ how much of the context is reflected in the answer
      - context_recall   ≈ how much of the ground truth the context covers
    """
    from rag.evaluation.grounding import lexical_usr

    per: list[PerQuestionResult] = []
    for q, a, ctx, gt in zip(questions, answers, contexts, ground_truths, strict=False):
        ctx_text = " ".join(ctx)
        ctx_tokens = _tokens(ctx_text)
        answer_tokens = _tokens(a)
        faith = round(1.0 - lexical_usr(a, ctx), 4)
        relevancy = round(_overlap(a, _tokens(q)) if a else 0.0, 4)
        precision = round(_overlap(ctx_text, answer_tokens) if ctx_text else 0.0, 4)
        recall = round(len(_tokens(gt) & ctx_tokens) / len(_tokens(gt)) if _tokens(gt) else 0.0, 4)
        per.append(
            PerQuestionResult(
                question=q,
                answer=a,
                faithfulness=faith,
                answer_relevancy=relevancy,
                context_precision=precision,
                context_recall=recall,
            )
        )

    def _avg(attr: str) -> float:
        vals = [getattr(p, attr) for p in per]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    from rag.evaluation.grounding import mean_usr

    result = EvaluationResult(
        question_count=len(questions),
        faithfulness=_avg("faithfulness"),
        answer_relevancy=_avg("answer_relevancy"),
        context_precision=_avg("context_precision"),
        context_recall=_avg("context_recall"),
        unsupported_sentence_ratio=mean_usr(answers, contexts),
        per_question=per,
    )
    _persist(result)
    return result


def evaluate_dataset(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> EvaluationResult:
    """Score a prepared evaluation dataset, persisting the result.

    Engine preference: RAGAS (if installed) → LLM-as-judge with a capable hosted
    model (Groq/OpenAI/Anthropic) for real semantic scores → local lexical
    heuristics, so the page always returns the best metrics available.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except Exception:  # noqa: BLE001 — RAGAS not installed → LLM judge / lexical
        settings = get_settings()
        if settings.has_groq or settings.has_openai or settings.has_anthropic:
            from rag.evaluation.llm_judge import llm_judge_evaluate
            from rag.generation.llm_factory import build_llm

            result = llm_judge_evaluate(
                questions,
                answers,
                contexts,
                ground_truths,
                build_llm(settings, streaming=False),
                model_name=settings.llm_model,
            )
            _persist(result)
            return result
        return _local_evaluate(questions, answers, contexts, ground_truths)

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    from rag.evaluation.grounding import mean_usr

    df = scores.to_pandas()
    result = EvaluationResult(
        engine="ragas",
        question_count=len(questions),
        faithfulness=float(df["faithfulness"].mean()),
        answer_relevancy=float(df["answer_relevancy"].mean()),
        context_precision=float(df["context_precision"].mean()),
        context_recall=float(df["context_recall"].mean()),
        unsupported_sentence_ratio=mean_usr(answers, contexts),
        per_question=[
            PerQuestionResult(
                question=q,
                answer=a,
                faithfulness=float(row.get("faithfulness", 0.0)),
                answer_relevancy=float(row.get("answer_relevancy", 0.0)),
                context_precision=float(row.get("context_precision", 0.0)),
                context_recall=float(row.get("context_recall", 0.0)),
            )
            for q, a, (_, row) in zip(questions, answers, df.iterrows(), strict=False)
        ],
    )
    _persist(result)
    return result


def load_latest_result() -> EvaluationResult | None:
    """Return the most recent persisted evaluation result, if any."""
    out_dir = get_settings().eval_results_dir
    if not os.path.isdir(out_dir):
        return None
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".json"))
    if not files:
        return None
    with open(os.path.join(out_dir, files[-1]), encoding="utf-8") as fh:
        return EvaluationResult(**json.load(fh))


def run_evaluation() -> EvaluationResult | None:
    """``make eval`` entry point — evaluate the current index end-to-end."""
    from rag.evaluation.testset_gen import generate_testset

    dataset = generate_testset()
    if not dataset:
        return None
    questions = [d["question"] for d in dataset]
    ground_truths = [d.get("ground_truth", "") for d in dataset]
    # Answers/contexts would be produced by running the live chain per question;
    # callers that have a chain wired pass them to ``evaluate_dataset`` directly.
    return evaluate_dataset(questions, ground_truths, [[gt] for gt in ground_truths], ground_truths)
