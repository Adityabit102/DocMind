"""LLM-as-judge evaluation — real semantic metrics without RAGAS/OpenAI.

Uses the configured chat model (e.g. Groq) to grade each answer the way RAGAS
would, but in a single JSON call per question so it stays cheap and fast. Scores
are semantic (the model reasons about support and relevance) rather than the
lexical overlap of the dependency-free fallback. Parsing is defensive: a bad
verdict for one question degrades to neutral rather than failing the run.
"""

from __future__ import annotations

import json
import re

from app.models.evaluation import EvaluationResult, PerQuestionResult

_PROMPT = """You are a strict RAG evaluation judge. Score the answer on four \
metrics from 0.0 to 1.0.

- faithfulness: are ALL claims in the answer supported by the context?
- answer_relevancy: does the answer directly address the question?
- context_precision: how much of the context is relevant to the question?
- context_recall: does the context contain the information in the ground truth?

Question: {question}
Context: {context}
Answer: {answer}
Ground truth: {ground_truth}

Reply with ONLY a compact JSON object, no prose:
{{"faithfulness": <float>, "answer_relevancy": <float>, "context_precision": <float>, "context_recall": <float>}}"""


def _parse_scores(raw: str) -> dict[str, float]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    out: dict[str, float] = {}
    for key in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        try:
            out[key] = max(0.0, min(1.0, float(data.get(key, 0.0))))
        except (TypeError, ValueError):
            out[key] = 0.0
    return out


def llm_judge_evaluate(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
    llm,  # type: ignore[no-untyped-def]
    model_name: str = "",
) -> EvaluationResult:
    """Grade a dataset with the LLM judge; one call per question."""
    from rag.evaluation.grounding import mean_usr

    per: list[PerQuestionResult] = []
    for q, a, ctx, gt in zip(questions, answers, contexts, ground_truths, strict=False):
        prompt = _PROMPT.format(
            question=q, context="\n".join(ctx)[:4000], answer=a, ground_truth=gt
        )
        try:
            result = llm.invoke(prompt)
            scores = _parse_scores(getattr(result, "content", str(result)))
        except Exception:  # noqa: BLE001 — judge failure → neutral row
            scores = {}
        per.append(
            PerQuestionResult(
                question=q,
                answer=a,
                faithfulness=scores.get("faithfulness", 0.0),
                answer_relevancy=scores.get("answer_relevancy", 0.0),
                context_precision=scores.get("context_precision", 0.0),
                context_recall=scores.get("context_recall", 0.0),
            )
        )

    def _avg(attr: str) -> float:
        vals = [getattr(p, attr) for p in per]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return EvaluationResult(
        engine=f"llm-judge:{model_name}" if model_name else "llm-judge",
        question_count=len(questions),
        faithfulness=_avg("faithfulness"),
        answer_relevancy=_avg("answer_relevancy"),
        context_precision=_avg("context_precision"),
        context_recall=_avg("context_recall"),
        unsupported_sentence_ratio=mean_usr(answers, contexts),
        per_question=per,
    )
