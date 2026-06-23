"""Grounding metrics: Unsupported Sentence Ratio (USR) and an LLM-as-judge score.

USR is the fraction of answer sentences that are *not* supported by the
retrieved context. The lexical estimator is dependency-free and deterministic
(token-overlap per sentence), suitable for every eval run. ``llm_grounding_score``
offers a sharper LLM-as-judge alternative when a model is available.
"""

from __future__ import annotations

import re

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")
# Min fraction of a sentence's content words that must appear in the context
# for the sentence to count as supported.
_SUPPORT_THRESHOLD = 0.5


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def sentence_support(answer: str, contexts: list[str]) -> list[dict]:
    """Per-sentence grounding: ``[{"text", "supported", "overlap"}, ...]``.

    A sentence counts as supported when at least ``_SUPPORT_THRESHOLD`` of its
    content words appear in the retrieved context. Used to highlight unsupported
    claims in the UI and to derive USR.
    """
    sentences = _sentences(answer)
    context_tokens = _tokens(" ".join(contexts))
    out: list[dict] = []
    for sentence in sentences:
        words = _tokens(sentence)
        overlap = (len(words & context_tokens) / len(words)) if words else 1.0
        # Sentences with no content words (e.g. "Here is the answer:") are neutral.
        supported = (not words) or overlap >= _SUPPORT_THRESHOLD
        out.append(
            {"text": sentence, "supported": supported, "overlap": round(overlap, 3)}
        )
    return out


def lexical_usr(answer: str, contexts: list[str]) -> float:
    """Fraction of answer sentences not lexically supported by the context."""
    spans = sentence_support(answer, contexts)
    scored = [s for s in spans if _tokens(s["text"])]
    if not scored:
        return 0.0
    if not _tokens(" ".join(contexts)):
        return 1.0
    unsupported = sum(1 for s in scored if not s["supported"])
    return round(unsupported / len(scored), 4)


def mean_usr(answers: list[str], contexts: list[list[str]]) -> float:
    """Average lexical USR across a batch of (answer, contexts) pairs."""
    if not answers:
        return 0.0
    scores = [lexical_usr(a, c) for a, c in zip(answers, contexts, strict=False)]
    return round(sum(scores) / len(scores), 4)


def llm_grounding_score(answer: str, contexts: list[str], llm: object) -> float:
    """LLM-as-judge grounding score in [0, 1] (1.0 = fully grounded)."""
    from rag.generation.self_rag import reflect

    verdict = reflect(answer, "\n\n".join(contexts), llm)  # type: ignore[arg-type]
    return round(verdict.score, 4)
