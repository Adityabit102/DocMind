"""Self-RAG reflection: judge whether a generated answer is grounded in context.

A lightweight post-generation reflection step. The LLM grades its own answer
against the retrieved context; a low grounding score lets the chain abstain or
retry rather than surface an unsupported answer. Parsing is defensive so a
malformed verdict never breaks the request — it degrades to "assume grounded".
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "prompts")

_GROUNDED_RE = re.compile(r"grounded\s*:\s*(yes|no)", re.IGNORECASE)
_SCORE_RE = re.compile(r"score\s*:\s*([01](?:\.\d+)?)", re.IGNORECASE)
_REASON_RE = re.compile(r"reason\s*:\s*(.+)$", re.IGNORECASE | re.DOTALL)


@dataclass
class GroundingVerdict:
    grounded: bool
    score: float
    reason: str = ""


def _load_prompt(name: str) -> str:
    with open(os.path.join(_PROMPTS_DIR, name), encoding="utf-8") as fh:
        return fh.read()


def reflect(answer: str, context: str, llm: BaseLanguageModel) -> GroundingVerdict:
    """Grade ``answer`` against ``context`` for grounding via the LLM."""
    template = PromptTemplate.from_template(_load_prompt("self_rag.txt"))
    chain = template | llm | StrOutputParser()
    try:
        raw = chain.invoke({"context": context, "answer": answer})
    except Exception:  # noqa: BLE001 — reflection is best-effort
        return GroundingVerdict(grounded=True, score=1.0, reason="reflection unavailable")

    grounded_m = _GROUNDED_RE.search(raw)
    score_m = _SCORE_RE.search(raw)
    reason_m = _REASON_RE.search(raw)
    grounded = grounded_m.group(1).lower() == "yes" if grounded_m else True
    score = float(score_m.group(1)) if score_m else (1.0 if grounded else 0.0)
    reason = reason_m.group(1).strip()[:200] if reason_m else ""
    return GroundingVerdict(grounded=grounded, score=score, reason=reason)
