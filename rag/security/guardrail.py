"""Prompt-injection / jailbreak guardrail for inbound queries.

A fast heuristic screen that runs before retrieval when ``ENABLE_PROMPT_GUARD``
is set. It flags the well-known injection/jailbreak patterns ("ignore previous
instructions", "you are now DAN", system-prompt exfiltration, etc.). It is
deliberately conservative — a signal, not a content filter — and pairs with the
grounded-answer design that already constrains the model to retrieved context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Each pattern is a compiled, case-insensitive regex for a known attack shape.
_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ignore_instructions", re.compile(r"\bignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|earlier)\b.*\binstructions?\b", re.I)),
    ("disregard_instructions", re.compile(r"\b(?:disregard|forget|override)\b.*\b(?:instructions?|rules?|prompt)\b", re.I)),
    ("reveal_system_prompt", re.compile(r"\b(?:reveal|show|print|repeat|leak|expose)\b.*\b(?:system\s+prompt|your\s+instructions?|initial\s+prompt)\b", re.I)),
    ("role_override", re.compile(r"\byou\s+are\s+now\b|\bact\s+as\b.*\b(?:DAN|jailbreak|unfiltered|unrestricted)\b|\bpretend\s+(?:you|to\s+be)\b", re.I)),
    ("dev_mode", re.compile(r"\b(?:developer|debug|god)\s+mode\b|\bjailbreak\b", re.I)),
    ("bypass_safety", re.compile(r"\b(?:bypass|disable|turn\s+off)\b.*\b(?:safety|guardrails?|filters?|restrictions?)\b", re.I)),
    ("exfiltrate", re.compile(r"\bprint\b.*\b(?:everything\s+above|all\s+prior\s+text)\b", re.I)),
]


@dataclass
class GuardVerdict:
    blocked: bool
    reasons: list[str]

    @property
    def message(self) -> str:
        return "Query blocked by the prompt-injection guardrail: " + ", ".join(self.reasons)


def screen_query(text: str) -> GuardVerdict:
    """Return a verdict flagging any injection/jailbreak patterns in ``text``."""
    reasons = [name for name, pattern in _INJECTION_PATTERNS if pattern.search(text)]
    return GuardVerdict(blocked=bool(reasons), reasons=reasons)
