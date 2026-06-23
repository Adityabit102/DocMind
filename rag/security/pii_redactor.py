"""PII detection and redaction.

Prefers Microsoft Presidio when installed; otherwise applies a regex fallback for
the most common identifiers (email, phone, SSN, credit card). Redaction is opt-in
via ``ENABLE_PII_REDACTION`` and applied to model output before it is returned.
"""

from __future__ import annotations

import re

from app.config import get_settings

_PATTERNS = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE": re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
}


def _regex_redact(text: str) -> str:
    for label, pattern in _PATTERNS.items():
        text = pattern.sub(f"[REDACTED_{label}]", text)
    return text


def redact(text: str, force: bool = False) -> str:
    """Redact PII from ``text`` when redaction is enabled (or ``force``)."""
    if not force and not get_settings().enable_pii_redaction:
        return text
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()
        results = analyzer.analyze(text=text, language="en")
        return anonymizer.anonymize(text=text, analyzer_results=results).text
    except Exception:  # noqa: BLE001 — Presidio absent/failed → regex fallback
        return _regex_redact(text)


def contains_pii(text: str) -> bool:
    """Quick boolean check using the regex patterns."""
    return any(pattern.search(text) for pattern in _PATTERNS.values())
