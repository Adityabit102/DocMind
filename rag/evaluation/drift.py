"""Embedding-distribution drift detection via Evidently AI.

Compares the embedding distribution of recent queries/chunks against a baseline.
A drift score above the configured threshold flags that inputs have shifted and
retrieval quality may degrade. Evidently is imported lazily; a numpy-only
fallback (mean-vector cosine distance) keeps the check working without it.
"""

from __future__ import annotations

from app.config import get_settings
from app.models.evaluation import DriftStatus


def _fallback_drift(reference: list[list[float]], current: list[list[float]]) -> float:
    """Cosine distance between mean embedding vectors (cheap proxy for drift)."""
    import numpy as np

    if not reference or not current:
        return 0.0
    ref = np.asarray(reference).mean(axis=0)
    cur = np.asarray(current).mean(axis=0)
    denom = float(np.linalg.norm(ref) * np.linalg.norm(cur)) or 1.0
    cosine = float(np.dot(ref, cur)) / denom
    return max(0.0, 1.0 - cosine)


def detect_drift(
    reference: list[list[float]], current: list[list[float]]
) -> DriftStatus:
    """Return a ``DriftStatus`` comparing current vs baseline embeddings."""
    settings = get_settings()
    threshold = settings.drift_alert_threshold

    try:
        import pandas as pd
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        ref_df = pd.DataFrame(reference)
        cur_df = pd.DataFrame(current)
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_df, current_data=cur_df)
        summary = report.as_dict()
        share = (
            summary["metrics"][0]["result"]
            .get("share_of_drifted_columns", 0.0)
        )
        score = float(share)
    except Exception:  # noqa: BLE001 — Evidently absent/failed → numeric fallback
        score = _fallback_drift(reference, current)

    detected = score > threshold
    return DriftStatus(
        drift_detected=detected,
        drift_score=round(score, 4),
        threshold=threshold,
        message="Drift detected" if detected else "OK",
    )
