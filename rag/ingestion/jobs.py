"""Durable ingestion job queue with a background worker.

Uploads enqueue an :class:`IngestionJob` that is persisted to ``jobs.json`` and
processed by a single daemon worker thread. Persistence makes progress pollable
(``/documents/jobs/{id}``) and durable: jobs left ``queued``/``processing`` when
the process died are re-queued on startup, so an interrupted batch resumes
instead of vanishing (the gap with FastAPI ``BackgroundTasks``).

Deliberately dependency-free (stdlib ``queue`` + a thread) so the local-first
stack needs no Celery/Redis worker. A thread lock guards the registry; the store
is mutated only by the single worker, matching its internal locking.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading

from app.config import get_settings
from app.models.jobs import IngestionJob, JobStatus

logger = logging.getLogger("docmind")

_jobs: dict[str, IngestionJob] = {}
_lock = threading.Lock()
_work: queue.Queue[str] = queue.Queue()
_worker: threading.Thread | None = None
_store_provider = None  # set by start_worker so the worker can reach the index


def _jobs_path() -> str:
    return os.path.join(os.path.dirname(get_settings().metadata_file) or ".", "jobs.json")


def _persist() -> None:
    path = _jobs_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with _lock:
        snapshot = {k: json.loads(v.model_dump_json()) for k, v in _jobs.items()}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, indent=2)


def _load() -> None:
    path = _jobs_path()
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    with _lock:
        _jobs.clear()
        for k, v in raw.items():
            _jobs[k] = IngestionJob(**v)


def _update(job_id: str, **fields) -> None:  # type: ignore[no-untyped-def]
    from datetime import datetime, timezone

    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        for key, value in fields.items():
            setattr(job, key, value)
        job.updated_at = datetime.now(timezone.utc)
    _persist()


def enqueue(path: str, tags: list[str] | None = None, owner_id: str | None = None) -> IngestionJob:
    """Register and queue a file for ingestion. Returns the created job."""
    job = IngestionJob(
        filename=os.path.basename(path), path=path, tags=tags or [], owner_id=owner_id
    )
    with _lock:
        _jobs[job.id] = job
    _persist()
    _work.put(job.id)
    return job


def get_job(job_id: str) -> IngestionJob | None:
    with _lock:
        return _jobs.get(job_id)


def list_jobs() -> list[IngestionJob]:
    with _lock:
        return sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)


def _process(job_id: str) -> None:
    from rag.ingestion.pipeline import ingest_file, load_registry

    job = get_job(job_id)
    if job is None:
        return
    if not os.path.exists(job.path):
        _update(job_id, status=JobStatus.FAILED, error="file no longer available")
        return
    _update(job_id, status=JobStatus.PROCESSING, progress=0.1)
    try:
        store = _store_provider() if _store_provider else None
        registry = load_registry()
        record = ingest_file(
            job.path, store, registry, tags=job.tags, owner_id=job.owner_id
        )
        if record.status.value == "failed":
            _update(job_id, status=JobStatus.FAILED, error=record.error, progress=1.0)
        else:
            _update(
                job_id,
                status=JobStatus.COMPLETED,
                document_id=record.id,
                progress=1.0,
            )
    except Exception as exc:  # noqa: BLE001 — record failure, keep worker alive
        logger.warning("Ingestion job %s failed: %s", job_id, exc)
        _update(job_id, status=JobStatus.FAILED, error=str(exc), progress=1.0)


def _run() -> None:
    while True:
        job_id = _work.get()
        try:
            _process(job_id)
        finally:
            _work.task_done()


def start_worker(store_provider) -> None:  # type: ignore[no-untyped-def]
    """Start the daemon worker and re-queue any unfinished jobs from disk."""
    global _worker, _store_provider
    _store_provider = store_provider
    _load()
    # Recovery: anything not finished when we died goes back on the queue.
    with _lock:
        pending = [
            j.id for j in _jobs.values() if j.status in (JobStatus.QUEUED, JobStatus.PROCESSING)
        ]
    for job_id in pending:
        _update(job_id, status=JobStatus.QUEUED, progress=0.0)
        _work.put(job_id)
    if pending:
        logger.info("Re-queued %d unfinished ingestion job(s)", len(pending))
    if _worker is None or not _worker.is_alive():
        _worker = threading.Thread(target=_run, name="ingestion-worker", daemon=True)
        _worker.start()
