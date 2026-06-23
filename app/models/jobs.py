"""Pydantic schema for ingestion jobs (durable, pollable upload tracking)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestionJob(BaseModel):
    """One queued file ingestion, persisted so progress survives a restart."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    path: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0  # 0.0 – 1.0
    document_id: str | None = None
    error: str | None = None
    tags: list[str] = Field(default_factory=list)
    owner_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
