"""Pydantic schemas for documents and their chunks."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """Lifecycle state of an indexed document."""

    QUEUED = "queued"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class Classification(str, Enum):
    """Access-control classification level."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"


class DocumentRecord(BaseModel):
    """Registry entry for one ingested document."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    file_type: str
    file_size_bytes: int = 0
    page_count: int = 0
    word_count: int = 0
    chunk_count: int = 0
    tags: list[str] = Field(default_factory=list)
    indexed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    embedding_model: str = "all-MiniLM-L6-v2"
    status: DocumentStatus = DocumentStatus.QUEUED
    classification: Classification = Classification.INTERNAL
    hash: str = ""
    error: str | None = None
    # Versioning: a re-upload of the same filename with new content supersedes
    # the prior record. Only the latest version is retrieved against.
    version: int = 1
    is_latest: bool = True
    supersedes: str | None = None
    superseded_by: str | None = None
    owner_id: str | None = None


class ChunkPreview(BaseModel):
    """A single chunk surfaced for preview / inspection."""

    chunk_id: str
    document_id: str
    filename: str
    page_number: int = 0
    chunk_index: int = 0
    char_offset: int = 0
    char_count: int = 0
    text: str


class UploadResponse(BaseModel):
    """Returned by the async upload endpoint."""

    job_id: str = Field(default_factory=lambda: str(uuid4()))
    job_ids: list[str] = Field(default_factory=list)
    documents: list[DocumentRecord] = Field(default_factory=list)
    message: str = "Ingestion started"


class TagUpdate(BaseModel):
    """Body for PATCH /documents/{id}/tags."""

    tags: list[str]
