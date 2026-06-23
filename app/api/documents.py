"""Document management endpoints: upload, list, get, delete, chunks, tags, download."""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.auth import UserPublic, optional_user
from app.config import Settings
from app.dependencies import get_store, settings_dep
from app.models.document import (
    ChunkPreview,
    DocumentRecord,
    TagUpdate,
    UploadResponse,
)
from app.models.jobs import IngestionJob
from rag.ingestion.indexer import IndexStore
from rag.ingestion.jobs import enqueue, get_job, list_jobs
from rag.ingestion.loader import SUPPORTED_EXTENSIONS, is_supported
from rag.ingestion.pipeline import (
    delete_document,
    ingest_file,
    load_registry,
    save_registry,
)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


def _authorize(record: DocumentRecord, user: UserPublic | None) -> None:
    """Reject access to a document owned by a different user (auth on only)."""
    if user is not None and record.owner_id not in (None, user.id):
        raise HTTPException(status_code=404, detail="Document not found")


@router.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: list[UploadFile],
    settings: Settings = Depends(settings_dep),
    user: UserPublic | None = Depends(optional_user),
) -> UploadResponse:
    """Validate, persist, and queue one or more documents for durable ingestion."""
    os.makedirs(settings.upload_dir, exist_ok=True)
    records: list[DocumentRecord] = []
    job_ids: list[str] = []
    owner_id = user.id if user else None
    for upload in files:
        if not is_supported(upload.filename or ""):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
            )
        data = await upload.read()
        if len(data) > settings.max_upload_size_bytes:
            raise HTTPException(status_code=413, detail="File exceeds size limit")
        dest = os.path.join(settings.upload_dir, upload.filename)  # type: ignore[arg-type]
        with open(dest, "wb") as fh:
            fh.write(data)
        records.append(
            DocumentRecord(
                filename=upload.filename or "unknown",
                file_type=os.path.splitext(upload.filename or "")[1].lstrip(".").lower(),
                file_size_bytes=len(data),
            )
        )
        job_ids.append(enqueue(dest, owner_id=owner_id).id)
    return UploadResponse(job_ids=job_ids, documents=records, message="Ingestion queued")


@router.get("/jobs", response_model=list[IngestionJob])
def list_ingestion_jobs() -> list[IngestionJob]:
    """All ingestion jobs (most recent first) for upload-progress polling."""
    return list_jobs()


@router.get("/jobs/{job_id}", response_model=IngestionJob)
def get_ingestion_job(job_id: str) -> IngestionJob:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("", response_model=list[DocumentRecord])
def list_documents(
    include_superseded: bool = False,
    user: UserPublic | None = Depends(optional_user),
) -> list[DocumentRecord]:
    """List documents; by default only the latest version of each file.

    When auth is enabled, results are scoped to the caller's own documents
    (plus any unowned/shared ones).
    """
    docs = list(load_registry().values())
    if not include_superseded:
        docs = [d for d in docs if d.is_latest]
    if user is not None:
        docs = [d for d in docs if d.owner_id in (None, user.id)]
    return docs


@router.get("/{doc_id}", response_model=DocumentRecord)
def get_document(
    doc_id: str, user: UserPublic | None = Depends(optional_user)
) -> DocumentRecord:
    registry = load_registry()
    if doc_id not in registry:
        raise HTTPException(status_code=404, detail="Document not found")
    _authorize(registry[doc_id], user)
    return registry[doc_id]


@router.delete("/{doc_id}")
def remove_document(
    doc_id: str,
    store: IndexStore = Depends(get_store),
    user: UserPublic | None = Depends(optional_user),
) -> dict[str, str]:
    registry = load_registry()
    if doc_id in registry:
        _authorize(registry[doc_id], user)
    if not delete_document(doc_id, store):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "deleted", "id": doc_id}


@router.get("/{doc_id}/chunks", response_model=list[ChunkPreview])
def get_chunks(
    doc_id: str,
    store: IndexStore = Depends(get_store),
    user: UserPublic | None = Depends(optional_user),
) -> list[ChunkPreview]:
    registry = load_registry()
    if doc_id not in registry:
        raise HTTPException(status_code=404, detail="Document not found")
    _authorize(registry[doc_id], user)
    previews: list[ChunkPreview] = []
    for chunk in store.chunks_for(doc_id):
        previews.append(
            ChunkPreview(
                chunk_id=str(chunk.metadata.get("chunk_index", 0)),
                document_id=doc_id,
                filename=chunk.metadata.get("filename", ""),
                page_number=chunk.metadata.get("page_number", 0),
                chunk_index=chunk.metadata.get("chunk_index", 0),
                char_offset=chunk.metadata.get("char_offset", 0),
                char_count=len(chunk.page_content),
                text=chunk.page_content,
            )
        )
    return previews


@router.post("/{doc_id}/reindex")
def reindex_document(
    doc_id: str,
    store: IndexStore = Depends(get_store),
    settings: Settings = Depends(settings_dep),
    user: UserPublic | None = Depends(optional_user),
) -> dict[str, str]:
    registry = load_registry()
    if doc_id not in registry:
        raise HTTPException(status_code=404, detail="Document not found")
    _authorize(registry[doc_id], user)
    record = registry[doc_id]
    path = os.path.join(settings.upload_dir, record.filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=410, detail="Original file no longer available")
    store.delete_document(doc_id)
    ingest_file(path, store, registry, tags=record.tags, classification=record.classification)
    return {"status": "reindexed", "id": doc_id}


@router.get("/{doc_id}/download")
def download_document(
    doc_id: str,
    settings: Settings = Depends(settings_dep),
    user: UserPublic | None = Depends(optional_user),
) -> FileResponse:
    registry = load_registry()
    if doc_id not in registry:
        raise HTTPException(status_code=404, detail="Document not found")
    _authorize(registry[doc_id], user)
    path = os.path.join(settings.upload_dir, registry[doc_id].filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=410, detail="Original file no longer available")
    return FileResponse(path, filename=registry[doc_id].filename)


@router.patch("/{doc_id}/tags", response_model=DocumentRecord)
def update_tags(
    doc_id: str, body: TagUpdate, user: UserPublic | None = Depends(optional_user)
) -> DocumentRecord:
    registry = load_registry()
    if doc_id not in registry:
        raise HTTPException(status_code=404, detail="Document not found")
    _authorize(registry[doc_id], user)
    registry[doc_id].tags = body.tags
    save_registry(registry)
    return registry[doc_id]
