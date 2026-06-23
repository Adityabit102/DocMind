"""Ingestion orchestration: load → split → embed → index → persist.

Also owns the on-disk document registry (``metadata.json``) that records one
``DocumentRecord`` per ingested file. Hash-based dedup short-circuits identical
re-uploads.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from langchain_core.documents import Document

from app.config import get_settings
from app.models.document import Classification, DocumentRecord, DocumentStatus
from rag.ingestion.deduplicator import file_hash
from rag.ingestion.indexer import IndexStore
from rag.ingestion.loader import load_file
from rag.ingestion.splitter import split_documents


# ── Registry (metadata.json) ─────────────────────────────────────────
def load_registry() -> dict[str, DocumentRecord]:
    path = get_settings().metadata_file
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return {k: DocumentRecord(**v) for k, v in raw.items()}


def save_registry(registry: dict[str, DocumentRecord]) -> None:
    path = get_settings().metadata_file
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({k: json.loads(v.model_dump_json()) for k, v in registry.items()}, fh, indent=2)


def known_hashes(registry: dict[str, DocumentRecord]) -> set[str]:
    return {r.hash for r in registry.values() if r.hash}


# ── Ingestion ────────────────────────────────────────────────────────
def _stamp_document_id(chunks: list[Document], document_id: str, filename: str) -> None:
    for chunk in chunks:
        chunk.metadata["document_id"] = document_id
        chunk.metadata.setdefault("filename", filename)


def ingest_file(
    path: str,
    store: IndexStore,
    registry: dict[str, DocumentRecord] | None = None,
    tags: list[str] | None = None,
    classification: Classification = Classification.INTERNAL,
    owner_id: str | None = None,
) -> DocumentRecord:
    """Ingest a single file into ``store`` and update the registry.

    Returns the resulting ``DocumentRecord`` (status ``indexed`` or ``failed``,
    or an existing record when the file is an exact duplicate). Re-uploading a
    file whose content changed creates a new *version* that supersedes the prior
    one (whose chunks are dropped from the index).
    """
    settings = get_settings()
    registry = load_registry() if registry is None else registry
    filename = os.path.basename(path)
    digest = file_hash(path)

    # Exact-duplicate short-circuit (same bytes, already indexed).
    for existing in registry.values():
        if existing.hash == digest and existing.status == DocumentStatus.INDEXED:
            return existing

    # Versioning: a prior *latest* record for this filename with different bytes
    # is superseded by this upload.
    prior = next(
        (
            r
            for r in registry.values()
            if r.filename == filename and r.is_latest and r.status == DocumentStatus.INDEXED
        ),
        None,
    )

    record = DocumentRecord(
        filename=filename,
        file_type=os.path.splitext(filename)[1].lstrip(".").lower(),
        file_size_bytes=os.path.getsize(path),
        tags=tags or (prior.tags if prior else []),
        embedding_model=settings.embedding_model,
        classification=classification,
        hash=digest,
        status=DocumentStatus.PROCESSING,
        version=(prior.version + 1) if prior else 1,
        supersedes=prior.id if prior else None,
        owner_id=owner_id or (prior.owner_id if prior else None),
    )
    try:
        docs = load_file(path)
        # Optional vision captioning of embedded images (PDF only; no-op if off).
        if settings.enable_image_captioning and record.file_type == "pdf":
            from rag.ingestion.image_caption import caption_pdf_images

            docs = docs + caption_pdf_images(path, settings)
        chunks = split_documents(docs, settings)
        _stamp_document_id(chunks, record.id, filename)
        store.add_documents(chunks)
        store.save()

        record.page_count = len({d.metadata.get("page_number", 1) for d in docs})
        record.word_count = sum(len(d.page_content.split()) for d in docs)
        record.chunk_count = len(chunks)
        record.status = DocumentStatus.INDEXED
        record.indexed_at = datetime.now(timezone.utc)
        # Retire the superseded version: drop its chunks and flag it stale.
        if prior is not None:
            store.delete_document(prior.id)
            prior.is_latest = False
            prior.superseded_by = record.id
            registry[prior.id] = prior
    except Exception as exc:  # noqa: BLE001 — surface failure to UI, keep serving
        record.status = DocumentStatus.FAILED
        record.error = str(exc)

    registry[record.id] = record
    save_registry(registry)
    return record


def ingest_directory(directory: str, store: IndexStore | None = None) -> list[DocumentRecord]:
    """Ingest every supported file in ``directory`` (used by ``make ingest``)."""
    from rag.ingestion.loader import is_supported

    store = store or IndexStore().load()
    registry = load_registry()
    records: list[DocumentRecord] = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and is_supported(path):
            records.append(ingest_file(path, store, registry))
    return records


def reconcile_index(store: IndexStore) -> int:
    """Drop index chunks whose document is no longer in the registry.

    Keeps the FAISS/BM25 index consistent with ``metadata.json``: orphaned chunks
    (e.g. left by an interrupted delete) are invisible in the UI yet still pollute
    retrieval, so we prune them on startup. Returns the number removed.
    """
    registry = load_registry()
    known = set(registry.keys())
    before = len(store.chunks)
    orphans = {
        c.metadata.get("document_id")
        for c in store.chunks
        if c.metadata.get("document_id") not in known
    }
    if not orphans:
        return 0
    store.chunks = [c for c in store.chunks if c.metadata.get("document_id") in known]
    store._rebuild_vectorstore()
    store.save()
    return before - len(store.chunks)


def delete_document(document_id: str, store: IndexStore) -> bool:
    """Remove a document from the index and registry. Returns True if found."""
    registry = load_registry()
    if document_id not in registry:
        return False
    store.delete_document(document_id)
    store.save()
    del registry[document_id]
    save_registry(registry)
    return True
