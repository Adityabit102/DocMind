"""Persistent dual index: FAISS (dense) + BM25 (sparse).

``IndexStore`` owns the canonical list of indexed chunks and derives both the
FAISS vector index and the BM25 keyword index from it. Add/delete operations
rebuild the indices from the surviving chunks, which keeps the two stores
consistent and gives free auto-recovery (rebuild from chunk metadata) when the
on-disk FAISS files are missing or corrupt.
"""

from __future__ import annotations

import os
import pickle
import threading

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import get_settings
from rag.ingestion.embedder import get_embeddings

_CHUNKS_FILE = "chunks.pkl"  # canonical chunk list, used to rebuild BM25 + FAISS


class IndexStore:
    """In-process holder of the FAISS vectorstore plus the raw chunk corpus."""

    def __init__(self, embeddings: Embeddings | None = None, index_dir: str | None = None):
        # Embeddings are built lazily — an empty index never loads the model.
        self._embeddings = embeddings
        self.index_dir = index_dir or get_settings().faiss_index_dir
        self.chunks: list[Document] = []
        self.vectorstore: FAISS | None = None
        self._lock = threading.Lock()

    @property
    def embeddings(self) -> Embeddings:
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    # ── State ────────────────────────────────────────────────────────
    @property
    def is_empty(self) -> bool:
        return not self.chunks

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def document_ids(self) -> set[str]:
        return {c.metadata.get("document_id", "") for c in self.chunks} - {""}

    # ── Mutations ────────────────────────────────────────────────────
    def add_documents(self, chunks: list[Document]) -> None:
        """Append chunks and (re)build the FAISS index."""
        if not chunks:
            return
        with self._lock:
            self.chunks.extend(chunks)
            self._rebuild_vectorstore()

    def delete_document(self, document_id: str) -> int:
        """Remove all chunks of one document; rebuild. Returns chunks removed."""
        with self._lock:
            before = len(self.chunks)
            self.chunks = [
                c for c in self.chunks if c.metadata.get("document_id") != document_id
            ]
            removed = before - len(self.chunks)
            self._rebuild_vectorstore()
            return removed

    def chunks_for(self, document_id: str) -> list[Document]:
        return [c for c in self.chunks if c.metadata.get("document_id") == document_id]

    def _rebuild_vectorstore(self) -> None:
        self.vectorstore = (
            FAISS.from_documents(self.chunks, self.embeddings) if self.chunks else None
        )

    # ── Persistence ──────────────────────────────────────────────────
    def save(self) -> None:
        os.makedirs(self.index_dir, exist_ok=True)
        with open(os.path.join(self.index_dir, _CHUNKS_FILE), "wb") as fh:
            pickle.dump(self.chunks, fh)
        if self.vectorstore is not None:
            self.vectorstore.save_local(self.index_dir)

    def load(self) -> "IndexStore":
        """Load chunks (and FAISS, rebuilding it if the index file is bad)."""
        chunks_path = os.path.join(self.index_dir, _CHUNKS_FILE)
        if not os.path.exists(chunks_path):
            return self
        with open(chunks_path, "rb") as fh:
            self.chunks = pickle.load(fh)
        try:
            self.vectorstore = FAISS.load_local(
                self.index_dir,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception:  # noqa: BLE001 — corrupt/missing index → rebuild from chunks
            self._rebuild_vectorstore()
        return self
