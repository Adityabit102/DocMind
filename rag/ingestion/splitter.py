"""Chunking strategies. All preserve character offsets for exact citations."""

from __future__ import annotations

import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings, get_settings

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _recursive_splitter(chunk_size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        add_start_index=True,  # preserves char offset for citation
        separators=_SEPARATORS,
    )


def _semantic_split(docs: list[Document]) -> list[Document]:
    """Split at embedding-distance breakpoints (more coherent chunks)."""
    from langchain_experimental.text_splitter import SemanticChunker

    from rag.ingestion.embedder import get_embeddings

    chunker = SemanticChunker(get_embeddings())
    return chunker.split_documents(docs)


def _stamp(chunks: list[Document]) -> list[Document]:
    """Add chunk_index + char_offset + char_count metadata to every chunk."""
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["char_offset"] = int(chunk.metadata.get("start_index", 0))
        chunk.metadata["char_count"] = len(chunk.page_content)
    return chunks


def _parent_split(docs: list[Document], settings: Settings) -> list[Document]:
    """Parent-document strategy: index small child chunks but carry the larger
    parent passage in each child's metadata, so retrieval can return the parent
    for fuller context while still matching on fine-grained children."""
    parent, child = parent_child_splitters(settings)
    children: list[Document] = []
    for parent_doc in parent.split_documents(docs):
        parent_id = str(uuid.uuid4())
        for child_doc in child.split_documents([parent_doc]):
            child_doc.metadata["parent_id"] = parent_id
            child_doc.metadata["parent_content"] = parent_doc.page_content
            children.append(child_doc)
    return children


def _is_tabular(docs: list[Document]) -> bool:
    """True when the source is row-structured (CSV) — keep one chunk per row."""
    src = docs[0].metadata.get("source", "") if docs else ""
    return src.lower().endswith(".csv")


def split_documents(
    docs: list[Document],
    settings: Settings | None = None,
    strategy: str | None = None,
) -> list[Document]:
    """Chunk ``docs`` using the configured (or overridden) strategy."""
    settings = settings or get_settings()
    strategy = strategy or settings.chunk_strategy

    # Table-aware: CSVLoader already yields one Document per row — preserve that
    # structure instead of re-splitting and merging across row boundaries.
    if _is_tabular(docs):
        return _stamp(docs)

    if strategy == "semantic":
        return _stamp(_semantic_split(docs))

    if strategy == "parent":
        return _stamp(_parent_split(docs, settings))

    splitter = _recursive_splitter(settings.chunk_size, settings.chunk_overlap)
    return _stamp(splitter.split_documents(docs))


def parent_child_splitters(
    settings: Settings | None = None,
) -> tuple[RecursiveCharacterTextSplitter, RecursiveCharacterTextSplitter]:
    """Return (parent, child) splitters for the ParentDocumentRetriever pattern."""
    settings = settings or get_settings()
    parent = _recursive_splitter(settings.chunk_size * 4, settings.chunk_overlap)
    child = _recursive_splitter(settings.chunk_size, settings.chunk_overlap)
    return parent, child
