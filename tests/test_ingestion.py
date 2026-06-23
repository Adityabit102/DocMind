"""Ingestion tests: loader routing, chunking, dedup hashing."""

from __future__ import annotations

from rag.ingestion.deduplicator import bytes_hash, file_hash, is_duplicate
from rag.ingestion.loader import SUPPORTED_EXTENSIONS, is_supported
from rag.ingestion.splitter import split_documents


def test_supported_extensions():
    assert is_supported("a.pdf")
    assert is_supported("notes.MD")
    assert not is_supported("archive.zip")
    assert ".csv" in SUPPORTED_EXTENSIONS


def test_file_hash_is_stable(temp_text_file):
    h1 = file_hash(temp_text_file)
    h2 = file_hash(temp_text_file)
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_bytes_hash_and_dedup():
    h = bytes_hash(b"hello world")
    assert is_duplicate(h, {h})
    assert not is_duplicate(h, {"sha256:other"})


def test_recursive_splitter_stamps_metadata(sample_docs):
    chunks = split_documents(sample_docs, strategy="recursive")
    assert chunks, "expected at least one chunk"
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == i
        assert "char_offset" in chunk.metadata
        assert chunk.metadata["char_count"] == len(chunk.page_content)
        # source metadata is preserved for citations
        assert "filename" in chunk.metadata
