"""SHA-256 based duplicate-file detection.

Hashes raw file bytes so an identical upload (any filename) is detected and the
re-index is skipped. Hashes of indexed documents are tracked in the registry.
"""

from __future__ import annotations

import hashlib


def file_hash(path: str, chunk_bytes: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file's contents, prefixed ``sha256:``."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_bytes), b""):
            h.update(block)
    return f"sha256:{h.hexdigest()}"


def bytes_hash(data: bytes) -> str:
    """SHA-256 of an in-memory byte string, prefixed ``sha256:``."""
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def is_duplicate(candidate_hash: str, known_hashes: set[str]) -> bool:
    """True if ``candidate_hash`` already exists in the known set."""
    return candidate_hash in known_hashes
