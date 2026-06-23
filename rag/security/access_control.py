"""Document classification + role-based retrieval trimming.

Maps roles to the classification levels they may read, and filters retrieved
chunks so users only ever see authorised content (security trimming at retrieval
time). Classification travels in chunk metadata for full provenance/auditability.
"""

from __future__ import annotations

from langchain_core.documents import Document

from app.models.document import Classification

# Which classification levels each role is permitted to read.
_ROLE_PERMISSIONS: dict[str, set[Classification]] = {
    "admin": {Classification.PUBLIC, Classification.INTERNAL, Classification.CONFIDENTIAL},
    "user": {Classification.PUBLIC, Classification.INTERNAL},
    "guest": {Classification.PUBLIC},
}


def allowed_levels(role: str) -> set[Classification]:
    """Return the classification levels a role may access (default: guest)."""
    return _ROLE_PERMISSIONS.get(role, _ROLE_PERMISSIONS["guest"])


def can_access(role: str, classification: Classification) -> bool:
    return classification in allowed_levels(role)


def security_trim(docs: list[Document], role: str) -> list[Document]:
    """Drop chunks whose classification the role is not permitted to read."""
    permitted = allowed_levels(role)
    out: list[Document] = []
    for doc in docs:
        raw = doc.metadata.get("classification", Classification.INTERNAL.value)
        try:
            level = Classification(raw)
        except ValueError:
            level = Classification.INTERNAL
        if level in permitted:
            out.append(doc)
    return out
