"""Parent-document expansion for the ParentDocumentRetriever pattern.

Children are what get indexed and matched (small, precise); parents are what we
hand to the LLM (larger, fuller context). After retrieval we swap each matched
child for its parent passage, de-duplicating so a parent that produced several
matching children appears once. Chunks without a ``parent_id`` pass through
unchanged, so this is a no-op for non-parent indexes.
"""

from __future__ import annotations

from langchain_core.documents import Document

from rag.ingestion.indexer import IndexStore


def expand_to_parents(docs: list[Document], store: IndexStore) -> list[Document]:
    """Replace matched child chunks with their (deduplicated) parent passages."""
    out: list[Document] = []
    seen_parents: set[str] = set()
    for doc in docs:
        parent_id = doc.metadata.get("parent_id")
        parent_content = doc.metadata.get("parent_content")
        if not parent_id or not parent_content:
            out.append(doc)
            continue
        if parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)
        meta = {k: v for k, v in doc.metadata.items() if k != "parent_content"}
        meta["is_parent"] = True
        out.append(Document(page_content=parent_content, metadata=meta))
    return out
