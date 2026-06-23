"""File-type router that maps an uploaded file to the correct LangChain loader.

Returns a list of ``langchain_core.documents.Document`` with normalised metadata
(``source`` filename and ``page`` number) so downstream chunking can stamp exact
citations. Falls back to OCR for PDFs whose text layer is effectively empty.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

from rag.ingestion.ocr import ocr_pdf, pdf_has_text_layer

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".html", ".htm"}

# Minimum characters/page below which a PDF is treated as scanned → OCR.
_OCR_CHAR_THRESHOLD = 50


def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def is_supported(path: str) -> bool:
    """True if the file extension is one DocMind can ingest."""
    return _ext(path) in SUPPORTED_EXTENSIONS


def _load_pdf(path: str) -> list[Document]:
    """Load a PDF; OCR-fallback when the text layer is empty/scanned."""
    docs = PyPDFLoader(path).load()
    avg_chars = (
        sum(len(d.page_content) for d in docs) / len(docs) if docs else 0
    )
    if not docs or avg_chars < _OCR_CHAR_THRESHOLD or not pdf_has_text_layer(docs):
        return ocr_pdf(path)
    return docs


def _load_text(path: str) -> list[Document]:
    return TextLoader(path, autodetect_encoding=True).load()


_LOADERS: dict[str, Callable[[str], list[Document]]] = {
    ".pdf": _load_pdf,
    ".docx": lambda p: Docx2txtLoader(p).load(),
    ".txt": _load_text,
    ".md": _load_text,
    ".csv": lambda p: CSVLoader(p).load(),
    ".html": lambda p: BSHTMLLoader(p).load(),
    ".htm": lambda p: BSHTMLLoader(p).load(),
}


def load_file(path: str) -> list[Document]:
    """Load a single file into LangChain ``Document`` objects.

    Raises ``ValueError`` for unsupported types.
    """
    ext = _ext(path)
    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    docs = loader(path)
    filename = os.path.basename(path)
    for i, doc in enumerate(docs):
        doc.metadata.setdefault("source", filename)
        doc.metadata["filename"] = filename
        # PyPDFLoader sets 0-based "page"; normalise to 1-based for citations.
        if "page" in doc.metadata:
            doc.metadata["page_number"] = int(doc.metadata["page"]) + 1
        else:
            doc.metadata["page_number"] = i + 1
    return docs
