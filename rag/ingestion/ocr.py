"""OCR fallback for scanned PDFs.

Uses ``pdf2image`` to rasterise pages and ``pytesseract`` (Tesseract) to extract
text. Imports of the heavy libraries are deferred so the module can be imported
even where Tesseract / poppler are not installed.
"""

from __future__ import annotations

import os

from langchain_core.documents import Document

_MIN_TEXT_LAYER_CHARS = 50


def pdf_has_text_layer(docs: list[Document]) -> bool:
    """Heuristic: a real text layer has more than a trivial number of chars."""
    total = sum(len(d.page_content.strip()) for d in docs)
    return total >= _MIN_TEXT_LAYER_CHARS


def _preprocess(image):  # type: ignore[no-untyped-def]
    """Light denoise + grayscale to improve OCR accuracy."""
    from PIL import ImageOps

    gray = ImageOps.grayscale(image)
    return ImageOps.autocontrast(gray)


def ocr_pdf(path: str) -> list[Document]:
    """Rasterise each PDF page and OCR it into one ``Document`` per page."""
    from pdf2image import convert_from_path
    from pytesseract import image_to_string

    filename = os.path.basename(path)
    pages = convert_from_path(path)
    documents: list[Document] = []
    for idx, image in enumerate(pages):
        text = image_to_string(_preprocess(image))
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "filename": filename,
                    "page_number": idx + 1,
                    "ocr": True,
                },
            )
        )
    return documents
