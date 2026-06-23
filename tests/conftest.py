"""Shared pytest fixtures: sample documents, a fake LLM, and a temp index dir.

The ``_isolate_global_state`` fixture is autouse + session-scoped so the whole
suite is hermetic: the API tests run against an empty, temp-backed index and
isolated eval/storage directories, and runtime settings PATCH never touches the
real ``.env``. Without it, the "no documents → 409" and "no results → 404" tests
depend on whatever happens to be in ``data/`` on the developer's machine.
"""

from __future__ import annotations

import os
import tempfile

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake import FakeListLLM


@pytest.fixture(scope="session", autouse=True)
def _isolate_global_state():
    """Redirect all on-disk state to a temp dir and neutralise .env writes."""
    import app.api.settings as settings_api
    import app.dependencies as dependencies
    from app.config import get_settings

    settings = get_settings()
    redirected = {
        "faiss_index_dir": "faiss_index",
        "metadata_file": "metadata.json",
        "upload_dir": "uploads",
        "eval_results_dir": "results",
        "eval_testset_dir": "testsets",
        "log_file": "app.log",
        "users_file": "users.json",
    }
    originals = {field: getattr(settings, field) for field in redirected}
    original_persist = settings_api.persist_settings_to_env

    with tempfile.TemporaryDirectory() as tmp:
        for field, leaf in redirected.items():
            setattr(settings, field, os.path.join(tmp, leaf))
        # Persist settings to a throwaway .env so tests never mutate the real one.
        env_path = os.path.join(tmp, ".env")
        settings_api.persist_settings_to_env = (
            lambda updates, _p=env_path: original_persist(updates, _p)
        )
        dependencies._state = None  # force a fresh, empty index on next init
        try:
            yield
        finally:
            for field, value in originals.items():
                setattr(settings, field, value)
            settings_api.persist_settings_to_env = original_persist
            dependencies._state = None


@pytest.fixture
def sample_docs() -> list[Document]:
    """A few documents with citation metadata, as the loader would produce."""
    return [
        Document(
            page_content="The total revenue reported for 2024 was 4.2 million dollars.",
            metadata={"filename": "report.pdf", "page_number": 1, "document_id": "doc-1"},
        ),
        Document(
            page_content="Operating expenses rose by twelve percent year over year.",
            metadata={"filename": "report.pdf", "page_number": 2, "document_id": "doc-1"},
        ),
        Document(
            page_content="The clinical trial enrolled 300 participants across four sites.",
            metadata={"filename": "study.pdf", "page_number": 1, "document_id": "doc-2"},
        ),
    ]


@pytest.fixture
def fake_llm() -> FakeListLLM:
    """Deterministic LLM that returns canned responses (no API key needed)."""
    return FakeListLLM(responses=["This is a grounded test answer. [Source: report.pdf, Page: 1]"])


@pytest.fixture
def temp_index_dir():
    """Isolated temp directory for FAISS/chunk persistence."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def temp_text_file():
    """A small on-disk .txt file for loader/ingestion tests."""
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("DocMind is a retrieval augmented generation system.\n" * 20)
    yield path
    os.remove(path)
