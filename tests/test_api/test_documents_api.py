"""API tests for health, settings, and document listing endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_root(client):
    assert client.get("/").status_code == 200


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "rag_documents_total" in resp.text


def test_list_documents_empty(client):
    resp = client.get("/api/v1/documents")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_get_settings(client):
    resp = client.get("/api/v1/settings")
    assert resp.status_code == 200
    body = resp.json()
    # local-first defaults
    assert body["embedding_provider"] == "huggingface"
    assert body["retrieval_mode"] == "hybrid"


def test_unknown_document_404(client):
    assert client.get("/api/v1/documents/does-not-exist").status_code == 404


def test_admin_stats(client):
    resp = client.get("/api/v1/admin/stats")
    assert resp.status_code == 200
    assert "chunk_count" in resp.json()
