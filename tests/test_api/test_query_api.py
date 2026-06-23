"""API tests for query + conversation endpoints (no documents indexed)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_query_without_documents_returns_409(client):
    resp = client.post("/api/v1/query", json={"question": "What is the revenue?"})
    assert resp.status_code == 409


def test_query_validation_rejects_empty(client):
    resp = client.post("/api/v1/query", json={"question": ""})
    assert resp.status_code == 422  # pydantic min_length


def test_conversation_crud(client):
    created = client.post("/api/v1/conversations", json={"title": "Test chat"})
    assert created.status_code == 200
    sid = created.json()["id"]

    listed = client.get("/api/v1/conversations")
    assert any(c["id"] == sid for c in listed.json())

    got = client.get(f"/api/v1/conversations/{sid}")
    assert got.json()["title"] == "Test chat"

    export = client.get(f"/api/v1/conversations/{sid}/export?fmt=markdown")
    assert export.status_code == 200
    assert "Test chat" in export.text

    deleted = client.delete(f"/api/v1/conversations/{sid}")
    assert deleted.status_code == 200
    assert client.get(f"/api/v1/conversations/{sid}").status_code == 404


def test_feedback_recorded(client):
    resp = client.post(
        "/api/v1/feedback", json={"query_id": "q-1", "helpful": True, "comment": "great"}
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "recorded"
