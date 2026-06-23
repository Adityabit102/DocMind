"""API tests for evaluation + settings-mutation endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_evaluate_without_documents_409(client):
    resp = client.post("/api/v1/evaluate", json={"test_size": 5})
    assert resp.status_code == 409


def test_results_before_any_run_404(client):
    assert client.get("/api/v1/evaluate/results").status_code == 404


def test_drift_with_no_data(client):
    resp = client.get("/api/v1/evaluate/drift")
    assert resp.status_code == 200
    assert "drift_detected" in resp.json()


def test_patch_settings(client):
    resp = client.patch("/api/v1/settings", json={"retrieval_mode": "mmr", "retrieval_k": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert body["retrieval_mode"] == "mmr"
    assert body["retrieval_k"] == 10
    # restore
    client.patch("/api/v1/settings", json={"retrieval_mode": "hybrid", "retrieval_k": 20})


def test_clear_cache(client):
    resp = client.delete("/api/v1/cache")
    assert resp.status_code == 200
