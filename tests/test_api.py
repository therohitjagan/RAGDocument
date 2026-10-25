"""Unit tests for FastAPI REST endpoints."""

import pytest
from fastapi.testclient import TestClient
from api.server import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "chroma_vectors" in data
    assert "bm25_chunks" in data
    assert data["bm25_chunks"] > 0


def test_documents_summary_endpoint():
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    data = response.json()
    assert "total_chunks" in data
    assert "domain_distribution" in data
    assert data["total_chunks"] > 0


def test_query_endpoint():
    payload = {
        "query": "What is the recommended buffer capacity for Raft Consensus Protocol Heartbeat?",
        "top_k": 3,
        "strategy": "hybrid_rerank",
        "alpha": 0.5,
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert "retrieved_chunks" in data
    assert len(data["retrieved_chunks"]) <= 3
    assert data["strategy_used"] == "hybrid_rerank"


def test_custom_ingest_endpoint():
    payload = {
        "doc_id": "test_api_rfc_99",
        "title": "API Test Custom Ingestion Specification",
        "content": "# API Test Custom Ingestion\n\n## Overview\nTesting on-the-fly dual indexing.",
        "domain": "testing",
        "doc_type": "rfc",
        "tags": ["test", "api"],
    }
    response = client.post("/api/v1/ingest", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["chunks_created"] >= 1
