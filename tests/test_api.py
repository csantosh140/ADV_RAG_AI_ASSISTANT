"""Integration tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "total_chunks_indexed" in data


def test_document_upload_and_query():
    # 1. Upload a markdown file
    file_content = b"# Advanced RAG\nThis production RAG system uses LangGraph and FAISS for grounded retrieval."
    files = {"file": ("test_doc.md", file_content, "text/markdown")}

    upload_res = client.post("/api/v1/documents/upload", files=files)
    assert upload_res.status_code == 201
    upload_data = upload_res.json()
    assert upload_data["status"] == "success"
    doc_id = upload_data["document"]["doc_id"]

    # 2. List documents
    list_res = client.get("/api/v1/documents")
    assert list_res.status_code == 200
    docs = list_res.json()["documents"]
    assert any(d["doc_id"] == doc_id for d in docs)

    # 3. Query the assistant
    query_payload = {
        "query": "What technologies does the production RAG system use?",
        "enable_query_rewriting": True,
        "enable_reranking": True,
        "top_k": 2
    }
    query_res = client.post("/api/v1/query", json=query_payload)
    assert query_res.status_code == 200
    query_data = query_res.json()
    assert "answer" in query_data
    assert "citations" in query_data
    assert query_data["confidence_level"] in ["HIGH", "MEDIUM", "LOW", "ABSTAINED"]

    # 4. Clean up
    del_res = client.delete(f"/api/v1/documents/{doc_id}")
    assert del_res.status_code == 200


def test_url_ingestion_api(monkeypatch):
    class MockResponse:
        status_code = 200
        text = "<html><head><title>RAG Benchmark</title></head><body><h1>Benchmark Study</h1><p>FlashRank demonstrates 5x lower latency than traditional cross-encoders.</p></body></html>"
        def raise_for_status(self):
            pass

    monkeypatch.setattr("requests.get", lambda url, headers=None, timeout=15: MockResponse())

    res = client.post("/api/v1/documents/url", json={"url": "https://example.com/benchmark"})
    assert res.status_code == 201
    data = res.json()
    assert data["status"] == "success"
    assert data["chunks_created"] >= 1
    doc_id = data["document"]["doc_id"]

    # Cleanup
    del_res = client.delete(f"/api/v1/documents/{doc_id}")
    assert del_res.status_code == 200


def test_csv_upload_api():
    csv_bytes = b"Product,Category,Price\nLaptop,Electronics,1200\nHeadphones,Audio,150\n"
    files = {"file": ("products.csv", csv_bytes, "text/csv")}

    upload_res = client.post("/api/v1/documents/upload", files=files)
    assert upload_res.status_code == 201
    upload_data = upload_res.json()
    doc_id = upload_data["document"]["doc_id"]

    # Cleanup
    del_res = client.delete(f"/api/v1/documents/{doc_id}")
    assert del_res.status_code == 200

