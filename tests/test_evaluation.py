"""Tests for RAG Evaluation and Benchmarking Suite."""

import pytest
from fastapi.testclient import TestClient
from api.main import app
from schemas.evaluation import EvalTestCase, BatchEvaluationRequest

client = TestClient(app)


def test_evaluation_endpoints():
    # 1. Ingest a test document first
    file_content = b"# Vector Search Systems\nFAISS provides fast dense similarity search. BM25 provides sparse lexical search. FlashRank reranks top hybrid candidates."
    files = {"file": ("eval_doc.md", file_content, "text/markdown")}

    upload_res = client.post("/api/v1/documents/upload", files=files)
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document"]["doc_id"]

    # 2. Test Single Evaluation
    eval_payload = {
        "query": "What is FAISS used for?",
        "ground_truth": "FAISS provides fast dense similarity search.",
        "enable_query_rewriting": True,
        "enable_reranking": True,
        "top_k": 2
    }
    res = client.post("/api/v1/evaluate", json=eval_payload)
    assert res.status_code == 200
    data = res.json()
    assert "metrics" in data
    assert "faithfulness" in data["metrics"]
    assert "answer_relevance" in data["metrics"]
    assert data["overall_score"] >= 0.0

    # 3. Test Batch Benchmark Suite
    batch_payload = {
        "suite_name": "Test Suite",
        "test_cases": [
            {
                "query": "What is BM25 used for?",
                "expected_answer": "BM25 provides sparse lexical search."
            },
            {
                "query": "What does FlashRank do?",
                "expected_answer": "FlashRank reranks top hybrid candidates."
            }
        ],
        "top_k": 2
    }
    batch_res = client.post("/api/v1/evaluate/batch", json=batch_payload)
    assert batch_res.status_code == 200
    batch_data = batch_res.json()
    assert batch_data["total_queries"] == 2
    assert "average_faithfulness" in batch_data
    assert "overall_benchmark_score" in batch_data

    # 4. Clean up
    client.delete(f"/api/v1/documents/{doc_id}")
