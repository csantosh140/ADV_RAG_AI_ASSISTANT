"""Tests for Multi-Turn Conversational Memory and Query Rewriting."""

import pytest
from retrieval.query_rewriter import QueryRewriter


def test_query_rewriter_without_history():
    rewriter = QueryRewriter()
    queries = rewriter.rewrite_query("can you tell me what is FAISS?")
    assert len(queries) >= 1
    assert any("faiss" in q.lower() for q in queries)


def test_query_rewriter_with_history():
    rewriter = QueryRewriter()
    history = [
        {"role": "user", "content": "What is FlashRank?"},
        {"role": "assistant", "content": "FlashRank is a lightweight neural reranking library based on ONNX."}
    ]
    queries = rewriter.rewrite_query("How does it work?", chat_history=history)
    assert len(queries) >= 1
