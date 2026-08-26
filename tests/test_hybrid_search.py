"""Tests for Vector Store, BM25 Index, and Hybrid RRF Fusion."""

import pytest
from schemas.common import SourceChunk
from indexing.vector_store import FAISSVectorStore
from indexing.bm25_index import BM25Index
from indexing.hybrid_search import HybridSearchRetriever


@pytest.fixture
def sample_chunks():
    return [
        SourceChunk(
            chunk_id="doc1_0",
            doc_id="doc1",
            filename="rag_intro.md",
            text="LangGraph enables cyclic graphs and agentic workflows with state persistence.",
            page_number=1,
            section_title="LangGraph Overview",
            chunk_index=0,
            token_count=15
        ),
        SourceChunk(
            chunk_id="doc1_1",
            doc_id="doc1",
            filename="rag_intro.md",
            text="FAISS is a library for dense vector similarity search and clustering.",
            page_number=1,
            section_title="FAISS Overview",
            chunk_index=1,
            token_count=14
        ),
        SourceChunk(
            chunk_id="doc2_0",
            doc_id="doc2",
            filename="bm25_guide.txt",
            text="BM25 is a sparse ranking function used by search engines for keyword relevance.",
            page_number=1,
            section_title="BM25 Guide",
            chunk_index=0,
            token_count=16
        ),
    ]


def test_bm25_search(sample_chunks):
    bm25 = BM25Index()
    bm25.build_index(sample_chunks)
    results = bm25.search("BM25 ranking function", top_k=2)
    assert len(results) >= 1
    assert results[0].doc_id == "doc2"
    assert results[0].sparse_score is not None


def test_hybrid_search(sample_chunks):
    vector_store = FAISSVectorStore()
    vector_store.chunks_map = {i: c for i, c in enumerate(sample_chunks)}

    hybrid = HybridSearchRetriever(vector_store=vector_store)
    hybrid.sync()

    results = hybrid.retrieve("LangGraph workflows", top_k=2)
    assert len(results) >= 1
    assert any("LangGraph" in c.text for c in results)
    assert results[0].hybrid_score is not None
