"""Schemas for RAG Evaluation and Benchmarking Suite."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class EvalTestCase(BaseModel):
    """A test case for evaluating RAG pipeline performance."""
    query: str = Field(..., description="Evaluation query")
    expected_answer: Optional[str] = Field(None, description="Ground truth answer (optional)")
    expected_doc_ids: Optional[List[str]] = Field(None, description="Expected relevant document IDs (optional)")
    expected_keywords: Optional[List[str]] = Field(None, description="Key facts or terminology that must appear in context/answer")


class EvalMetricScore(BaseModel):
    """Individual metric evaluation result."""
    metric_name: str
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized score [0.0 - 1.0]")
    passed: bool = Field(..., description="Whether score meets passing threshold")
    reasoning: Optional[str] = Field(None, description="LLM/rule explanation of score")


class EvaluationRequest(BaseModel):
    """Request to evaluate a single query or interaction."""
    query: str
    ground_truth: Optional[str] = None
    expected_doc_ids: Optional[List[str]] = None
    doc_ids_filter: Optional[List[str]] = None
    enable_query_rewriting: bool = True
    enable_reranking: bool = True
    top_k: int = 4


class EvaluationResult(BaseModel):
    """Detailed evaluation report for a single query."""
    query: str
    generated_answer: str
    confidence_level: str
    ground_truth: Optional[str] = None
    metrics: Dict[str, EvalMetricScore]
    overall_score: float = Field(..., ge=0.0, le=1.0)
    latency_ms: float
    retrieved_chunks_count: int
    citations_count: int


class BatchEvaluationRequest(BaseModel):
    """Batch evaluation benchmark request."""
    suite_name: str = "Default RAG Benchmark Suite"
    test_cases: List[EvalTestCase]
    doc_ids_filter: Optional[List[str]] = None
    enable_query_rewriting: bool = True
    enable_reranking: bool = True
    top_k: int = 4


class BatchEvaluationResponse(BaseModel):
    """Aggregated benchmark evaluation summary."""
    suite_name: str
    total_queries: int
    passed_queries: int
    average_faithfulness: float
    average_answer_relevance: float
    average_context_precision: float
    average_latency_ms: float
    overall_benchmark_score: float
    results: List[EvaluationResult]
