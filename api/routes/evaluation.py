"""Evaluation and benchmark endpoints for RAG quality assurance."""

from fastapi import APIRouter, Depends, HTTPException, status
from schemas.evaluation import (
    EvaluationRequest,
    EvaluationResult,
    BatchEvaluationRequest,
    BatchEvaluationResponse,
)
from api.dependencies import Container, get_container
from core.logger import logger

router = APIRouter(prefix="/api/v1/evaluate", tags=["Evaluation & Benchmarking"])


@router.post("", response_model=EvaluationResult)
def evaluate_single_query(
    request: EvaluationRequest,
    c: Container = Depends(get_container)
):
    """
    Evaluate a single query on Groundedness, Faithfulness, Relevance, and Latency.
    """
    if not c.vector_store.chunks_map:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents have been indexed yet. Please upload documents before evaluating."
        )

    try:
        result = c.evaluator.evaluate_single(
            query=request.query,
            ground_truth=request.ground_truth,
            expected_doc_ids=request.expected_doc_ids,
            doc_ids_filter=request.doc_ids_filter,
            enable_rewriting=request.enable_query_rewriting,
            enable_reranking=request.enable_reranking,
            top_k=request.top_k,
        )
        return result
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )


@router.post("/batch", response_model=BatchEvaluationResponse)
def evaluate_benchmark_batch(
    request: BatchEvaluationRequest,
    c: Container = Depends(get_container)
):
    """
    Run an automated benchmark suite of test cases and generate an aggregated quality report.
    """
    if not c.vector_store.chunks_map:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents have been indexed yet. Please upload documents before evaluating."
        )

    try:
        report = c.evaluator.evaluate_batch(
            suite_name=request.suite_name,
            test_cases=request.test_cases,
            doc_ids_filter=request.doc_ids_filter,
            enable_rewriting=request.enable_query_rewriting,
            enable_reranking=request.enable_reranking,
            top_k=request.top_k,
        )
        return report
    except Exception as e:
        logger.error(f"Batch evaluation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch evaluation failed: {str(e)}"
        )
