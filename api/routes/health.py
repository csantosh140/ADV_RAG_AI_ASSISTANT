"""Health check and system observability metrics endpoints."""

from fastapi import APIRouter, Depends
from api.dependencies import Container, get_container
from core.config import settings

router = APIRouter(tags=["System"])


@router.get("/health")
def health_check():
    """Liveness probe returning application health status."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
    }


@router.get("/metrics")
def get_system_metrics(c: Container = Depends(get_container)):
    """Observability metrics detailing index sizes, models, and registered documents."""
    total_docs = len(c.documents_registry)
    total_chunks = len(c.vector_store.chunks_map)

    return {
        "total_documents": total_docs,
        "total_chunks_indexed": total_chunks,
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.LLM_MODEL,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "reranker_model": settings.RERANKER_MODEL,
        "hybrid_alpha": settings.HYBRID_ALPHA,
    }
