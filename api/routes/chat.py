"""Chat and grounded query endpoints supporting synchronous and streaming SSE."""

import json
import asyncio
from typing import AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from schemas.query import QueryRequest, QueryResponse
from api.dependencies import Container, get_container
from core.logger import logger

router = APIRouter(prefix="/api/v1/query", tags=["Chat & Query"])


@router.post("", response_model=QueryResponse)
def query_rag_assistant(
    request: QueryRequest,
    c: Container = Depends(get_container)
):
    """
    Execute full Agentic RAG pipeline:
    Query Rewriting -> Hybrid Retrieval -> FlashRank Rerank -> Grade -> Grounded Generation -> Hallucination Audit
    """
    if not c.vector_store.chunks_map:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents have been indexed yet. Please upload documents first."
        )

    try:
        history_list = [m.model_dump() for m in request.chat_history] if request.chat_history else None
        response = c.rag_workflow.run(
            query=request.query,
            doc_ids=request.doc_ids,
            chat_history=history_list,
            top_k=request.top_k or 4,
            enable_rewriting=request.enable_query_rewriting,
            enable_reranking=request.enable_reranking,
        )
        return response
    except Exception as e:
        logger.error(f"RAG execution error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred processing your query: {str(e)}"
        )


@router.post("/stream")
async def stream_rag_assistant(
    request: QueryRequest,
    c: Container = Depends(get_container)
):
    """
    Stream grounded RAG tokens in real-time via Server-Sent Events (SSE).
    """
    if not c.vector_store.chunks_map:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents have been indexed yet. Please upload documents first."
        )

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Yield status event
            yield f"data: {json.dumps({'event': 'status', 'data': 'Transforming query and retrieving hybrid context...'})}\n\n"
            await asyncio.sleep(0.02)

            history_list = [m.model_dump() for m in request.chat_history] if request.chat_history else None
            response = c.rag_workflow.run(
                query=request.query,
                doc_ids=request.doc_ids,
                chat_history=history_list,
                top_k=request.top_k or 4,
                enable_rewriting=request.enable_query_rewriting,
                enable_reranking=request.enable_reranking,
            )

            # Yield metadata event
            meta_payload = {
                "event": "metadata",
                "citations": [cit.model_dump() for cit in response.citations],
                "sources": [s.model_dump() for s in response.sources],
                "confidence_level": response.confidence_level,
                "audit": response.audit.model_dump(),
                "time_taken_ms": response.time_taken_ms,
            }
            yield f"data: {json.dumps(meta_payload)}\n\n"

            # Stream words of answer
            words = response.answer.split(" ")
            for idx, token in enumerate(words):
                word_out = token if idx == len(words) - 1 else token + " "
                yield f"data: {json.dumps({'event': 'token', 'data': word_out})}\n\n"
                await asyncio.sleep(0.01)

            yield f"data: {json.dumps({'event': 'done', 'data': '[DONE]'})}\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'event': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )
