"""LangGraph state schema for Agentic RAG workflow."""

from typing import List, Optional, Dict, Any, TypedDict
from schemas.common import SourceChunk, Citation


class AgentGraphState(TypedDict, total=False):
    """Internal state passed across LangGraph nodes."""
    original_query: str
    chat_history: Optional[List[Dict[str, str]]]
    rewritten_queries: List[str]
    doc_ids_filter: Optional[List[str]]
    raw_chunks: List[SourceChunk]
    reranked_chunks: List[SourceChunk]
    filtered_chunks: List[SourceChunk]
    answer: str
    citations: List[Citation]
    groundedness_score: float
    is_grounded: bool
    confidence_level: str
    retry_count: int
    execution_time_ms: float
    error_message: Optional[str]
