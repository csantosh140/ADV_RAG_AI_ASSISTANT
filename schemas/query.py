"""Query request and response models with confidence indicators and audit trails."""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from schemas.common import SourceChunk, Citation


class ChatMessage(BaseModel):
    """Conversational message history turn."""
    role: Literal["user", "assistant", "system"] = Field(..., description="Message author role")
    content: str = Field(..., description="Message text content")


class QueryRequest(BaseModel):
    """User query payload for RAG assistant."""
    query: str = Field(..., min_length=1, max_length=2000, description="Natural language question")
    doc_ids: Optional[List[str]] = Field(None, description="Optional document ID filter")
    chat_history: Optional[List[ChatMessage]] = Field(None, description="Prior conversational turns for context-aware retrieval")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of chunks to rerank and feed to LLM")
    enable_query_rewriting: bool = Field(True, description="Whether to perform query expansion / HyDE")
    enable_reranking: bool = Field(True, description="Whether to apply neural cross-encoder reranking")
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0, description="LLM temperature override")


class RetrievalAudit(BaseModel):
    """Observability trace of the retrieval & reranking process."""
    original_query: str
    rewritten_queries: List[str] = Field(default_factory=list)
    raw_retrieved_chunks_count: int
    reranked_chunks_count: int
    graded_relevant_chunks_count: int
    execution_time_ms: float
    groundedness_check_passed: bool


class QueryResponse(BaseModel):
    """Grounded RAG assistant response with complete observability."""
    answer: str = Field(..., description="Generated grounded response")
    confidence_level: Literal["HIGH", "MEDIUM", "LOW", "ABSTAINED"] = Field(
        ..., description="System confidence indicator"
    )
    citations: List[Citation] = Field(default_factory=list, description="Extracted citation references")
    sources: List[SourceChunk] = Field(default_factory=list, description="Retrieved and verified source chunks")
    audit: RetrievalAudit = Field(..., description="Audit trace of the agentic workflow")
    time_taken_ms: float = Field(..., description="Total round-trip latency in ms")
