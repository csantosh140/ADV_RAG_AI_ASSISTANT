"""Common typed models for chunks, metadata, and provenance."""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata describing an ingested document."""
    doc_id: str = Field(..., description="Unique UUID for the document")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File extension (pdf, md, txt)")
    file_size_bytes: int = Field(..., description="File size in bytes")
    total_pages: Optional[int] = Field(None, description="Total number of pages (for PDF)")
    total_chunks: int = Field(0, description="Total chunks created from document")
    created_at: str = Field(..., description="ISO 8601 timestamp of ingestion")
    custom_metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary custom tags")


class SourceChunk(BaseModel):
    """Represents a text chunk with complete attribution lineage."""
    chunk_id: str = Field(..., description="Unique chunk identifier: {doc_id}_{chunk_index}")
    doc_id: str = Field(..., description="Parent document UUID")
    filename: str = Field(..., description="Source document filename")
    text: str = Field(..., description="Raw text content of the chunk")
    page_number: Optional[int] = Field(None, description="1-indexed page number (if PDF)")
    section_title: Optional[str] = Field(None, description="Markdown/Heading section title")
    chunk_index: int = Field(..., description="0-indexed position within the document")
    token_count: int = Field(0, description="Estimated token count")
    dense_score: Optional[float] = Field(None, description="Dense FAISS cosine similarity score")
    sparse_score: Optional[float] = Field(None, description="BM25 sparse score")
    hybrid_score: Optional[float] = Field(None, description="RRF fused score")
    rerank_score: Optional[float] = Field(None, description="FlashRank neural reranking score")


class Citation(BaseModel):
    """Structured citation reference in generated responses."""
    citation_id: int = Field(..., description="Numerical citation index [1], [2], etc.")
    doc_id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Filename")
    page_number: Optional[int] = Field(None, description="Source page number")
    section_title: Optional[str] = Field(None, description="Section heading")
    snippet: str = Field(..., description="Relevant snippet text")
