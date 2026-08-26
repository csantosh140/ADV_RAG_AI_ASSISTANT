"""Ingestion request and response schemas."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from schemas.common import DocumentMetadata, SourceChunk


class IngestionRequest(BaseModel):
    """Configuration for ingestion."""
    chunk_size: Optional[int] = Field(None, description="Override default chunk size")
    chunk_overlap: Optional[int] = Field(None, description="Override default chunk overlap")
    custom_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata tags to attach")


class IngestionResponse(BaseModel):
    """Response returned upon successful document ingestion."""
    status: str = Field("success", description="Status code (success, partial, error)")
    message: str = Field(..., description="Human-readable result summary")
    document: DocumentMetadata = Field(..., description="Processed document metadata")
    chunks_created: int = Field(..., description="Number of vector chunks generated")
    time_taken_ms: float = Field(..., description="Ingestion execution time in ms")


class DocumentListResponse(BaseModel):
    """List of all indexed documents in the system."""
    total_documents: int = Field(..., description="Total documents stored")
    total_chunks: int = Field(..., description="Total chunks across all documents")
    documents: List[DocumentMetadata] = Field(default_factory=list)


class DeleteDocumentResponse(BaseModel):
    """Response returned upon deleting a document."""
    status: str = Field("success")
    message: str
    doc_id: str
    chunks_deleted: int


class UrlIngestionRequest(BaseModel):
    """Request schema for ingesting a remote Web URL."""
    url: str = Field(..., description="Full URL to scrape, parse, and index (http:// or https://)")
    custom_metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata tags to attach")

