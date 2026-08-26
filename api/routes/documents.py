"""Document ingestion, listing, and deletion endpoints."""

import time
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, status

from schemas.ingestion import (
    IngestionResponse,
    DocumentListResponse,
    DeleteDocumentResponse,
    UrlIngestionRequest,
)
from schemas.common import DocumentMetadata
from api.dependencies import Container, get_container
from core.config import settings
from core.logger import logger
from core.exceptions import RAGException

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])

ALLOWED_EXTENSIONS = {
    ".pdf", ".txt", ".text", ".md", ".markdown",
    ".json", ".html", ".htm", ".docx", ".doc",
    ".csv", ".tsv", ".log"
}


@router.post("/upload", response_model=IngestionResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None),
    c: Container = Depends(get_container)
):
    """
    Upload a document (PDF, DOCX, Markdown, HTML, JSON, CSV, TXT), parse, chunk, and index into FAISS + BM25.
    """
    start_time = time.time()
    filename = file.filename or "uploaded_document"
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats: PDF, DOCX, MD, HTML, JSON, CSV, TSV, TXT"
        )

    # Save file to disk
    save_path = settings.RAW_DOCS_DIR / filename
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save uploaded file {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")

    try:
        doc_meta, chunks = c.ingestion_pipeline.process_file(
            file_path=save_path,
            custom_metadata={"uploaded_by": "api_user"}
        )

        # Add to Vector Store & BM25
        c.vector_store.add_chunks(chunks)
        c.hybrid_retriever.sync()

        # Update registry
        c.documents_registry[doc_meta.doc_id] = doc_meta
        c.save_registry()

        elapsed_ms = (time.time() - start_time) * 1000
        return IngestionResponse(
            status="success",
            message=f"Successfully ingested and indexed '{filename}'",
            document=doc_meta,
            chunks_created=len(chunks),
            time_taken_ms=round(elapsed_ms, 2)
        )
    except RAGException as e:
        logger.error(f"Ingestion domain error: {e.message}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected ingestion failure: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/url", response_model=IngestionResponse, status_code=status.HTTP_201_CREATED)
async def ingest_url(
    request: UrlIngestionRequest,
    c: Container = Depends(get_container)
):
    """
    Fetch remote web page URL, parse HTML headings/content, chunk, and index into FAISS + BM25.
    """
    start_time = time.time()
    url = request.url.strip()

    if not url.startswith("http://") and not url.startswith("https://"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid URL schema. URL must begin with http:// or https://"
        )

    try:
        doc_meta, chunks = c.ingestion_pipeline.process_url(
            url=url,
            custom_metadata=request.custom_metadata
        )

        # Add to Vector Store & BM25
        c.vector_store.add_chunks(chunks)
        c.hybrid_retriever.sync()

        # Update registry
        c.documents_registry[doc_meta.doc_id] = doc_meta
        c.save_registry()

        elapsed_ms = (time.time() - start_time) * 1000
        return IngestionResponse(
            status="success",
            message=f"Successfully scraped and indexed URL '{url}'",
            document=doc_meta,
            chunks_created=len(chunks),
            time_taken_ms=round(elapsed_ms, 2)
        )
    except RAGException as e:
        logger.error(f"URL Ingestion domain error: {e.message}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected URL ingestion failure: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("", response_model=DocumentListResponse)
def list_documents(c: Container = Depends(get_container)):
    """Retrieve all indexed documents and summary statistics."""
    docs = list(c.documents_registry.values())
    total_chunks = len(c.vector_store.chunks_map)
    return DocumentListResponse(
        total_documents=len(docs),
        total_chunks=total_chunks,
        documents=docs
    )


@router.delete("/{doc_id}", response_model=DeleteDocumentResponse)
def delete_document(doc_id: str, c: Container = Depends(get_container)):
    """Delete a document and purge its vectors from FAISS and BM25."""
    if doc_id not in c.documents_registry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{doc_id}' not found."
        )

    doc_meta = c.documents_registry.pop(doc_id)
    chunks_deleted = c.vector_store.delete_document(doc_id)
    c.hybrid_retriever.sync()
    c.save_registry()

    # Attempt to remove raw file
    raw_path = settings.RAW_DOCS_DIR / doc_meta.filename
    if raw_path.exists():
        try:
            raw_path.unlink()
        except Exception:
            pass

    return DeleteDocumentResponse(
        status="success",
        message=f"Document '{doc_meta.filename}' successfully removed.",
        doc_id=doc_id,
        chunks_deleted=chunks_deleted
    )


@router.delete("", response_model=dict)
def clear_all_documents(c: Container = Depends(get_container)):
    """Clear all documents and reset indexes."""
    c.vector_store.clear_all()
    c.hybrid_retriever.sync()
    c.documents_registry.clear()
    c.save_registry()
    return {"status": "success", "message": "All documents and vectors cleared."}
