"""End-to-end ingestion pipeline coordinator."""

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Dict, Any

from schemas.common import DocumentMetadata, SourceChunk
from ingestion.parsers import (
    parse_pdf,
    parse_txt,
    parse_markdown,
    parse_json,
    parse_html,
    parse_docx,
    parse_csv,
    parse_url,
)
from ingestion.chunker import DocumentChunker
from core.exceptions import UnsupportedFileTypeError, DocumentParsingError
from core.logger import logger
from core.config import settings


class IngestionPipeline:
    """Coordinates parsing, chunking, and metadata generation for uploaded documents and URLs."""

    def __init__(self, chunk_size: int = settings.CHUNK_SIZE, chunk_overlap: int = settings.CHUNK_OVERLAP):
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def process_file(
        self,
        file_path: str | Path,
        custom_metadata: Dict[str, Any] | None = None
    ) -> Tuple[DocumentMetadata, List[SourceChunk]]:
        """
        Process a single file into DocumentMetadata and a list of SourceChunks.
        """
        path = Path(file_path)
        if not path.exists():
            raise DocumentParsingError(f"File not found: {file_path}")

        start_time = time.time()
        file_ext = path.suffix.lower().lstrip(".")
        filename = path.name
        file_size = path.stat().st_size
        doc_id = str(uuid.uuid4())

        # Select parser
        if file_ext == "pdf":
            parsed_sections, total_pages = parse_pdf(path)
        elif file_ext in ["txt", "text", "log"]:
            parsed_sections, total_pages = parse_txt(path)
        elif file_ext in ["csv", "tsv"]:
            parsed_sections, total_pages = parse_csv(path)
        elif file_ext in ["md", "markdown"]:
            parsed_sections, total_pages = parse_markdown(path)
        elif file_ext == "json":
            parsed_sections, total_pages = parse_json(path)
        elif file_ext in ["html", "htm"]:
            parsed_sections, total_pages = parse_html(path)
        elif file_ext in ["docx", "doc"]:
            parsed_sections, total_pages = parse_docx(path)
        else:
            raise UnsupportedFileTypeError(
                f"Unsupported file format '.{file_ext}'. Supported formats: PDF, TXT, MD, JSON, HTML, DOCX, CSV, TSV"
            )

        # Chunk sections
        chunks = self.chunker.chunk_document(
            doc_id=doc_id,
            filename=filename,
            parsed_sections=parsed_sections
        )

        doc_meta = DocumentMetadata(
            doc_id=doc_id,
            filename=filename,
            file_type=file_ext,
            file_size_bytes=file_size,
            total_pages=total_pages if file_ext in ["pdf", "docx"] else None,
            total_chunks=len(chunks),
            created_at=datetime.now(timezone.utc).isoformat(),
            custom_metadata=custom_metadata or {}
        )

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Ingestion completed for '{filename}': {len(chunks)} chunks in {elapsed_ms:.2f}ms")
        return doc_meta, chunks

    def process_url(
        self,
        url: str,
        custom_metadata: Dict[str, Any] | None = None
    ) -> Tuple[DocumentMetadata, List[SourceChunk]]:
        """
        Fetch remote web page, parse sections, and generate SourceChunks.
        """
        start_time = time.time()
        doc_id = str(uuid.uuid4())

        parsed_sections, page_title, total_sections = parse_url(url)
        filename = f"web_{page_title[:40].replace(' ', '_').replace('/', '_')}.html"

        chunks = self.chunker.chunk_document(
            doc_id=doc_id,
            filename=filename,
            parsed_sections=parsed_sections
        )

        meta = custom_metadata or {}
        meta["source_url"] = url
        meta["page_title"] = page_title

        doc_meta = DocumentMetadata(
            doc_id=doc_id,
            filename=filename,
            file_type="url",
            file_size_bytes=sum(len(c.text.encode("utf-8")) for c in chunks),
            total_pages=total_sections,
            total_chunks=len(chunks),
            created_at=datetime.now(timezone.utc).isoformat(),
            custom_metadata=meta
        )

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"URL ingestion completed for '{url}': {len(chunks)} chunks in {elapsed_ms:.2f}ms")
        return doc_meta, chunks

