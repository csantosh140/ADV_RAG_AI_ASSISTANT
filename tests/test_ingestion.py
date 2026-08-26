"""Tests for multi-format ingestion and chunking logic."""

import pytest
from pathlib import Path
from ingestion.pipeline import IngestionPipeline
from ingestion.parsers.md_parser import parse_markdown
from ingestion.parsers.txt_parser import parse_txt
from ingestion.chunker import DocumentChunker


def test_chunker_basic():
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    sections = [
        {
            "page_number": 1,
            "section_title": "Introduction",
            "text": "This is a test document explaining the Advanced RAG AI system architecture and components."
        }
    ]
    chunks = chunker.chunk_document("doc-123", "test.txt", sections)
    assert len(chunks) >= 1
    assert chunks[0].doc_id == "doc-123"
    assert chunks[0].filename == "test.txt"
    assert chunks[0].section_title == "Introduction"


def test_markdown_parser(tmp_path):
    md_file = tmp_path / "sample.md"
    md_file.write_text(
        "# Heading 1\nContent for section 1.\n\n## Heading 2\nContent for section 2 with detailed points.",
        encoding="utf-8"
    )

    sections, total_sec = parse_markdown(md_file)
    assert len(sections) >= 2
    assert "Heading 1" in sections[0]["section_title"] or "Heading 1" in sections[0]["text"]


def test_txt_parser(tmp_path):
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Hello world! This is a simple test document.", encoding="utf-8")

    sections, total = parse_txt(txt_file)
    assert len(sections) == 1
    assert "Hello world" in sections[0]["text"]


def test_pipeline_end_to_end(tmp_path):
    pipeline = IngestionPipeline(chunk_size=150, chunk_overlap=30)
    test_file = tmp_path / "guide.md"
    test_file.write_text("# RAG Architecture\nRetrieval-Augmented Generation blends vector search with LLMs.", encoding="utf-8")

    doc_meta, chunks = pipeline.process_file(test_file)
    assert doc_meta.filename == "guide.md"
    assert doc_meta.file_type == "md"
    assert len(chunks) > 0
    assert doc_meta.total_chunks == len(chunks)
