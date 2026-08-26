"""Document chunking strategy preserving section context, page lineage, and token estimates."""

from typing import List, Dict, Any, Optional
from schemas.common import SourceChunk
from core.logger import logger
from core.config import settings


class DocumentChunker:
    """Recursive chunker splitting on natural boundaries (paragraphs, sentences, words)."""

    def __init__(self, chunk_size: int = settings.CHUNK_SIZE, chunk_overlap: int = settings.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (approx 4 chars per token)."""
        return max(1, len(text) // 4)

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively splits text into chunks under `chunk_size`."""
        final_chunks: List[str] = []
        if not text:
            return final_chunks

        separator = separators[-1]
        new_separators = []
        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)
        good_splits: List[str] = []

        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s[:self.chunk_size])
                else:
                    other_chunks = self._split_text_recursive(s, new_separators)
                    final_chunks.extend(other_chunks)

        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merges small splits together up to chunk_size with overlap."""
        docs: List[str] = []
        current_doc: List[str] = []
        total = 0

        for d in splits:
            _len = len(d)
            if total + _len + (len(separator) if current_doc else 0) > self.chunk_size:
                if current_doc:
                    doc = separator.join(current_doc)
                    if doc.strip():
                        docs.append(doc.strip())
                    # Keep elements for overlap
                    while total > self.chunk_overlap and current_doc:
                        popped = current_doc.pop(0)
                        total -= len(popped) + len(separator)
            current_doc.append(d)
            total += _len + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            doc = separator.join(current_doc)
            if doc.strip():
                docs.append(doc.strip())

        return docs

    def chunk_document(
        self,
        doc_id: str,
        filename: str,
        parsed_sections: List[Dict[str, Any]],
    ) -> List[SourceChunk]:
        """
        Takes parsed sections (with page / header lineage) and creates enriched SourceChunk objects.
        """
        all_chunks: List[SourceChunk] = []
        global_chunk_idx = 0

        for section in parsed_sections:
            raw_text = section.get("text", "").strip()
            page_num = section.get("page_number")
            section_title = section.get("section_title")

            if len(raw_text) < settings.MIN_CHUNK_LENGTH:
                # Still include if meaningful
                if raw_text:
                    chunk = SourceChunk(
                        chunk_id=f"{doc_id}_{global_chunk_idx}",
                        doc_id=doc_id,
                        filename=filename,
                        text=raw_text,
                        page_number=page_num,
                        section_title=section_title,
                        chunk_index=global_chunk_idx,
                        token_count=self._estimate_tokens(raw_text),
                    )
                    all_chunks.append(chunk)
                    global_chunk_idx += 1
                continue

            text_splits = self._split_text_recursive(raw_text, self.separators)
            for split_text in text_splits:
                if len(split_text.strip()) < 10:
                    continue

                chunk = SourceChunk(
                    chunk_id=f"{doc_id}_{global_chunk_idx}",
                    doc_id=doc_id,
                    filename=filename,
                    text=split_text.strip(),
                    page_number=page_num,
                    section_title=section_title,
                    chunk_index=global_chunk_idx,
                    token_count=self._estimate_tokens(split_text),
                )
                all_chunks.append(chunk)
                global_chunk_idx += 1

        logger.info(f"Generated {len(all_chunks)} chunks for '{filename}'")
        return all_chunks
