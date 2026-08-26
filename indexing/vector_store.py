"""FAISS vector store manager with persistence, filtering, and NumPy cosine-similarity fallback."""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from schemas.common import SourceChunk
from indexing.embeddings import EmbeddingService
from core.config import settings
from core.logger import logger
from core.exceptions import VectorStoreError


class FAISSVectorStore:
    """Production FAISS vector index with metadata mapping, numpy cosine-similarity fallback, and persistence."""

    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        self.embedding_service = embedding_service or EmbeddingService()
        self.dimension = self.embedding_service.dimension
        self.index_dir = settings.FAISS_INDEX_DIR
        self.index_file = self.index_dir / "faiss.index"
        self.vectors_file = self.index_dir / "vectors.npy"
        self.metadata_file = self.index_dir / "chunks_metadata.json"

        self.index = None
        self.vectors: Optional[np.ndarray] = None
        self.chunks_map: Dict[int, SourceChunk] = {}  # Index position -> SourceChunk
        self._initialize_index()

    def _initialize_index(self):
        """Load existing index from disk or initialize fresh index."""
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load metadata if available
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    meta_raw = json.load(f)
                    self.chunks_map = {
                        int(k): SourceChunk(**v) for k, v in meta_raw.items()
                    }
                logger.info(f"Loaded {len(self.chunks_map)} chunks metadata from {self.metadata_file}")
            except Exception as e:
                logger.warning(f"Failed to load chunk metadata: {e}")
                self.chunks_map = {}

        # 2. Load vectors matrix if available
        if self.vectors_file.exists():
            try:
                self.vectors = np.load(str(self.vectors_file))
                logger.info(f"Loaded vectors matrix with shape {self.vectors.shape}")
            except Exception as e:
                logger.warning(f"Failed to load vectors.npy: {e}")
                self.vectors = None

        # 3. Load or build FAISS index if library is available
        if FAISS_AVAILABLE:
            if self.index_file.exists():
                try:
                    self.index = faiss.read_index(str(self.index_file))
                    logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors.")
                    return
                except Exception as e:
                    logger.warning(f"Failed to read faiss.index: {e}. Reinitializing.")

            self.index = faiss.IndexFlatIP(self.dimension)
            if self.vectors is not None and len(self.vectors) > 0:
                self.index.add(self.vectors)
        else:
            self.index = None
            logger.info("FAISS C++ library not available. Vector store running in optimized NumPy cosine similarity mode.")

    def save(self):
        """Persist FAISS index, NumPy vectors matrix, and chunk metadata to disk."""
        try:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            if FAISS_AVAILABLE and self.index is not None:
                faiss.write_index(self.index, str(self.index_file))

            if self.vectors is not None:
                np.save(str(self.vectors_file), self.vectors)

            with open(self.metadata_file, "w", encoding="utf-8") as f:
                serializable = {
                    str(k): v.model_dump() for k, v in self.chunks_map.items()
                }
                json.dump(serializable, f, indent=2)

            logger.info(f"Persisted vector store ({len(self.chunks_map)} chunks) to disk.")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise VectorStoreError(f"Failed to save vector index: {e}") from e

    def add_chunks(self, chunks: List[SourceChunk]) -> int:
        """Embed and insert chunks into vector store."""
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = self.embedding_service.embed_documents(texts)
        new_vectors = np.array(embeddings, dtype="float32")

        # Normalize for cosine similarity
        norms = np.linalg.norm(new_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        new_vectors = new_vectors / norms

        start_idx = len(self.chunks_map)
        if FAISS_AVAILABLE and self.index is not None:
            self.index.add(new_vectors)

        if self.vectors is None or len(self.vectors) == 0:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])

        for i, chunk in enumerate(chunks):
            self.chunks_map[start_idx + i] = chunk

        self.save()
        logger.info(f"Added {len(chunks)} chunks to vector store. Total: {len(self.chunks_map)}")
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = settings.RETRIEVAL_TOP_K,
        doc_ids: Optional[List[str]] = None,
        similarity_threshold: float = settings.SIMILARITY_THRESHOLD,
    ) -> List[SourceChunk]:
        """Search vector store for top_k relevant chunks with optional doc_id filtering."""
        if not self.chunks_map:
            return []

        query_vec = np.array(self.embedding_service.embed_query(query), dtype="float32")
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        results: List[SourceChunk] = []

        if FAISS_AVAILABLE and self.index is not None and self.index.ntotal > 0:
            fetch_k = min(len(self.chunks_map), top_k * 3 if doc_ids else top_k)
            distances, indices = self.index.search(np.array([query_vec]), fetch_k)

            for score, idx in zip(distances[0], indices[0]):
                if idx == -1 or idx not in self.chunks_map:
                    continue
                if score < similarity_threshold:
                    continue

                chunk = self.chunks_map[idx].model_copy()
                if doc_ids and chunk.doc_id not in doc_ids:
                    continue

                chunk.dense_score = float(score)
                results.append(chunk)
                if len(results) >= top_k:
                    break
        elif self.vectors is not None and len(self.vectors) == len(self.chunks_map):
            # NumPy vectorized cosine similarity
            scores = np.dot(self.vectors, query_vec)
            sorted_indices = np.argsort(scores)[::-1]

            for idx in sorted_indices:
                score = float(scores[idx])
                if score < similarity_threshold:
                    continue
                chunk = self.chunks_map[int(idx)].model_copy()
                if doc_ids and chunk.doc_id not in doc_ids:
                    continue

                chunk.dense_score = round(score, 4)
                results.append(chunk)
                if len(results) >= top_k:
                    break
        else:
            # Fallback linear iteration
            for idx, chunk in self.chunks_map.items():
                if doc_ids and chunk.doc_id not in doc_ids:
                    continue
                chunk_copy = chunk.model_copy()
                chunk_copy.dense_score = 0.5
                results.append(chunk_copy)
            results = results[:top_k]

        return results

    def delete_document(self, doc_id: str) -> int:
        """Remove all chunks associated with a doc_id and rebuild index."""
        initial_count = len(self.chunks_map)
        remaining_chunks = [
            chunk for chunk in self.chunks_map.values() if chunk.doc_id != doc_id
        ]
        deleted_count = initial_count - len(remaining_chunks)

        if deleted_count > 0:
            if FAISS_AVAILABLE:
                self.index = faiss.IndexFlatIP(self.dimension)
            self.vectors = None
            self.chunks_map = {}
            if remaining_chunks:
                self.add_chunks(remaining_chunks)
            else:
                self.save()
            logger.info(f"Deleted doc_id '{doc_id}', removed {deleted_count} chunks.")

        return deleted_count

    def clear_all(self):
        """Reset the vector store entirely."""
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.dimension)
        self.vectors = None
        self.chunks_map = {}
        self.save()
        logger.info("Cleared all records from vector store.")

    def get_all_chunks(self) -> List[SourceChunk]:
        """Return all indexed chunks."""
        return list(self.chunks_map.values())
