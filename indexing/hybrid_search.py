"""Hybrid search orchestrator fusing Dense (FAISS) and Sparse (BM25) with Reciprocal Rank Fusion (RRF)."""

from typing import List, Optional, Dict
from schemas.common import SourceChunk
from indexing.vector_store import FAISSVectorStore
from indexing.bm25_index import BM25Index
from core.config import settings
from core.logger import logger


class HybridSearchRetriever:
    """Fuses Dense Vector Search and BM25 Sparse Search using Reciprocal Rank Fusion."""

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        bm25_index: Optional[BM25Index] = None,
        rrf_k: int = 60
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index or BM25Index()
        self.rrf_k = rrf_k
        self._sync_bm25()

    def _sync_bm25(self):
        """Synchronize BM25 index with current FAISS chunks."""
        all_chunks = self.vector_store.get_all_chunks()
        self.bm25_index.build_index(all_chunks)

    def sync(self):
        """Public method to rebuild BM25 after updates."""
        self._sync_bm25()

    def retrieve(
        self,
        query: str,
        top_k: int = settings.RETRIEVAL_TOP_K,
        doc_ids: Optional[List[str]] = None,
        dense_weight: float = settings.HYBRID_ALPHA,
    ) -> List[SourceChunk]:
        """
        Execute hybrid search combining FAISS and BM25 using RRF:
        RRF_Score(d) = dense_weight * (1 / (rrf_k + rank_dense)) + (1 - dense_weight) * (1 / (rrf_k + rank_sparse))
        """
        sparse_weight = 1.0 - dense_weight

        # 1. Fetch dense candidates
        dense_results = self.vector_store.search(
            query=query,
            top_k=top_k * 2,
            doc_ids=doc_ids
        )

        # 2. Fetch sparse candidates
        sparse_results = self.bm25_index.search(
            query=query,
            top_k=top_k * 2,
            doc_ids=doc_ids
        )

        # 3. Fuse scores with RRF
        chunk_map: Dict[str, SourceChunk] = {}
        rrf_scores: Dict[str, float] = {}

        # Dense rank scoring
        for rank, chunk in enumerate(dense_results, start=1):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            score = dense_weight * (1.0 / (self.rrf_k + rank))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score

        # Sparse rank scoring
        for rank, chunk in enumerate(sparse_results, start=1):
            cid = chunk.chunk_id
            if cid not in chunk_map:
                chunk_map[cid] = chunk
            else:
                chunk_map[cid].sparse_score = chunk.sparse_score
            score = sparse_weight * (1.0 / (self.rrf_k + rank))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score

        # Sort chunks by fused RRF score
        sorted_chunk_ids = sorted(
            rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True
        )

        fused_chunks: List[SourceChunk] = []
        for cid in sorted_chunk_ids[:top_k]:
            chunk = chunk_map[cid]
            chunk.hybrid_score = round(rrf_scores[cid], 5)
            fused_chunks.append(chunk)

        logger.info(
            f"Hybrid search returned {len(fused_chunks)} chunks for query: '{query[:40]}...' "
            f"(Dense: {len(dense_results)}, Sparse: {len(sparse_results)})"
        )
        return fused_chunks
