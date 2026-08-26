"""High-level retrieval service orchestrating rewriting, hybrid search, and neural reranking."""

from typing import List, Optional
from schemas.common import SourceChunk
from indexing.hybrid_search import HybridSearchRetriever
from retrieval.query_rewriter import QueryRewriter
from retrieval.reranker import FlashRankReranker
from core.config import settings
from core.logger import logger


class RetrieverService:
    """End-to-end retrieval coordinator."""

    def __init__(
        self,
        hybrid_retriever: HybridSearchRetriever,
        query_rewriter: Optional[QueryRewriter] = None,
        reranker: Optional[FlashRankReranker] = None,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.reranker = reranker or FlashRankReranker()

    def retrieve_and_rerank(
        self,
        query: str,
        doc_ids: Optional[List[str]] = None,
        top_k: int = settings.RERANK_TOP_K,
        enable_rewriting: bool = True,
        enable_reranking: bool = True,
    ) -> tuple[List[SourceChunk], List[str]]:
        """
        Execute multi-stage retrieval:
        1. Rewrite / expand query
        2. Hybrid retrieve from FAISS + BM25 across queries
        3. Deduplicate candidates
        4. Cross-encoder neural reranking
        """
        # Step 1: Query transformation
        queries = [query]
        if enable_rewriting:
            queries = self.query_rewriter.rewrite_query(query)

        # Step 2: Multi-query hybrid retrieval
        all_candidates: List[SourceChunk] = []
        seen_ids = set()

        for q in queries:
            results = self.hybrid_retriever.retrieve(
                query=q,
                top_k=settings.RETRIEVAL_TOP_K,
                doc_ids=doc_ids
            )
            for chunk in results:
                if chunk.chunk_id not in seen_ids:
                    seen_ids.add(chunk.chunk_id)
                    all_candidates.append(chunk)

        # Step 3: Neural reranking
        if enable_reranking and all_candidates:
            final_chunks = self.reranker.rerank(query=query, chunks=all_candidates, top_k=top_k)
        else:
            final_chunks = all_candidates[:top_k]

        logger.info(f"RetrieverService finished: {len(final_chunks)} final chunks selected.")
        return final_chunks, queries
