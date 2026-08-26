"""Neural cross-encoder reranker using FlashRank for high-precision context scoring."""

from typing import List
from schemas.common import SourceChunk
from core.config import settings
from core.logger import logger

try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_AVAILABLE = True
except ImportError:
    FLASHRANK_AVAILABLE = False


class FlashRankReranker:
    """Ultra-fast neural reranker to filter out irrelevant retrieved chunks."""

    def __init__(self, model_name: str = settings.RERANKER_MODEL):
        self.model_name = model_name
        self.ranker = None
        self._init_ranker()

    def _init_ranker(self):
        """Initialize FlashRank model if available."""
        if FLASHRANK_AVAILABLE:
            try:
                self.ranker = Ranker(model_name=self.model_name, cache_dir=str(settings.DATA_DIR / "models"))
                logger.info(f"FlashRank initialized with model: {self.model_name}")
            except Exception as e:
                logger.warning(f"FlashRank initialization failed: {e}. Running in pass-through mode.")
                self.ranker = None
        else:
            logger.warning("FlashRank not installed. Running in pass-through score mode.")

    def rerank(
        self,
        query: str,
        chunks: List[SourceChunk],
        top_k: int = settings.RERANK_TOP_K
    ) -> List[SourceChunk]:
        """
        Rerank a list of SourceChunks against the user's query.
        """
        if not chunks:
            return []

        if self.ranker is not None and len(chunks) > 1:
            try:
                passages = [
                    {"id": idx, "text": chunk.text, "meta": chunk.model_dump()}
                    for idx, chunk in enumerate(chunks)
                ]
                rerank_request = RerankRequest(query=query, passages=passages)
                results = self.ranker.rerank(rerank_request)

                reranked_chunks: List[SourceChunk] = []
                for res in results[:top_k]:
                    idx = res["id"]
                    score = float(res.get("score", 0.0))
                    chunk = chunks[idx].model_copy()
                    chunk.rerank_score = round(score, 4)
                    reranked_chunks.append(chunk)

                logger.info(f"FlashRank reranked {len(chunks)} chunks down to {len(reranked_chunks)}")
                return reranked_chunks
            except Exception as e:
                logger.error(f"Error during FlashRank rerank: {e}. Falling back to original order.")

        # Fallback / Pass-through
        for i, chunk in enumerate(chunks[:top_k]):
            chunk.rerank_score = chunk.hybrid_score or chunk.dense_score or round(1.0 - (i * 0.1), 3)

        return chunks[:top_k]
