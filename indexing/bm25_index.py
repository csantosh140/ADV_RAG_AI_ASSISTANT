"""BM25 sparse index with pure-python fallback for keyword retrieval."""

import math
import re
from typing import List, Dict, Any, Optional
from collections import Counter
from schemas.common import SourceChunk
from core.logger import logger

try:
    from rank_bm25 import BM25Okapi
    RANK_BM25_AVAILABLE = True
except ImportError:
    RANK_BM25_AVAILABLE = False


class PureBM25:
    """Lightweight pure-python BM25 implementation when external library is absent."""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / max(1, self.corpus_size)
        self.doc_freqs: List[Counter] = [Counter(doc) for doc in corpus]
        self.doc_len = [len(doc) for doc in corpus]
        self.nd: Dict[str, int] = Counter()
        for doc in corpus:
            for word in set(doc):
                self.nd[word] += 1

    def get_scores(self, query: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        for q in query:
            if q not in self.nd:
                continue
            # Standard IDF
            idf = math.log((self.corpus_size - self.nd[q] + 0.5) / (self.nd[q] + 0.5) + 1.0)
            for idx in range(self.corpus_size):
                freq = self.doc_freqs[idx].get(q, 0)
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * (self.doc_len[idx] / max(1.0, self.avgdl)))
                scores[idx] += idf * (numerator / max(1e-6, denominator))
        return scores


class BM25Index:
    """BM25 sparse keyword retriever for technical terminology and exact matches."""

    def __init__(self):
        self.bm25 = None
        self.chunks: List[SourceChunk] = []

    def _tokenize(self, text: str) -> List[str]:
        """Lowercases and extracts alpha-numeric token sequence."""
        return re.findall(r"\w+", text.lower())

    def build_index(self, chunks: List[SourceChunk]):
        """Build BM25 index from a list of SourceChunks."""
        self.chunks = chunks
        if not chunks:
            self.bm25 = None
            return

        corpus = [self._tokenize(c.text) for c in chunks]
        if RANK_BM25_AVAILABLE:
            self.bm25 = BM25Okapi(corpus)
        else:
            self.bm25 = PureBM25(corpus)
        logger.info(f"Built BM25 index with {len(chunks)} documents (rank_bm25={RANK_BM25_AVAILABLE}).")

    def search(
        self,
        query: str,
        top_k: int = 10,
        doc_ids: Optional[List[str]] = None
    ) -> List[SourceChunk]:
        """Query BM25 index and return scored chunks."""
        if not self.bm25 or not self.chunks:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: List[SourceChunk] = []
        for idx in ranked_indices:
            score = scores[idx]
            if score <= 0:
                continue

            chunk = self.chunks[idx].model_copy()
            if doc_ids and chunk.doc_id not in doc_ids:
                continue

            chunk.sparse_score = float(score)
            results.append(chunk)
            if len(results) >= top_k:
                break

        return results
