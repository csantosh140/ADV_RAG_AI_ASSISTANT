"""Embedding service supporting SentenceTransformers, OpenAI, Google, and lightweight offline mode."""

from typing import List
import numpy as np
from core.config import settings
from core.logger import logger


class EmbeddingService:
    """Wrapper providing normalized dense embeddings for documents and queries."""

    def __init__(self):
        self.provider = settings.EMBEDDING_PROVIDER
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.dimension = settings.EMBEDDING_DIMENSION
        self._model = None
        self._init_provider()

    def _init_provider(self):
        """Lazy load embedding models with graceful fallback."""
        if self.provider == "google" and settings.GOOGLE_API_KEY:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                model = self.model_name if "embedding" in self.model_name else "models/text-embedding-004"
                self._model = GoogleGenerativeAIEmbeddings(
                    model=model,
                    google_api_key=settings.GOOGLE_API_KEY,
                )
                self.dimension = 768  # text-embedding-004 output dimension
                logger.info(f"Initialized Google Generative AI Embeddings ({model})")
                return
            except Exception as e:
                logger.warning(f"Failed to init Google embeddings: {e}, falling back to local.")

        if self.provider == "openai" and settings.OPENAI_API_KEY:
            try:
                from langchain_openai import OpenAIEmbeddings
                self._model = OpenAIEmbeddings(
                    model=self.model_name,
                    openai_api_key=settings.OPENAI_API_KEY
                )
                logger.info(f"Initialized OpenAI Embeddings ({self.model_name})")
                return
            except Exception as e:
                logger.warning(f"Failed to init OpenAI embeddings: {e}, falling back to local.")

        if self.provider == "sentence-transformers" or self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                if hasattr(self._model, "get_embedding_dimension"):
                    self.dimension = self._model.get_embedding_dimension()
                elif hasattr(self._model, "get_sentence_embedding_dimension"):
                    self.dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"SentenceTransformer loaded. Dimension: {self.dimension}")
                return
            except Exception as e:
                logger.warning(f"SentenceTransformer unavailable: {e}. Using deterministic mock embedding.")
                self.provider = "mock"

        # Mock fallback for test / offline without heavy weights
        logger.info(f"Using lightweight deterministic embeddings (dim={self.dimension})")

    def _mock_embed(self, text: str) -> List[float]:
        """Generate deterministic pseudo-embedding vector for offline testing."""
        np.random.seed(abs(hash(text)) % (2**32))
        vec = np.random.randn(self.dimension).astype("float32")
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a list of document chunks."""
        if not texts:
            return []

        if self.provider in ("openai", "google") and hasattr(self._model, "embed_documents"):
            return self._model.embed_documents(texts)
        elif self.provider == "sentence-transformers" and self._model is not None:
            embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return [vec.tolist() for vec in embeddings]
        else:
            return [self._mock_embed(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        """Compute embedding for a single search query."""
        if self.provider in ("openai", "google") and hasattr(self._model, "embed_query"):
            return self._model.embed_query(query)
        elif self.provider == "sentence-transformers" and self._model is not None:
            emb = self._model.encode(query, normalize_embeddings=True, show_progress_bar=False)
            return emb.tolist()
        else:
            return self._mock_embed(query)
