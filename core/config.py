"""System configuration management using Pydantic Settings."""

import os
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable override support."""

    # Project metadata
    PROJECT_NAME: str = "Advanced RAG AI System"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"

    # LLM Settings
    LLM_PROVIDER: Literal["openai", "groq", "google", "ollama", "mock"] = "mock"
    OPENAI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.0

    # Embeddings
    EMBEDDING_PROVIDER: Literal["sentence-transformers", "openai", "google", "mock"] = "sentence-transformers"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # Chunking & Preprocessing
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 100
    MIN_CHUNK_LENGTH: int = 50

    # Retrieval & Reranking
    RETRIEVAL_TOP_K: int = 8
    RERANK_TOP_K: int = 4
    HYBRID_ALPHA: float = 0.5  # Weight for dense (1.0 = pure dense, 0.0 = pure sparse)
    SIMILARITY_THRESHOLD: float = 0.30
    RERANKER_MODEL: str = "ms-marco-TinyBERT-L-2-v2"

    # Storage Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
    RAW_DOCS_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "raw_documents"
    FAISS_INDEX_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "faiss_indexes"

    # API & Server Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    BACKEND_API_URL: str = "http://localhost:8000"
    FRONTEND_PORT: int = 8501
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def init_directories(self) -> None:
        """Ensure all storage directories exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
        self.FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.init_directories()
