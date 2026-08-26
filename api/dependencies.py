"""Dependency injection container and singleton services for FastAPI."""

import json
from pathlib import Path
from typing import Dict
from schemas.common import DocumentMetadata
from ingestion.pipeline import IngestionPipeline
from indexing.vector_store import FAISSVectorStore
from indexing.bm25_index import BM25Index
from indexing.hybrid_search import HybridSearchRetriever
from retrieval.query_rewriter import QueryRewriter
from retrieval.reranker import FlashRankReranker
from retrieval.retriever_service import RetrieverService
from agent.llm_factory import get_llm
from agent.graph import AgenticRAGWorkflow
from core.evaluator import RAGEvaluator
from core.config import settings
from core.logger import logger


class Container:
    """Service container managing shared components and document registry."""

    def __init__(self):
        logger.info("Initializing system dependency container...")
        self.doc_registry_file = settings.DATA_DIR / "documents_registry.json"

        # Ingestion
        self.ingestion_pipeline = IngestionPipeline()

        # Vector & Sparse Indexing
        self.vector_store = FAISSVectorStore()
        self.bm25_index = BM25Index()
        self.hybrid_retriever = HybridSearchRetriever(
            vector_store=self.vector_store,
            bm25_index=self.bm25_index
        )

        # LLM & Retrieval Services
        self.llm = get_llm()
        self.query_rewriter = QueryRewriter(llm_client=self.llm)
        self.reranker = FlashRankReranker()
        self.retriever_service = RetrieverService(
            hybrid_retriever=self.hybrid_retriever,
            query_rewriter=self.query_rewriter,
            reranker=self.reranker
        )

        # Agentic LangGraph Workflow
        self.rag_workflow = AgenticRAGWorkflow(
            retriever_service=self.retriever_service,
            llm=self.llm
        )

        # RAG Evaluation Engine
        self.evaluator = RAGEvaluator(
            rag_workflow=self.rag_workflow,
            llm_client=self.llm
        )

        # In-memory document registry mapped to disk
        self.documents_registry: Dict[str, DocumentMetadata] = {}
        self._load_registry()

    def _load_registry(self):
        """Load document metadata registry from disk."""
        if self.doc_registry_file.exists():
            try:
                with open(self.doc_registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.documents_registry = {
                        k: DocumentMetadata(**v) for k, v in data.items()
                    }
                logger.info(f"Loaded {len(self.documents_registry)} documents into registry.")
            except Exception as e:
                logger.warning(f"Failed to load documents registry: {e}")

    def save_registry(self):
        """Persist document metadata registry to disk."""
        try:
            with open(self.doc_registry_file, "w", encoding="utf-8") as f:
                data = {k: v.model_dump() for k, v in self.documents_registry.items()}
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save document registry: {e}")


container = Container()


def get_container() -> Container:
    """Dependency provider for FastAPI route endpoints."""
    return container
