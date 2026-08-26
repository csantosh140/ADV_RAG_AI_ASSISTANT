"""LangGraph state graph compiler and execution pipeline for Agentic RAG."""

import time
from typing import Optional, Dict, Any, List

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    END = "__END__"

from schemas.graph_state import AgentGraphState
from schemas.query import QueryResponse, RetrievalAudit
from agent.nodes import (
    node_rewrite_query,
    node_retrieve,
    node_rerank,
    node_grade_documents,
    node_generate_answer,
    node_check_groundedness,
    node_fallback,
)
from core.logger import logger


class FallbackWorkflowExecutor:
    """Sequential executor replicating LangGraph flow if langgraph package is not installed."""

    def __init__(self, retriever_service, llm):
        self.retriever_service = retriever_service
        self.llm = llm

    def invoke(self, state: AgentGraphState) -> AgentGraphState:
        # Step 1: Rewrite
        state.update(node_rewrite_query(state, self.retriever_service))
        # Step 2: Retrieve
        state.update(node_retrieve(state, self.retriever_service))
        # Step 3: Rerank
        state.update(node_rerank(state, self.retriever_service))
        # Step 4: Grade
        state.update(node_grade_documents(state, self.llm))

        filtered = state.get("filtered_chunks", [])
        if not filtered:
            state.update(node_fallback(state))
            return state

        # Step 5: Generate
        state.update(node_generate_answer(state, self.llm))
        # Step 6: Groundedness Check
        state.update(node_check_groundedness(state, self.llm))

        if not state.get("is_grounded", True):
            state.update(node_fallback(state))

        return state


class AgenticRAGWorkflow:
    """Compiled LangGraph orchestrator executing the full self-reflective RAG loop."""

    def __init__(self, retriever_service, llm):
        self.retriever_service = retriever_service
        self.llm = llm
        self.graph = self._build_graph()

    def _build_graph(self):
        """Construct the LangGraph workflow with fallback."""
        if not LANGGRAPH_AVAILABLE:
            logger.info("LangGraph package not installed. Using sequential workflow executor.")
            return FallbackWorkflowExecutor(self.retriever_service, self.llm)

        workflow = StateGraph(AgentGraphState)

        # Add Nodes
        workflow.add_node("rewrite", lambda state: node_rewrite_query(state, self.retriever_service))
        workflow.add_node("retrieve", lambda state: node_retrieve(state, self.retriever_service))
        workflow.add_node("rerank", lambda state: node_rerank(state, self.retriever_service))
        workflow.add_node("grade_docs", lambda state: node_grade_documents(state, self.llm))
        workflow.add_node("generate", lambda state: node_generate_answer(state, self.llm))
        workflow.add_node("groundedness_check", lambda state: node_check_groundedness(state, self.llm))
        workflow.add_node("fallback", lambda state: node_fallback(state))

        # Define Edges
        workflow.set_entry_point("rewrite")
        workflow.add_edge("rewrite", "retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "grade_docs")

        def decide_to_generate(state: AgentGraphState) -> str:
            filtered = state.get("filtered_chunks", [])
            if not filtered:
                return "fallback"
            return "generate"

        workflow.add_conditional_edges(
            "grade_docs",
            decide_to_generate,
            {
                "generate": "generate",
                "fallback": "fallback",
            }
        )

        workflow.add_edge("generate", "groundedness_check")

        def decide_groundedness_verdict(state: AgentGraphState) -> str:
            is_grounded = state.get("is_grounded", True)
            if not is_grounded:
                return "fallback"
            return END

        workflow.add_conditional_edges(
            "groundedness_check",
            decide_groundedness_verdict,
            {
                END: END,
                "fallback": "fallback",
            }
        )

        workflow.add_edge("fallback", END)

        return workflow.compile()

    def run(
        self,
        query: str,
        doc_ids: Optional[List[str]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 4,
        enable_rewriting: bool = True,
        enable_reranking: bool = True,
    ) -> QueryResponse:
        """Executes the Agentic RAG workflow and returns a validated QueryResponse."""
        start_time = time.time()
        initial_state: AgentGraphState = {
            "original_query": query,
            "chat_history": chat_history,
            "rewritten_queries": [query] if not enable_rewriting else [],
            "doc_ids_filter": doc_ids,
            "raw_chunks": [],
            "reranked_chunks": [],
            "filtered_chunks": [],
            "answer": "",
            "citations": [],
            "groundedness_score": 0.0,
            "is_grounded": True,
            "confidence_level": "HIGH",
            "retry_count": 0,
            "execution_time_ms": 0.0,
        }

        final_state = self.graph.invoke(initial_state)
        elapsed_ms = (time.time() - start_time) * 1000

        audit = RetrievalAudit(
            original_query=query,
            rewritten_queries=final_state.get("rewritten_queries", [query]),
            raw_retrieved_chunks_count=len(final_state.get("raw_chunks", [])),
            reranked_chunks_count=len(final_state.get("reranked_chunks", [])),
            graded_relevant_chunks_count=len(final_state.get("filtered_chunks", [])),
            execution_time_ms=round(elapsed_ms, 2),
            groundedness_check_passed=final_state.get("is_grounded", True),
        )

        return QueryResponse(
            answer=final_state.get("answer", ""),
            confidence_level=final_state.get("confidence_level", "HIGH"),
            citations=final_state.get("citations", []),
            sources=final_state.get("filtered_chunks", []) or final_state.get("reranked_chunks", []),
            audit=audit,
            time_taken_ms=round(elapsed_ms, 2),
        )
