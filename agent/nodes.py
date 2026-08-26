"""LangGraph workflow node functions for Agentic RAG."""

import re
import time
from typing import Dict, Any, List

from schemas.graph_state import AgentGraphState
from schemas.common import SourceChunk, Citation
from agent.prompts import (
    QUERY_REWRITE_PROMPT,
    DOCUMENT_GRADER_PROMPT,
    GROUNDED_GENERATION_PROMPT,
    HALLUCINATION_CHECK_PROMPT,
)
from core.logger import logger


def node_rewrite_query(state: AgentGraphState, retriever_service) -> Dict[str, Any]:
    """Expands/refines the original query for higher-recall retrieval, incorporating conversational history."""
    query = state.get("original_query", "")
    chat_history = state.get("chat_history")
    logger.info(f"--- NODE: REWRITE QUERY ('{query}') ---")

    queries = retriever_service.query_rewriter.rewrite_query(query, chat_history=chat_history)
    return {"rewritten_queries": queries}


def node_retrieve(state: AgentGraphState, retriever_service) -> Dict[str, Any]:
    """Retrieves candidates using hybrid FAISS + BM25 across queries."""
    queries = state.get("rewritten_queries") or [state.get("original_query", "")]
    doc_ids = state.get("doc_ids_filter")
    logger.info(f"--- NODE: HYBRID RETRIEVAL across {len(queries)} queries ---")

    all_chunks: List[SourceChunk] = []
    seen_ids = set()

    for q in queries:
        chunks = retriever_service.hybrid_retriever.retrieve(
            query=q,
            top_k=8,
            doc_ids=doc_ids
        )
        for c in chunks:
            if c.chunk_id not in seen_ids:
                seen_ids.add(c.chunk_id)
                all_chunks.append(c)

    return {"raw_chunks": all_chunks}


def node_rerank(state: AgentGraphState, retriever_service) -> Dict[str, Any]:
    """Applies FlashRank neural cross-encoder to rank the raw candidates."""
    query = state.get("original_query", "")
    raw_chunks = state.get("raw_chunks", [])
    logger.info(f"--- NODE: RERANK {len(raw_chunks)} candidates ---")

    reranked = retriever_service.reranker.rerank(
        query=query,
        chunks=raw_chunks,
        top_k=4
    )
    return {"reranked_chunks": reranked}


def node_grade_documents(state: AgentGraphState, llm) -> Dict[str, Any]:
    """Grades relevance of each candidate chunk to filter out noise."""
    query = state.get("original_query", "")
    reranked_chunks = state.get("reranked_chunks", [])
    logger.info(f"--- NODE: GRADE DOCUMENTS ({len(reranked_chunks)} chunks) ---")

    filtered: List[SourceChunk] = []
    for chunk in reranked_chunks:
        # Prompt LLM grader or perform heuristic score check
        prompt = DOCUMENT_GRADER_PROMPT.format(query=query, document_text=chunk.text)
        try:
            res = llm.invoke(prompt)
            verdict = res.content if hasattr(res, "content") else str(res)
            if "YES" in verdict.upper():
                filtered.append(chunk)
            else:
                logger.debug(f"Chunk {chunk.chunk_id} rejected by grader.")
        except Exception as e:
            logger.warning(f"Grading failed: {e}. Retaining chunk by default.")
            filtered.append(chunk)

    # Fallback to keep top reranked chunks if all were rejected
    if not filtered and reranked_chunks:
        filtered = reranked_chunks[:2]

    return {"filtered_chunks": filtered}


def node_generate_answer(state: AgentGraphState, llm) -> Dict[str, Any]:
    """Generates grounded answer with explicit citations based on verified chunks."""
    query = state.get("original_query", "")
    chunks = state.get("filtered_chunks", [])
    logger.info(f"--- NODE: GENERATE ANSWER using {len(chunks)} verified chunks ---")

    if not chunks:
        return {
            "answer": "I do not have sufficient information in the provided documents to answer this question.",
            "citations": [],
            "confidence_level": "ABSTAINED",
            "is_grounded": True,
            "groundedness_score": 1.0,
        }

    # Format context blocks with citation IDs [1], [2]
    context_blocks = ""
    citations: List[Citation] = []

    for i, c in enumerate(chunks, start=1):
        context_blocks += f"[{i}] File: {c.filename} (Page {c.page_number or 'N/A'}, Section: {c.section_title or 'N/A'})\n{c.text}\n\n"
        citations.append(Citation(
            citation_id=i,
            doc_id=c.doc_id,
            filename=c.filename,
            page_number=c.page_number,
            section_title=c.section_title,
            snippet=c.text[:200] + "..." if len(c.text) > 200 else c.text
        ))

    prompt = GROUNDED_GENERATION_PROMPT.format(
        context_blocks=context_blocks,
        query=query
    )

    try:
        response = llm.invoke(prompt)
        answer_text = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.error(f"LLM Generation failed: {e}")
        answer_text = "An error occurred while generating the answer. Please try again."

    return {
        "answer": answer_text,
        "citations": citations,
    }


def node_check_groundedness(state: AgentGraphState, llm) -> Dict[str, Any]:
    """Audits generated answer against source context for hallucinations."""
    answer = state.get("answer", "")
    chunks = state.get("filtered_chunks", [])
    logger.info("--- NODE: CHECK GROUNDEDNESS / HALLUCINATIONS ---")

    if not chunks or "do not have sufficient information" in answer.lower():
        return {
            "is_grounded": True,
            "groundedness_score": 1.0,
            "confidence_level": "ABSTAINED",
        }

    context_blocks = "\n\n".join([f"[{i+1}] {c.text}" for i, c in enumerate(chunks)])
    prompt = HALLUCINATION_CHECK_PROMPT.format(
        context_blocks=context_blocks,
        answer=answer
    )

    score = 0.90
    is_grounded = True
    try:
        verdict = llm.invoke(prompt)
        v_text = verdict.content if hasattr(verdict, "content") else str(verdict)

        # Parse score and decision
        score_match = re.search(r"SCORE:\s*([0-9.]+)", v_text)
        if score_match:
            score = float(score_match.group(1))

        if "GROUNDED: NO" in v_text.upper() or score < 0.5:
            is_grounded = False
    except Exception as e:
        logger.warning(f"Groundedness check error: {e}")

    confidence = "HIGH" if (score >= 0.8 and is_grounded) else "MEDIUM" if is_grounded else "LOW"

    return {
        "is_grounded": is_grounded,
        "groundedness_score": score,
        "confidence_level": confidence,
    }


def node_fallback(state: AgentGraphState) -> Dict[str, Any]:
    """Handles low-confidence / ungrounded fallbacks."""
    logger.info("--- NODE: FALLBACK / ABSTENTION ---")
    return {
        "answer": (
            "I could not find sufficient or verified context in the indexed documents "
            "to answer your question with high confidence. Please verify your indexed files or refine your query."
        ),
        "citations": [],
        "confidence_level": "ABSTAINED",
        "is_grounded": True,
        "groundedness_score": 0.0,
    }
