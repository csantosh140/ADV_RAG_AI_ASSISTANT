"""Observability, Chunk Inspector, and Audit Trail Component."""

import streamlit as st
from typing import Dict, Any, List


def render_observability_panel(audit_data: Dict[str, Any] | None, sources: List[Dict[str, Any]] | None, citations: List[Dict[str, Any]] | None):
    """Renders visual observability audit trail and chunk inspector."""
    st.subheader("🔍 Retrieval & Groundedness Inspector")

    if not audit_data and not sources:
        st.info("Ask a question in the Chat tab to inspect the Agentic RAG execution trace.")
        return

    # Confidence Badge
    if audit_data:
        passed = audit_data.get("groundedness_check_passed", True)
        exec_time = audit_data.get("execution_time_ms", 0)

        c1, c2, c3 = st.columns(3)
        c1.metric("Execution Latency", f"{exec_time} ms")
        c2.metric("Retrieved Chunks", audit_data.get("raw_retrieved_chunks_count", 0))
        c3.metric("Groundedness Audit", "PASSED ✅" if passed else "FAILED ⚠️")

        # Query Rewriting Step
        st.markdown("#### 🔄 Query Transformation Trace")
        st.write(f"**Original Query:** `{audit_data.get('original_query', '')}`")
        rewritten = audit_data.get("rewritten_queries", [])
        if rewritten:
            st.write("**Expanded Retrieval Queries:**")
            for rq in rewritten:
                st.markdown(f"- 🔎 `{rq}`")

    # Citations List
    if citations:
        st.markdown("---")
        st.markdown("#### 📌 Grounded Citations")
        for cit in citations:
            with st.container():
                st.markdown(
                    f"**[{cit['citation_id']}]** `{cit['filename']}` "
                    f"(Page: {cit.get('page_number') or 'N/A'}, Section: *{cit.get('section_title') or 'N/A'}*)"
                )
                st.caption(f"\"{cit.get('snippet', '')}\"")

    # Verified Chunks Inspector
    if sources:
        st.markdown("---")
        st.markdown("#### 🧩 Source Chunk Inspector (FAISS + BM25 + FlashRank)")
        for idx, chunk in enumerate(sources, start=1):
            with st.expander(
                f"Chunk {idx}: {chunk.get('filename', 'Doc')} | Rerank Score: {chunk.get('rerank_score', 'N/A')}"
            ):
                col_a, col_b, col_c = st.columns(3)
                col_a.write(f"**Dense Score:** {chunk.get('dense_score', 'N/A')}")
                col_b.write(f"**BM25 Score:** {chunk.get('sparse_score', 'N/A')}")
                col_c.write(f"**RRF Score:** {chunk.get('hybrid_score', 'N/A')}")

                st.markdown("**Chunk Text:**")
                st.text_area(
                    label=f"Content ({chunk.get('token_count', 0)} tokens)",
                    value=chunk.get("text", ""),
                    height=140,
                    key=f"chunk_txt_{idx}_{chunk.get('chunk_id')}"
                )
