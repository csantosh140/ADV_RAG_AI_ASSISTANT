"""Interactive Grounded Chat Interface with SSE Streaming, Multi-turn Memory, and Quick Test Prompts."""

import json
import time
import requests
import streamlit as st
from typing import Dict, Any, List


def _stream_response(backend_url: str, payload: dict):
    """
    Generator consuming Server-Sent Events (SSE) from the backend streaming endpoint.
    Yields chunks of text and updates session state with metadata when received.
    """
    try:
        response = requests.post(
            f"{backend_url}/api/v1/query/stream",
            json=payload,
            stream=True,
            timeout=120
        )
        if response.status_code != 200:
            yield f"❌ Server returned status {response.status_code}: {response.text}"
            return

        has_yielded = False
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            raw_line = line.strip()
            while raw_line.startswith("data:"):
                raw_line = raw_line[len("data:"):].strip()

            if not raw_line:
                continue

            try:
                event_data = json.loads(raw_line)
                event_type = event_data.get("event")

                if event_type == "token":
                    token = event_data.get("data", "")
                    if token:
                        has_yielded = True
                        yield token
                elif event_type == "metadata":
                    st.session_state.latest_citations = event_data.get("citations", [])
                    st.session_state.latest_sources = event_data.get("sources", [])
                    st.session_state.latest_audit = event_data.get("audit")
                    st.session_state.latest_confidence = event_data.get("confidence_level", "HIGH")
                    st.session_state.latest_latency = event_data.get("time_taken_ms", 0)
                elif event_type == "error":
                    has_yielded = True
                    yield f"\n\n❌ Error: {event_data.get('data', 'Unknown error')}"
            except json.JSONDecodeError:
                # If non-JSON text came in, yield it directly
                if raw_line and not raw_line.startswith("{"):
                    has_yielded = True
                    yield raw_line + " "

        if not has_yielded:
            # Fallback if stream was empty
            yield "Response completed."

    except Exception as e:
        yield f"\n\n❌ Connection error during stream: {str(e)}"


def render_chat_interface(backend_url: str):
    """Renders the conversational interface with grounded answers, citations, and controls."""
    st.subheader("💬 Grounded Q&A Assistant")

    # Document-scoped filter selector & query controls
    try:
        doc_res = requests.get(f"{backend_url}/api/v1/documents", timeout=3)
        doc_options = {}
        total_indexed_chunks = 0
        if doc_res.status_code == 200:
            doc_data = doc_res.json()
            total_indexed_chunks = doc_data.get("total_chunks", 0)
            for d in doc_data.get("documents", []):
                doc_options[d["doc_id"]] = d["filename"]
    except Exception:
        doc_options = {}
        total_indexed_chunks = 0

    # Initialize chat state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "latest_audit" not in st.session_state:
        st.session_state.latest_audit = None
    if "latest_sources" not in st.session_state:
        st.session_state.latest_sources = []
    if "latest_citations" not in st.session_state:
        st.session_state.latest_citations = []
    if "latest_confidence" not in st.session_state:
        st.session_state.latest_confidence = "HIGH"
    if "latest_latency" not in st.session_state:
        st.session_state.latest_latency = 0.0

    # Helpful guide if no documents exist or to assist first-time users
    if not doc_options or total_indexed_chunks == 0:
        st.warning(
            "⚠️ **No documents are indexed yet!**\n\n"
            "To test the assistant, go to the **📁 Document Management** tab above and upload a document "
            "(PDF, DOCX, Markdown, TXT, CSV, or Web URL). Once indexed, return here to ask questions."
        )

    # Controls Expander
    with st.expander("⚙️ Pipeline Controls & Document Scoping", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            chosen_names = st.multiselect(
                "Filter retrieval to specific documents:",
                options=list(doc_options.values()),
                default=[]
            )
            selected_doc_ids = [k for k, v in doc_options.items() if v in chosen_names]
            top_k_val = st.slider("Retrieval Top-K Candidates", min_value=1, max_value=10, value=4)

        with col2:
            enable_rewriting = st.toggle("Multi-Query Expansion / HyDE", value=True)
            enable_reranking = st.toggle("FlashRank Neural Reranking", value=True)
            enable_streaming = st.toggle("Real-Time SSE Streaming", value=True)

    # Action Toolbar & Quick Starters
    col_tools_1, col_tools_2, col_tools_3 = st.columns([1, 1.2, 1.2])
    with col_tools_1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.latest_audit = None
            st.session_state.latest_sources = []
            st.session_state.latest_citations = []
            st.rerun()

    with col_tools_2:
        if st.session_state.messages:
            md_export = "# RAG Assistant Chat Transcript\n\n"
            for m in st.session_state.messages:
                role = m['role'].capitalize()
                md_export += f"### {role}\n{m['content']}\n\n"
            st.download_button(
                label="📥 Export (MD)",
                data=md_export,
                file_name="rag_chat_transcript.md",
                mime="text/markdown",
                use_container_width=True
            )

    with col_tools_3:
        if st.session_state.messages:
            session_data = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_turns": len(st.session_state.messages),
                "messages": st.session_state.messages,
                "latest_audit": st.session_state.get("latest_audit"),
                "latest_sources": st.session_state.get("latest_sources"),
                "latest_citations": st.session_state.get("latest_citations"),
            }
            st.download_button(
                label="📥 Export (JSON)",
                data=json.dumps(session_data, indent=2),
                file_name="rag_session.json",
                mime="application/json",
                use_container_width=True
            )

    # Quick test suggestions
    if doc_options:
        st.markdown("**💡 Quick Test Questions (Click to ask):**")
        q_cols = st.columns(3)
        quick_query = None
        with q_cols[0]:
            if st.button("📋 Summarize Document", use_container_width=True):
                quick_query = "Please provide a comprehensive summary of the indexed documents."
        with q_cols[1]:
            if st.button("🛠️ Technology & Architecture", use_container_width=True):
                quick_query = "What is the technology stack and architecture described in the project?"
        with q_cols[2]:
            if st.button("🎯 Objectives & Scope", use_container_width=True):
                quick_query = "What are the main objectives, features, and results of this project?"

        if quick_query:
            st.session_state.messages.append({"role": "user", "content": quick_query})
            st.rerun()

    st.markdown("---")

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "confidence" in msg and msg["confidence"]:
                badge_color = {
                    "HIGH": "🟢",
                    "MEDIUM": "🟡",
                    "LOW": "🔴",
                    "ABSTAINED": "⚪",
                }.get(msg["confidence"], "⚪")
                st.caption(f"{badge_color} **Confidence:** `{msg['confidence']}` | Latency: `{msg.get('latency', 0)} ms`")

    # Determine if last turn needs assistant response (e.g. from quick test button)
    should_respond = False
    active_prompt = None

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        should_respond = True
        active_prompt = st.session_state.messages[-1]["content"]

    # Chat Input Box
    prompt = st.chat_input("Ask a grounded question about your indexed documents...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        active_prompt = prompt
        should_respond = True
        st.rerun()

    if should_respond and active_prompt:
        history_payload = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1][-6:]  # Previous history
        ]

        with st.chat_message("assistant"):
            payload = {
                "query": active_prompt,
                "doc_ids": selected_doc_ids if selected_doc_ids else None,
                "chat_history": history_payload if history_payload else None,
                "top_k": top_k_val,
                "enable_query_rewriting": enable_rewriting,
                "enable_reranking": enable_reranking,
            }

            if enable_streaming:
                with st.status("Executing Agentic RAG Pipeline...", expanded=False) as status_box:
                    st.write("1. 🔄 Context-aware query expansion & multi-query reformulation...")
                    st.write("2. 🔍 Hybrid dense vector search + sparse search (BM25)...")
                    st.write("3. ⚡ Neural Cross-Encoder Reranking (FlashRank)...")
                    st.write("4. 🛡️ Document relevance grading & hallucination auditing...")
                    status_box.update(label="Agentic RAG pipeline complete!", state="complete")

                stream_gen = _stream_response(backend_url, payload)
                answer = st.write_stream(stream_gen)
                confidence = st.session_state.get("latest_confidence", "HIGH")
                latency = st.session_state.get("latest_latency", 0.0)

                badge_color = {
                    "HIGH": "🟢",
                    "MEDIUM": "🟡",
                    "LOW": "🔴",
                    "ABSTAINED": "⚪",
                }.get(confidence, "⚪")
                st.caption(f"{badge_color} **Confidence:** `{confidence}` | Latency: `{latency} ms`")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer or "Response generated.",
                    "confidence": confidence,
                    "latency": latency,
                })
                st.rerun()
            else:
                with st.spinner("Executing Agentic RAG workflow..."):
                    start_t = time.time()
                    try:
                        res = requests.post(f"{backend_url}/api/v1/query", json=payload, timeout=60)
                        if res.status_code == 200:
                            data = res.json()
                            answer = data.get("answer", "")
                            confidence = data.get("confidence_level", "HIGH")
                            latency = data.get("time_taken_ms", round((time.time() - start_t) * 1000, 1))

                            st.markdown(answer)

                            st.session_state.latest_audit = data.get("audit")
                            st.session_state.latest_sources = data.get("sources")
                            st.session_state.latest_citations = data.get("citations")

                            badge_color = {
                                "HIGH": "🟢",
                                "MEDIUM": "🟡",
                                "LOW": "🔴",
                                "ABSTAINED": "⚪",
                            }.get(confidence, "⚪")
                            st.caption(f"{badge_color} **Confidence:** `{confidence}` | Latency: `{latency} ms`")

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "confidence": confidence,
                                "latency": latency,
                            })
                            st.rerun()
                        else:
                            err_msg = f"❌ Error ({res.status_code}): {res.text}"
                            st.error(err_msg)
                            st.session_state.messages.append({"role": "assistant", "content": err_msg})
                            st.rerun()
                    except Exception as e:
                        err_msg = f"❌ Request failed: {e}"
                        st.error(err_msg)
                        st.session_state.messages.append({"role": "assistant", "content": err_msg})
                        st.rerun()
