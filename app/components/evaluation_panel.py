"""Evaluation & Benchmark Studio component for Streamlit."""

import json
import time
import requests
import streamlit as st
from typing import Dict, Any, List


DEFAULT_BENCHMARK_PRESETS = [
    {
        "query": "What is the core architecture and orchestration engine of this RAG system?",
        "expected_answer": "The system uses LangGraph for stateful agentic orchestration, FAISS for dense vector search, BM25 for sparse retrieval, and FlashRank for neural reranking.",
        "expected_keywords": ["LangGraph", "FAISS", "BM25", "FlashRank"]
    },
    {
        "query": "How are documents parsed and indexed before retrieval?",
        "expected_answer": "Documents are parsed by specialized parsers (PDF, Markdown, TXT, JSON, HTML), recursively chunked with overlap, embedded using dense embeddings, and indexed into FAISS and BM25.",
        "expected_keywords": ["chunked", "embedded", "FAISS", "BM25"]
    },
    {
        "query": "What hallucination and groundedness checks are performed?",
        "expected_answer": "The system grades retrieved documents for relevance, enforces inline citations in answer generation, and audits claims with a strict hallucination check judge.",
        "expected_keywords": ["hallucination", "citations", "groundedness"]
    }
]


def render_evaluation_panel(backend_url: str):
    """Renders the interactive RAG evaluation and benchmark studio."""
    st.subheader("📊 RAG Evaluation & Benchmark Studio")
    st.caption("Automated quality auditing for Faithfulness, Answer Relevance, Context Precision, and Latency.")

    tab_eval_single, tab_eval_batch = st.tabs([
        "🎯 Single Query Evaluator",
        "🚀 Automated Benchmark Suite"
    ])

    # -------------------------------------------------------------
    # TAB 1: Single Query Evaluator
    # -------------------------------------------------------------
    with tab_eval_single:
        st.markdown("#### Test and Score Individual Query Groundedness")
        c1, c2 = st.columns([3, 2])
        with c1:
            eval_query = st.text_area(
                "Evaluation Query:",
                value="How does the hybrid search retriever combine dense and sparse results?",
                height=90
            )
        with c2:
            ground_truth_ref = st.text_area(
                "Reference / Ground Truth Answer (Optional):",
                value="It uses Reciprocal Rank Fusion (RRF) to merge ranks from FAISS vector search and BM25 sparse keyword search.",
                height=90
            )

        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1:
            enable_rewrite = st.checkbox("Enable Query Rewriting", value=True, key="eval_single_rw")
        with col_opt2:
            enable_rerank = st.checkbox("Enable Reranking", value=True, key="eval_single_rr")
        with col_opt3:
            eval_top_k = st.slider("Top K", min_value=1, max_value=8, value=4, key="eval_single_k")

        if st.button("🧪 Run Single Evaluation", type="primary", use_container_width=True):
            payload = {
                "query": eval_query,
                "ground_truth": ground_truth_ref if ground_truth_ref.strip() else None,
                "enable_query_rewriting": enable_rewrite,
                "enable_reranking": enable_rerank,
                "top_k": eval_top_k
            }
            with st.spinner("Executing RAG pipeline and calculating evaluation metrics..."):
                try:
                    res = requests.post(f"{backend_url}/api/v1/evaluate", json=payload, timeout=60)
                    if res.status_code == 200:
                        data = res.json()
                        st.success("✅ Evaluation Complete!")

                        # Top Metric Cards
                        m1, m2, m3, m4 = st.columns(4)
                        overall = data.get("overall_score", 0.0)
                        latency = data.get("latency_ms", 0.0)
                        faith = data.get("metrics", {}).get("faithfulness", {}).get("score", 0.0)
                        relevance = data.get("metrics", {}).get("answer_relevance", {}).get("score", 0.0)

                        m1.metric("Overall Quality Score", f"{round(overall * 100, 1)}%", delta="Passed" if overall >= 0.70 else "Review")
                        m2.metric("Faithfulness", f"{round(faith * 100, 1)}%")
                        m3.metric("Answer Relevance", f"{round(relevance * 100, 1)}%")
                        m4.metric("Latency", f"{latency} ms")

                        st.markdown("---")
                        st.markdown("#### 📝 Generated Output")
                        st.info(data.get("generated_answer", ""))

                        st.markdown("#### 🔍 Metric Breakdown & Explanations")
                        for m_name, m_val in data.get("metrics", {}).items():
                            with st.expander(f"**{m_val.get('metric_name', m_name)}**: `{round(m_val.get('score', 0)*100, 1)}%` ({'PASSED ✅' if m_val.get('passed') else 'NEEDS REVIEW ⚠️'})"):
                                st.write(f"**Score:** {m_val.get('score')}")
                                st.write(f"**Reasoning:** {m_val.get('reasoning')}")
                    else:
                        st.error(f"Evaluation request failed: {res.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

    # -------------------------------------------------------------
    # TAB 2: Automated Benchmark Suite
    # -------------------------------------------------------------
    with tab_eval_batch:
        st.markdown("#### Multi-Query Quality Benchmark & Regression Testing")
        suite_title = st.text_input("Benchmark Suite Name", value="Core Knowledge & Retrieval Benchmark")

        # Preset test cases display
        st.markdown("##### 📋 Benchmark Test Cases")
        test_cases_json = st.text_area(
            "Test Cases JSON:",
            value=json.dumps(DEFAULT_BENCHMARK_PRESETS, indent=2),
            height=200,
            help="Define test questions with optional reference ground truth."
        )

        if st.button("🚀 Run Full Benchmark Suite", type="primary", use_container_width=True):
            try:
                parsed_cases = json.loads(test_cases_json)
                batch_payload = {
                    "suite_name": suite_title,
                    "test_cases": parsed_cases,
                    "enable_query_rewriting": True,
                    "enable_reranking": True,
                    "top_k": 4
                }
            except Exception as e:
                st.error(f"Invalid JSON format for test cases: {e}")
                return

            with st.spinner(f"Running benchmark suite ({len(parsed_cases)} test cases)..."):
                try:
                    res = requests.post(f"{backend_url}/api/v1/evaluate/batch", json=batch_payload, timeout=180)
                    if res.status_code == 200:
                        report = res.json()
                        st.success(f"🎉 Benchmark '{report['suite_name']}' Completed!")

                        # Summary Metrics
                        c1, c2, c3, c4, c5 = st.columns(5)
                        c1.metric("Overall Score", f"{round(report['overall_benchmark_score'] * 100, 1)}%")
                        c2.metric("Pass Rate", f"{report['passed_queries']}/{report['total_queries']}")
                        c3.metric("Avg Faithfulness", f"{round(report['average_faithfulness'] * 100, 1)}%")
                        c4.metric("Avg Relevance", f"{round(report['average_answer_relevance'] * 100, 1)}%")
                        c5.metric("Avg Latency", f"{report['average_latency_ms']} ms")

                        st.markdown("---")
                        st.markdown("#### 📊 Test Case Results Breakdown")
                        for idx, item in enumerate(report.get("results", []), start=1):
                            status_icon = "✅" if item.get("overall_score", 0) >= 0.70 else "⚠️"
                            with st.expander(f"{status_icon} Test #{idx}: {item.get('query')} (Score: {round(item.get('overall_score', 0)*100, 1)}%)"):
                                st.write(f"**Generated Answer:** {item.get('generated_answer')}")
                                if item.get("ground_truth"):
                                    st.write(f"**Expected Answer:** {item.get('ground_truth')}")
                                st.write(f"**Confidence:** `{item.get('confidence_level')}` | **Latency:** `{item.get('latency_ms')} ms`")
                                st.json(item.get("metrics"))

                        # Export Report Download
                        st.download_button(
                            label="📥 Download Full Benchmark Report (JSON)",
                            data=json.dumps(report, indent=2),
                            file_name=f"rag_benchmark_{int(time.time())}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    else:
                        st.error(f"Benchmark run failed: {res.text}")
                except Exception as e:
                    st.error(f"Connection error during benchmark: {e}")
