"""Modern, high-aesthetic Streamlit Application for Advanced Agentic RAG."""

import os
import sys
from pathlib import Path

# Ensure both project root and app directory are in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
for p in [str(ROOT_DIR), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st

try:
    from app.components.chat_ui import render_chat_interface
    from app.components.document_manager import render_document_manager
    from app.components.observability_panel import render_observability_panel
    from app.components.evaluation_panel import render_evaluation_panel
except (ImportError, ModuleNotFoundError):
    from components.chat_ui import render_chat_interface
    from components.document_manager import render_document_manager
    from components.observability_panel import render_observability_panel
    from components.evaluation_panel import render_evaluation_panel

# Set Page Config
st.set_page_config(
    page_title="Advanced Agentic RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Glassmorphism & Modern Dark/Vibrant Elements)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .main {
        background: radial-gradient(circle at 10% 20%, rgb(18, 24, 38) 0%, rgb(11, 15, 25) 90.2%);
        color: #f1f5f9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        border-radius: 8px 8px 0px 0px;
        font-weight: 600;
        background-color: rgba(255, 255, 255, 0.04);
        color: #94a3b8;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.2s ease-in-out;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.08);
        color: #e2e8f0;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.25) 0%, rgba(99, 102, 241, 0.25) 100%) !important;
        color: #60a5fa !important;
        border: 1px solid rgba(96, 165, 250, 0.4) !important;
        border-bottom: 2px solid #3b82f6 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .badge-chip {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Configuration & Backend URL
BACKEND_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/isometric/512/brain.png", width=64)
    st.title("🧠 Agentic RAG")
    st.caption("Production-Grade Self-Correcting RAG System")

    st.markdown("---")
    st.markdown("### ⚙️ System Pipeline")
    st.markdown("""
    - **Orchestration:** LangGraph State Machine
    - **Query Optimization:** Multi-Query & Context HyDE
    - **Dense Index:** FAISS FlatIP (Normalized Cosine)
    - **Sparse Index:** BM25 Okapi (Technical terms)
    - **Hybrid Fusion:** Reciprocal Rank Fusion (RRF)
    - **Cross-Encoder:** FlashRank Neural Reranker
    - **Verification:** Self-reflective Hallucination Audit
    """)

    st.markdown("---")
    st.markdown(f"**Backend Service:** `{BACKEND_URL}`")

# Main Header
st.title("⚡ Advanced Agentic RAG Assistant")
st.caption("Multi-format Ingestion • Hybrid Retrieval • FlashRank Reranking • Self-Reflective Verification • Quality Benchmarking")

# Main Content Tabs
tab_chat, tab_docs, tab_obs, tab_eval = st.tabs([
    "💬 Grounded Chat",
    "📁 Document Management",
    "🔍 Retrieval Inspector",
    "📊 Evaluation Studio"
])

with tab_chat:
    render_chat_interface(BACKEND_URL)

with tab_docs:
    render_document_manager(BACKEND_URL)

with tab_obs:
    audit = st.session_state.get("latest_audit")
    sources = st.session_state.get("latest_sources")
    citations = st.session_state.get("latest_citations")
    render_observability_panel(audit, sources, citations)

with tab_eval:
    render_evaluation_panel(BACKEND_URL)
