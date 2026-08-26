# 🧠 Production-Style Advanced Agentic RAG AI System

An enterprise-ready, grounded Retrieval-Augmented Generation (RAG) assistant engineered with **FastAPI**, **Streamlit**, **LangGraph**, **FAISS**, **BM25**, **FlashRank Reranking**, **Pydantic V2**, and **Docker**.

---

## 🌟 Key Architectural Features

- **Multi-Format Ingestion Engine**:
  - Native page-level parsing for **PDF** files with metadata extraction (`pypdf`, `pdfplumber`).
  - Heading & section-aware parsing for **Markdown** (`.md`).
  - Encoding-safe stream parsing for **Plaintext** (`.txt`, `.csv`, `.log`).
  - Recursive chunking preserving section lineage, page numbers, and token counts.

- **Hybrid Retrieval & Reranking Pipeline**:
  - **Dense Vector Search**: FAISS Index Flat Inner Product (cosine similarity).
  - **Sparse Keyword Search**: BM25 Okapi for exact term matching and technical jargon.
  - **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse score distributions.
  - **FlashRank Neural Cross-Encoder**: Lightweight, ultra-fast reranking of top hybrid candidates.

- **Agentic LangGraph Workflow**:
  - **Query Transformation / HyDE**: Expands conversational queries into targeted retrieval variants.
  - **Document Relevance Grading**: Evaluates retrieved chunks and discards irrelevant noise.
  - **Grounded Answer Generation**: Strictly enforces citations (`[1]`, `[2]`) tied to specific source chunks.
  - **Hallucination & Groundedness Audit**: Evaluates generated claims against context; abstains or self-corrects if ungrounded.

- **Production-Grade API & Typed Schemas**:
  - Strongly-typed Pydantic V2 request, response, and audit schemas.
  - Synchronous (`POST /api/v1/query`) and SSE Streaming (`POST /api/v1/query/stream`) endpoints.
  - Full document lifecycle management (Upload, List, Delete, Metrics).

- **Rich Observability Frontend**:
  - Modern dark-mode Streamlit dashboard.
  - **Visual Trace Inspector**: View original vs rewritten queries, latency, and groundedness verdict.
  - **Chunk Inspector**: Examine raw text, BM25 scores, FAISS scores, and FlashRank rerank scores.
  - Document management studio with chunk statistics and index controls.

---

## 🏗️ Architecture Diagram

```
User Query / Document Upload
       │
       ▼
 ┌─────────────┐
 │ Streamlit UI│ ───► ┌─────────────┐
 └─────────────┘      │ FastAPI API │
                      └──────┬──────┘
                             │
     ┌───────────────────────┴───────────────────────┐
     ▼                                               ▼
[ Ingestion Pipeline ]                      [ Agentic LangGraph ]
 ├─ Parsers (PDF/MD/TXT)                     ├─ 1. Query Rewriter
 ├─ Recursive Chunker                        ├─ 2. Hybrid Retrieval (FAISS + BM25)
 ├─ Dense Embeddings (SentenceTransformers)  ├─ 3. FlashRank Neural Reranker
 └─ Persistent FAISS Index                   ├─ 4. Context Relevance Grader
                                             ├─ 5. Grounded Answer Generator
                                             └─ 6. Hallucination Audit & Citations
```

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup

```bash
# Clone repository
git clone <repo-url>
cd ADV_RAG_AI

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Configure your preferred LLM provider in `.env`:
- `LLM_PROVIDER=openai` (set `OPENAI_API_KEY`)
- `LLM_PROVIDER=groq` (set `GROQ_API_KEY`)
- `LLM_PROVIDER=google` (set `GOOGLE_API_KEY`)
- `LLM_PROVIDER=ollama` (local Ollama instance)
- `LLM_PROVIDER=mock` (deterministic offline test mode, no API key needed)

---

## 🖥️ Running Locally

### Start FastAPI Backend
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs available at `http://localhost:8000/docs`.

### Start Streamlit Frontend
```bash
streamlit run app/streamlit_app.py --server.port 8501
```
Access the UI at `http://localhost:8501`.

---

## 🐳 Docker Deployment

Run both backend and frontend via Docker Compose:
```bash
docker-compose up --build
```

- **Backend API**: `http://localhost:8000`
- **Streamlit UI**: `http://localhost:8501`

---

## 🧪 Running Automated Tests

Run the full pytest suite:
```bash
pytest tests/ -v
```
