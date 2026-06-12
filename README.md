# P-Abel — Multi-Document Intelligence

A local-first, production-grade **Retrieval-Augmented Generation (RAG)** system for querying, summarising, and analysing PDF, DOCX, TXT, and Excel documents through a conversational chat interface.

Built with **Groq API** (Llama 3.3 70B) as the primary LLM and **Ollama** (local models) as an automatic fallback — the system gracefully degrades to raw chunk excerpts when no LLM is reachable.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
  - [High-Level Overview](#high-level-overview)
  - [Ingestion Pipeline](#ingestion-pipeline)
  - [Parent-Child Chunking](#parent-child-chunking)
  - [Retrieval Pipeline](#retrieval-pipeline)
  - [LLM Tier System](#llm-tier-system)
  - [Query-Adaptive System](#query-adaptive-system)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Design Decisions & Tradeoffs](#design-decisions--tradeoffs)

---

## Features

- **Multi-document sessions** — Upload multiple files per session; each session has its own isolated FAISS index
- **Parent-Child RAG** — Retrieve with precision (small chunks), generate with context (full sections)
- **Hybrid retrieval** — FAISS semantic search + BM25 keyword search + summary vectors, fused with Reciprocal Rank Fusion (RRF)
- **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` scores query-chunk pairs for precision after RRF
- **HyDE query expansion** — Hypothetical Document Embedding improves semantic search quality
- **Query-adaptive prompting** — Comprehensive queries ("full breakdown") get wider retrieval, structured markdown table output; lookup queries get concise precise answers
- **Conversational memory** — Per-thread message history with multi-turn context
- **Multi-thread chat** — Create, switch, and delete conversation threads per session
- **Automatic LLM fallback** — Groq → Ollama (local) → raw chunk excerpts
- **Faithfulness scoring** — Every answer is scored for groundedness against retrieved context
- **Document summary index** — Per-document summaries built at ingest time, cached, used as a third retrieval lane
- **Session management** — Export, import, rename, and delete sessions with full FAISS + thread persistence
- **FAISS health checks** — Detect and repair corrupt or incomplete vector indexes

---

## Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (app.py)                        │
│              Upload Page          │          Chat Page              │
└──────────────┬────────────────────┴──────────────┬──────────────────┘
               │  Ingest                           │  Query
               ▼                                   ▼
┌──────────────────────────┐       ┌───────────────────────────────────┐
│    INGESTION PIPELINE    │       │         RETRIEVAL PIPELINE        │
│                          │       │                                   │
│  load_documents()        │       │  1. HyDE query expansion          │
│       ↓                  │       │  2. FAISS child chunk search k=10 │
│  chunk_documents_        │       │  3. BM25 keyword search      k=10 │
│  hierarchical()          │       │  4. Summary vector search    k=3  │
│       ↓                  │       │  5. RRF fusion            → top-12│
│  Child chunks → FAISS    │       │  6. Cross-encoder reranking → top-5│
│  Parent sections → disk  │       │  7. Parent section expansion      │
│  Summary → LLM/cache     │       │  8. Deduplicate parents           │
└──────────────────────────┘       └─────────────────┬─────────────────┘
                                                     │
                                                     ▼
                                   ┌───────────────────────────────────┐
                                   │           LLM TIER SYSTEM         │
                                   │                                   │
                                   │  Tier 1: Groq — Llama 3.3 70B    │
                                   │       ↓ (rate limit / key error)  │
                                   │  Tier 2: Ollama (phi3.5 local)   │
                                   │       ↓ (model unavailable)       │
                                   │  Tier 3: Raw chunk excerpts       │
                                   └───────────────────────────────────┘
```

---

### Ingestion Pipeline

When a document is uploaded:

1. **Load** — `ingestion/loaders.py` dispatches to PyMuPDF (PDF), docx2txt (DOCX), plain text, or the custom Excel parser based on file extension
2. **Hierarchical chunk** — `ingestion/chunker.py` runs a two-level split:
   - **Parent chunks** (~1500 chars) — semantic sections saved to disk via `indexing/parent_store.py`
   - **Child chunks** (~200 chars) — small precise units indexed into FAISS
3. **Index children** — `indexing/vector_store.py` embeds child chunks using `sentence-transformers/all-MiniLM-L6-v2` and merges them into the session FAISS index
4. **Save parents** — `indexing/parent_store.py` writes parent sections to `storage/parents/{session_id}/{doc_id}.json` keyed by `parent_index`
5. **Build summary** — `indexing/summary_index.py` generates a document-level summary via Groq (LLM-quality digest) and caches it to `storage/summary_index/{session_id}/{doc_id}.json`. Subsequent uploads of the same document skip this step (cache hit).

Excel documents skip the parent-child split — each row group is already an atomic unit — and go through single-level chunking directly.

---

### Parent-Child Chunking

This is the core architectural decision that separates this system from naive RAG implementations.

**The problem with single-level chunking:**
When information about a topic (e.g. compensation structure) spans multiple pages, a 400-char chunk from page 3 embeds well for the query but contains only part of the answer. The system retrieves 5 such fragments — none complete.

**The solution — dual granularity:**

```
Document
  └── Parent section  (~1500 chars — one full semantic section)     ← sent to LLM
       ├── Child chunk A  (~200 chars)  ← indexed in FAISS, retrieved by embedding
       ├── Child chunk B  (~200 chars)  ← indexed in FAISS, retrieved by embedding
       └── Child chunk C  (~200 chars)  ← indexed in FAISS, retrieved by embedding
```

**At retrieval time:**
- FAISS searches find the most relevant **child chunks** (precise embedding match)
- The retriever then looks up each child's `parent_index` and loads the full **parent section** from disk
- Multiple children from the same parent are deduplicated — the section is included once
- The LLM receives **complete sections**, not fragmented snippets

This means retrieving any chunk in a "Compensation & Benefits" section automatically surfaces the entire section to the LLM — covering all salary components, allowances, PF, gratuity, etc. — even if they appear on different lines.

**Chunk sizes:**

| Level | Size | Overlap | Purpose |
|-------|------|---------|---------|
| Parent | 1500 chars (~375 tokens) | 100 chars | Full section context for LLM |
| Child | 200 chars (~50 tokens) | 20 chars | Precise embedding target for retrieval |

---

### Retrieval Pipeline

Every query goes through 8 stages:

```
User Question
      │
      ▼ [1] HyDE — Hypothetical Document Embedding
      │    LLM generates a plausible answer to the question.
      │    This synthetic answer embeds better than the raw question,
      │    improving semantic recall. Skipped when using Ollama
      │    (too slow on CPU without GPU).
      │
      ▼ [2] FAISS Semantic Search  (child chunks, k=10 or k=15 for comprehensive)
      │    All-MiniLM-L6-v2 embeddings, L2 distance
      │
      ▼ [3] BM25 Keyword Search    (child chunks, k=10 or k=15)
      │    Exact keyword matching via rank-bm25
      │
      ▼ [4] Summary Vector Search  (cached doc summaries, k=3)
      │    Searches document-level summaries — helps surface the right
      │    document when the query is high-level ("what is this about?")
      │
      ▼ [5] RRF Fusion             (top-12 or top-18 candidates)
      │    Reciprocal Rank Fusion combines the three rankings without
      │    needing score normalisation across different retrieval methods
      │
      ▼ [6] Cross-Encoder Reranking  (top-5 or top-8)
      │    cross-encoder/ms-marco-MiniLM-L-6-v2 (~22 MB) scores each
      │    query-chunk pair directly — much more accurate than cosine
      │    similarity alone. Runs on CPU in ~100-200ms.
      │
      ▼ [7] Parent Section Expansion
      │    For each top child chunk: load its parent section from disk.
      │    Falls back to the child chunk for Excel or legacy documents.
      │
      ▼ [8] Deduplication
           Multiple children sharing the same parent → parent included once.
           Final: 3–8 complete document sections sent to the LLM.
```

---

### LLM Tier System

The system automatically falls through tiers without user intervention:

| Tier | Provider | Condition | HyDE |
|------|----------|-----------|------|
| 1 | Groq — Llama 3.3 70B Versatile | API key valid + within rate limits | ✅ Enabled |
| 2 | Ollama (phi3.5 local) | Groq unavailable | ❌ Skipped (CPU speed) |
| 3 | Raw chunk excerpts | Both LLMs unreachable | — |

**Error classification** — raw API errors are translated to human-readable messages:
- `429` → *"Groq free-tier rate limit hit. Wait a moment and retry."*
- `401` → *"Groq API key is invalid. Check GROQ_API_KEY in .env."*
- `404 model` → *"Ollama model not found. Run `ollama list`."*
- `OOM` → *"Ollama model requires more RAM than available."*

**Faithfulness scoring** — after generation, every answer is scored using lexical token overlap between the answer and retrieved context. No extra LLM call — the score and reason appear in the "Sources and confidence" expander.

---

### Query-Adaptive System

The system classifies every query into one of two types and adapts retrieval depth, context budget, and output format accordingly.

**Comprehensive queries** — triggered by keywords: `full`, `complete`, `all`, `breakdown`, `summarize`, `summary`, `overview`, `details`, `explain`, `describe`, `package`, `total`, `structure`, `comprehensive`

```
Retrieval: child_semantic_k=15, child_bm25_k=15, rrf_top_n=18, rerank_top_n=8
Context:   MAX_CONTEXT_CHARS × 2  (up to 12,000 chars)
Prompt:    Exhaustive mode — instructs LLM to:
           • Use ## headers and markdown tables for financial data
           • Cover EVERY component mentioned in context
           • Note (not refuse) when values are referenced elsewhere
           • Always end with a summary table
```

**Lookup queries** — everything else (single fact, date, name, status):

```
Retrieval: child_semantic_k=10, child_bm25_k=10, rrf_top_n=12, rerank_top_n=5
Context:   MAX_CONTEXT_CHARS  (6,000 chars)
Prompt:    Precise mode — concise answer citing the source section
```

---

## Project Structure

```
multi_doc_intelligence/
│
├── app.py                          # Streamlit entry point, session management
├── config.py                       # All constants, paths, env vars
├── requirements.txt
│
├── chains/
│   ├── llm_builder.py              # ChatGroq + ChatOllama factory with fallback
│   ├── qa_chain.py                 # Main QA chain: retrieval → prompt → LLM
│   │                               #   _classify_query_type(), _build_prompt()
│   ├── hallucination.py            # Lexical faithfulness scoring (no LLM call)
│   ├── summarize_chain.py          # Map-reduce document summarisation
│   └── insight_chain.py            # Insight extraction chain
│
├── indexing/
│   ├── vector_store.py             # FAISS index manager (ingest, search, health)
│   ├── parent_store.py             # Parent section disk store
│   │                               #   save_parents(), load_parent(), load_all_parents()
│   ├── summary_index.py            # Per-document summary builder + cache
│   └── llama_index_builder.py      # LlamaIndex FAISS wrapper (legacy)
│
├── ingestion/
│   ├── loaders.py                  # File type dispatcher (PDF/DOCX/TXT/XLSX)
│   ├── chunker.py                  # Hierarchical chunker
│   │                               #   chunk_documents_hierarchical() → (children, parents)
│   │                               #   chunk_documents()              → flat list (Excel)
│   └── excel_parser.py             # Row-level Excel → Document converter
│
├── retrieval/
│   ├── hybrid_retriever.py         # 8-step retrieval pipeline
│   ├── reranker.py                 # RRF fusion + cross-encoder reranking
│   ├── hyde.py                     # HyDE query expansion (skippable)
│   └── multi_vector.py             # Summary vector search (cache-only at query time)
│
├── ui/
│   ├── chat_page.py                # Chat UI, thread management
│   ├── upload_page.py              # Upload UI, hierarchical ingest flow
│   └── analysis_page.py            # Document analysis UI
│
├── scripts/
│   ├── faiss_local_smoke.py        # FAISS smoke test
│   └── faiss_session_health.py     # Session index health checker
│
└── storage/                        # Runtime data (gitignored)
    ├── faiss_index/{session_id}/   # FAISS child chunk indexes
    ├── parents/{session_id}/       # Parent section JSON store
    ├── summary_index/{session_id}/ # Document summary cache
    ├── threads/{session_id}/       # Conversation thread persistence
    └── uploads/                    # Uploaded source files
```

---

## Setup

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com) (14,400 requests/day free)
- [Ollama](https://ollama.com) installed and running locally (optional, for offline fallback)

### Install

```bash
# Clone the repo
git clone https://github.com/Samuel-0316/P-Abel.git
cd P-Abel/multi_doc_intelligence

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Pull a local Ollama model (optional — offline fallback only)

```bash
# phi3.5 is the recommended fallback (2.2 GB, fits in 4 GB RAM)
ollama pull phi3.5

# List what you have
ollama list
```

### Configure environment

```bash
copy .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
DEFAULT_OLLAMA_MODEL=phi3.5:latest
```

Get a free Groq API key at [console.groq.com](https://console.groq.com) — no credit card required, 14,400 requests/day on the free tier.

### Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## Configuration

All tunable parameters live in `config.py` and can be overridden via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Groq API key (get one at console.groq.com) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model identifier |
| `DEFAULT_OLLAMA_MODEL` | `phi3.5:latest` | Local fallback model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `EMBED_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `PARENT_CHUNK_SIZE` | `1500` | Parent section size in chars |
| `CHILD_CHUNK_SIZE` | `200` | Child chunk size in chars |
| `MAX_CONTEXT_CHARS` | `6000` | LLM context budget (doubled for comprehensive queries) |
| `USE_HYDE` | `true` | Enable HyDE query expansion (auto-disabled for Ollama) |

**Available Groq models** (all free tier):

| Model | Speed | Best for |
|-------|-------|----------|
| `llama-3.3-70b-versatile` | Fast | Best quality — recommended default |
| `llama-3.1-8b-instant` | Very fast | High volume / quick lookups |
| `mixtral-8x7b-32768` | Fast | Long context (32K) |

---

## Usage

### 1 — Upload Documents

1. Select **Upload** from the sidebar
2. Drag in one or more PDF, DOCX, TXT, or XLSX files
3. Click **Index uploaded files**
4. The status shows: `"128 child chunks · 18 parent sections stored"`

Each document is deduplicated by SHA-256 hash — uploading the same file twice is a no-op.

### 2 — Chat

1. Select **Chat** from the sidebar
2. Type a question in the chat input
3. The system retrieves relevant sections and generates a grounded answer
4. Expand **Sources and confidence** to see which pages and chunks were cited

**Query tips:**
- For a full overview: *"Give me the full compensation breakdown"* → structured table output
- For a specific fact: *"What is the joining date?"* → single concise answer
- For document comparison: Upload multiple files in the same session and ask cross-document questions

### 3 — Thread Management

- **New thread** — starts a fresh conversation (no history carried over)
- **Switch thread** — select from the dropdown to resume a previous conversation
- **Delete thread** — removes the thread and clears its memory

### 4 — Session Management

- **New session** — creates a fresh isolated FAISS index
- **Open session** — switch between previously indexed document sets
- **Rename** — give the session a human-readable name
- **Export** — download a ZIP containing the FAISS index, parent store, summaries, threads, and uploaded files

---

## Design Decisions & Tradeoffs

### Why Groq over other cloud LLM APIs?

Groq runs on custom LPU (Language Processing Unit) hardware, delivering ~500 tokens/second — 10–20× faster than typical GPU-based APIs. The free tier offers 14,400 requests/day (6,000 RPM) with no credit card required. Llama 3.3 70B on Groq matches GPT-4 quality for document QA tasks at zero cost.

### Why parent-child chunking over single-level?

Single-level chunking at any fixed size creates a dilemma: small chunks embed precisely but lack context; large chunks provide context but embed noisily. Parent-child solves this by using the right granularity for the right job — small for retrieval, large for generation.

The key insight: when information spans multiple pages (salary components, contract terms, multi-step processes), retrieving *any* child chunk in a section automatically surfaces the *entire section* to the LLM.

### Why cross-encoder reranking over score thresholds?

FAISS returns L2 distances where lower = more similar. A score threshold is tricky to calibrate and brittle across different documents and queries. A cross-encoder scores query-chunk relevance directly, making the decision much more accurate without requiring threshold tuning. At ~150ms on CPU, it's invisible behind LLM response time.

### Why is HyDE disabled for Ollama?

HyDE requires one LLM inference call before retrieval. For Groq this is ~0.5 seconds and free. For phi3.5 on a CPU-only laptop it's 10–30 seconds of extra wait before the actual answer generation begins. The raw question is a good enough retrieval signal for local model scenarios.

### Why lexical faithfulness scoring instead of LLM-based?

Making a second LLM call per query just to score faithfulness doubles API usage. Lexical token overlap (words > 4 chars in common between answer and context) is a reasonable faithfulness proxy and runs in <1ms. The UI still shows the confidence score and a human-readable reason.

### LLM call budget

| Scenario | Calls per query |
|----------|----------------|
| Groq available | 1 (answer generation) |
| Groq rate-limited, Ollama available | 1 (Ollama answer) |
| Both unavailable | 0 (raw chunks shown) |
| Document upload (first time) | 1 (summary generation, cached) |

---

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) — LCEL chains, FAISS integration, BM25
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) — embeddings + cross-encoder
- [Groq](https://console.groq.com) — primary LLM (Llama 3.3 70B, free tier)
- [Ollama](https://ollama.com) — local model inference fallback
- [Streamlit](https://streamlit.io) — UI framework
- [FAISS](https://github.com/facebookresearch/faiss) — vector similarity search
- [Meta Llama](https://ai.meta.com/llama/) — underlying model weights (via Groq)
