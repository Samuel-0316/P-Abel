# Multi-Document Intelligence Dashboard

Document intelligence app built with Streamlit, LangChain LCEL, LlamaIndex hooks, FAISS, and Google Gemini.

## Current Status

Implemented:

- Project structure and configuration
- Loader support for PDF, DOCX, TXT, and structured Excel
- Chunking router by document type
- Session-scoped FAISS persistence
- Document versioning via content hash
- Upload page wired to indexing
- Hybrid retrieval: HyDE + semantic FAISS + BM25 + RRF fusion
- Summary-vector support with cached summary index per document version
- LlamaIndex query engine integration in QA flow
- Chains for summary, insights extraction, QA, and faithfulness scoring
- Gemini-first LLM runtime with graceful fallback behavior
- Chat thread persistence to disk with thread create/delete/reset

Partially implemented / pending extension:

- Table-vector retrieval path (config exists; retrieval path not yet wired)
- Optional production hardening (monitoring, test suite expansion, deployment packaging)

## Run Locally

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and adjust values as needed.
4. Add your Gemini key to `.env` as `GOOGLE_API_KEY=...`.
5. Start Streamlit:

```bash
streamlit run app.py
```

## Project Structure

- `app.py`: Streamlit entry and page routing
- `config.py`: settings, paths, and model defaults
- `ingestion/`: loaders, Excel parser, chunking
- `indexing/`: FAISS manager, summary index cache, LlamaIndex builder
- `retrieval/`: HyDE, RRF, hybrid retriever, summary-vector retrieval
- `chains/`: summarize/insight/qa/faithfulness and LLM builder
- `ui/`: upload, analysis, chat pages
- `storage/`: runtime persisted artifacts (gitignored)
