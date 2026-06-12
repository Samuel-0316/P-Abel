"""
FastAPI REST API for Multi-Document Intelligence.

Wraps all existing Python backend functions (chains, indexing, ingestion,
retrieval) as REST endpoints. The Streamlit UI code is NOT touched — this
is a parallel entry point.

Run:
    uvicorn api.server:app --reload --port 8000
"""
from __future__ import annotations

import io
import json
import logging
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import DOC_REGISTRY_PATH, PATHS, ensure_storage_dirs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

class SessionCreate(BaseModel):
    name: str | None = None

class SessionRename(BaseModel):
    name: str

class ThreadRename(BaseModel):
    name: str

class ChatRequest(BaseModel):
    session_id: str
    thread_id: str
    question: str

class ChatResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    hyde_query: str
    faithfulness: dict[str, Any]
    elapsed_ms: float
    from_cache: bool
    error: str | None = None

class UploadResult(BaseModel):
    filename: str
    status: str
    doc_id: str = ""
    chunk_count: int = 0
    parent_count: int = 0
    message: str = ""

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Multi-Document Intelligence API",
    version="1.0.0",
    docs_url="/api/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    ensure_storage_dirs()


# ---------------------------------------------------------------------------
# Registry helpers (same logic as app.py, no Streamlit dependency)
# ---------------------------------------------------------------------------

def _load_registry() -> dict:
    if not DOC_REGISTRY_PATH.exists():
        return {"sessions": {}}
    try:
        return json.loads(DOC_REGISTRY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"sessions": {}}


def _write_registry(payload: dict) -> None:
    DOC_REGISTRY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _normalize_id(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    return "" if s.lower() == "none" else s


# ---------------------------------------------------------------------------
# Thread persistence helpers
# ---------------------------------------------------------------------------

def _threads_dir(session_id: str) -> Path:
    return PATHS.threads_dir / session_id


def _threads_file(session_id: str) -> Path:
    return _threads_dir(session_id) / "threads.json"


def _load_threads(session_id: str) -> dict:
    path = _threads_file(session_id)
    if not path.exists():
        return {
            "active_thread": "thread-1",
            "thread_names": {"thread-1": "Thread 1"},
            "threads": {"thread-1": []},
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "active_thread": "thread-1",
            "thread_names": {"thread-1": "Thread 1"},
            "threads": {"thread-1": []},
        }


def _save_threads(session_id: str, data: dict) -> None:
    d = _threads_dir(session_id)
    d.mkdir(parents=True, exist_ok=True)
    _threads_file(session_id).write_text(json.dumps(data, indent=2), encoding="utf-8")


# ===================================================================
# SESSION ENDPOINTS
# ===================================================================

@app.get("/api/sessions")
def list_sessions():
    """List all sessions (including empty projects with a name)."""
    payload = _load_registry()
    sessions = payload.get("sessions", {})
    result = []
    for sid, meta in sessions.items():
        if not isinstance(meta, dict):
            continue
        docs = meta.get("documents", [])
        if not isinstance(docs, list):
            docs = []
        name = str(meta.get("session_name", "")).strip()
        if not name:
            if not docs:
                continue  # Skip unnamed sessions with no docs
            first_doc = next((d for d in docs if isinstance(d, dict)), {})
            name = str(first_doc.get("filename", sid))
            for ext in (".pdf", ".docx", ".txt", ".xlsx"):
                if name.endswith(ext):
                    name = name[: -len(ext)]
                    break
        result.append({
            "session_id": sid,
            "session_name": name,
            "document_count": len(docs),
        })
    result.sort(key=lambda x: x["session_name"].lower())
    return result


@app.post("/api/sessions")
def create_session(body: SessionCreate):
    """Create a new empty session."""
    sid = uuid4().hex[:12]
    if body.name:
        payload = _load_registry()
        sessions = payload.setdefault("sessions", {})
        sessions[sid] = {"session_name": body.name.strip(), "documents": []}
        _write_registry(payload)
    return {"session_id": sid, "session_name": body.name or sid}


@app.put("/api/sessions/{session_id}/rename")
def rename_session(session_id: str, body: SessionRename):
    clean = body.name.strip()
    if not clean:
        raise HTTPException(400, "Name cannot be empty")
    payload = _load_registry()
    sessions = payload.setdefault("sessions", {})
    bucket = sessions.setdefault(session_id, {"session_name": clean, "documents": []})
    bucket["session_name"] = clean
    _write_registry(payload)
    return {"ok": True}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    sid = _normalize_id(session_id)
    if not sid:
        raise HTTPException(400, "Invalid session id")

    payload = _load_registry()
    session_meta = payload.get("sessions", {}).pop(sid, {})
    _write_registry(payload)

    # 1. Identify which files were uploaded to this session
    filenames_to_check = set()
    if isinstance(session_meta, dict):
        for doc in session_meta.get("documents", []):
            if isinstance(doc, dict):
                fn = str(doc.get("filename", "")).strip()
                if fn:
                    filenames_to_check.add(fn)

    # 2. Check if any OTHER session is still using those files
    still_in_use = set()
    for other_meta in payload.get("sessions", {}).values():
        if isinstance(other_meta, dict):
            for doc in other_meta.get("documents", []):
                if isinstance(doc, dict):
                    fn = str(doc.get("filename", "")).strip()
                    if fn:
                        still_in_use.add(fn)

    # 3. Delete files that are completely orphaned
    for fn in filenames_to_check:
        if fn not in still_in_use:
            file_path = PATHS.upload_dir / fn
            if file_path.exists():
                file_path.unlink()

    # 4. Clean up session-specific directories
    for folder in (
        PATHS.faiss_index_dir / sid,
        PATHS.summary_index_dir / sid,
        PATHS.threads_dir / sid,
        PATHS.storage_dir / "query_cache" / sid,
        PATHS.parents_dir / sid,
    ):
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)

    return {"ok": True, "message": "Session and orphaned files deleted"}


@app.get("/api/sessions/{session_id}/export")
def export_session(session_id: str):
    """Build and stream a ZIP export of the session."""
    sid = _normalize_id(session_id)
    if not sid:
        raise HTTPException(400, "Invalid session id")

    payload = _load_registry()
    session_meta = payload.get("sessions", {}).get(sid, {})

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        manifest = {
            "session_id": sid,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "session": session_meta,
        }
        zf.writestr("session/manifest.json", json.dumps(manifest, indent=2))

        if DOC_REGISTRY_PATH.exists():
            zf.write(DOC_REGISTRY_PATH, "session/doc_registry.json")

        for folder, prefix in [
            (PATHS.faiss_index_dir / sid, "session/faiss_index"),
            (PATHS.summary_index_dir / sid, "session/summary_index"),
            (PATHS.threads_dir / sid, "session/threads"),
            (PATHS.storage_dir / "query_cache" / sid, "session/query_cache"),
        ]:
            if folder.exists():
                for f in folder.rglob("*"):
                    if f.is_file():
                        zf.write(f, f"{prefix}/{f.relative_to(folder).as_posix()}")

        # Include uploaded source files
        referenced = set()
        if isinstance(session_meta, dict):
            for doc in session_meta.get("documents", []):
                if isinstance(doc, dict):
                    fn = str(doc.get("filename", "")).strip()
                    if fn:
                        referenced.add(fn)
        for fn in sorted(referenced):
            p = PATHS.upload_dir / fn
            if p.exists():
                zf.write(p, f"session/uploads/{fn}")

    buf.seek(0)
    export_name = f"session_{sid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={export_name}"},
    )


# ===================================================================
# THREAD ENDPOINTS
# ===================================================================

@app.get("/api/sessions/{session_id}/threads")
def list_threads(session_id: str):
    data = _load_threads(session_id)
    threads = data.get("threads", {})
    names = data.get("thread_names", {})
    active = data.get("active_thread", "")

    result = []
    for tid in threads:
        result.append({
            "thread_id": tid,
            "thread_name": names.get(tid, f"Thread {tid.split('-')[-1]}"),
            "message_count": len(threads.get(tid, [])),
        })
    return {"threads": result, "active_thread": active}


@app.post("/api/sessions/{session_id}/threads")
def create_thread(session_id: str):
    data = _load_threads(session_id)
    threads = data.setdefault("threads", {})
    names = data.setdefault("thread_names", {})

    # Find next unique ID
    n = len(threads) + 1
    new_id = f"thread-{n}"
    while new_id in threads:
        n += 1
        new_id = f"thread-{n}"

    threads[new_id] = []
    names[new_id] = f"Thread {n}"
    data["active_thread"] = new_id
    _save_threads(session_id, data)
    return {"thread_id": new_id, "thread_name": f"Thread {n}"}


@app.put("/api/sessions/{session_id}/threads/{thread_id}/rename")
def rename_thread(session_id: str, thread_id: str, body: ThreadRename):
    clean = body.name.strip()
    if not clean:
        raise HTTPException(400, "Name cannot be empty")
    data = _load_threads(session_id)
    names = data.setdefault("thread_names", {})
    names[thread_id] = clean
    _save_threads(session_id, data)
    return {"ok": True}


@app.delete("/api/sessions/{session_id}/threads/{thread_id}")
def delete_thread(session_id: str, thread_id: str):
    from chains.qa_chain import clear_thread_memory

    data = _load_threads(session_id)
    threads = data.get("threads", {})
    names = data.get("thread_names", {})

    if len(threads) <= 1:
        # Reset the only thread instead of deleting
        threads[thread_id] = []
        clear_thread_memory(thread_id)
    else:
        threads.pop(thread_id, None)
        names.pop(thread_id, None)
        clear_thread_memory(thread_id)
        data["active_thread"] = sorted(threads.keys())[0]

    _save_threads(session_id, data)
    return {"ok": True}


# ===================================================================
# CHAT ENDPOINT
# ===================================================================

@app.post("/api/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    from chains.qa_chain import ask_question, hydrate_thread_memory

    # Ensure thread memory is hydrated from disk
    data = _load_threads(body.session_id)
    threads = data.get("threads", {})
    messages = threads.get(body.thread_id, [])
    hydrate_thread_memory(body.thread_id, messages)

    t_start = time.perf_counter()
    result = ask_question(
        question=body.question,
        thread_id=body.thread_id,
        session_id=body.session_id,
        model_name=None,
    )
    elapsed_ms = (time.perf_counter() - t_start) * 1000

    # Persist the turn to thread storage
    threads.setdefault(body.thread_id, [])

    # Check if this is the first user message — auto-rename thread
    was_empty = len(threads[body.thread_id]) == 0

    threads[body.thread_id].append({"role": "user", "content": body.question})
    threads[body.thread_id].append({
        "role": "assistant",
        "content": result.answer,
        "meta": {
            "citations": result.citations,
            "hyde_query": result.hyde_query,
            "faithfulness": {
                "faithful": result.faithfulness.faithful,
                "confidence": result.faithfulness.confidence,
                "reason": result.faithfulness.reason,
            },
            "error": result.error,
            "from_cache": result.from_cache,
            "elapsed_ms": round(elapsed_ms, 1),
        },
    })

    # Auto-rename thread from first question (like ChatGPT)
    if was_empty:
        auto_name = body.question.strip()
        # Truncate to a reasonable title length
        if len(auto_name) > 40:
            auto_name = auto_name[:37] + "..."
        names = data.setdefault("thread_names", {})
        names[body.thread_id] = auto_name

    _save_threads(body.session_id, data)

    return ChatResponse(
        answer=result.answer,
        citations=result.citations,
        hyde_query=result.hyde_query,
        faithfulness={
            "faithful": result.faithfulness.faithful,
            "confidence": result.faithfulness.confidence,
            "reason": result.faithfulness.reason,
        },
        elapsed_ms=round(elapsed_ms, 1),
        from_cache=result.from_cache,
        error=result.error,
    )


# ===================================================================
# UPLOAD ENDPOINT
# ===================================================================

@app.post("/api/upload")
async def upload_files(session_id: str, files: list[UploadFile] = File(...)):
    from indexing.parent_store import save_parents
    from indexing.query_cache import clear_session_cache
    from indexing.summary_index import build_summary_index
    from indexing.vector_store import VectorStoreManager
    from ingestion.chunker import chunk_documents, chunk_documents_hierarchical
    from ingestion.loaders import load_documents

    results: list[dict[str, Any]] = []
    manager = VectorStoreManager(session_id=session_id)

    for uploaded in files:
        target = PATHS.upload_dir / uploaded.filename
        content = await uploaded.read()
        target.write_bytes(content)

        try:
            docs = load_documents(target)
            doc_type = docs[0].metadata.get("doc_type", "unknown") if docs else "unknown"
            is_excel = doc_type == "excel"

            if is_excel:
                children = chunk_documents(docs)
                parents = []
            else:
                children, parents = chunk_documents_hierarchical(docs)

            ingest_result = manager.ingest_document(file_path=target, chunks=children)

            if ingest_result.get("status") == "skipped":
                results.append({
                    "filename": uploaded.filename,
                    "status": "skipped",
                    "doc_id": str(ingest_result.get("doc_id", "")),
                    "chunk_count": 0,
                    "parent_count": 0,
                    "message": "Duplicate content already indexed",
                })
                continue

            doc_id = str(ingest_result.get("doc_id", ""))
            chunk_count = int(ingest_result.get("chunk_count", 0))

            if parents and doc_id:
                for p in parents:
                    p.metadata["doc_id"] = doc_id
                    p.metadata["session_id"] = session_id
                try:
                    save_parents(session_id=session_id, doc_id=doc_id, parents=parents)
                except Exception as exc:
                    logger.warning("Parent store save failed: %s", exc)

            if doc_id:
                try:
                    build_summary_index(session_id=session_id, doc_id=doc_id, model_name=None)
                except Exception:
                    pass

            clear_session_cache(session_id)

            results.append({
                "filename": uploaded.filename,
                "status": "indexed",
                "doc_id": doc_id,
                "chunk_count": chunk_count,
                "parent_count": len(parents),
                "message": f"{chunk_count} chunks indexed, {len(parents)} parent sections",
            })

        except Exception as exc:
            logger.exception("Upload failed for %s", uploaded.filename)
            results.append({
                "filename": uploaded.filename,
                "status": "error",
                "message": str(exc)[:300],
            })

    return {"results": results}


# ===================================================================
# THREAD MESSAGES (for loading chat history)
# ===================================================================

@app.get("/api/sessions/{session_id}/threads/{thread_id}/messages")
def get_thread_messages(session_id: str, thread_id: str):
    data = _load_threads(session_id)
    threads = data.get("threads", {})
    messages = threads.get(thread_id, [])
    return {"messages": messages}
