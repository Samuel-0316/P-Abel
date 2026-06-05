"""
Parent chunk store — disk-backed lookup for the parent-child RAG architecture.

Layout on disk:
    storage/parents/{session_id}/{doc_id}.json
        {
          "parents": [
            {"index": 0, "content": "...", "metadata": {...}},
            {"index": 1, "content": "...", "metadata": {...}},
            ...
          ]
        }

Child chunks (indexed in FAISS) carry `parent_index` in their metadata.
At retrieval time, call load_parent(session_id, doc_id, parent_index) to
get the full section text that child belongs to.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from config import PATHS


def _session_parents_dir(session_id: str) -> Path:
    return PATHS.parents_dir / session_id


def _doc_parents_path(session_id: str, doc_id: str) -> Path:
    return _session_parents_dir(session_id) / f"{doc_id}.json"


def save_parents(session_id: str, doc_id: str, parents: list[Document]) -> None:
    """
    Persist parent chunks for a document to disk.

    Called once at ingest time — never during a query.
    If a file already exists for this doc_id it is silently overwritten
    (handles re-index of the same document).
    """
    if not parents:
        return

    parent_dir = _session_parents_dir(session_id)
    parent_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for doc in parents:
        records.append(
            {
                "index": int(doc.metadata.get("parent_index", 0)),
                "content": doc.page_content,
                "metadata": {k: v for k, v in doc.metadata.items()},
            }
        )

    payload = {"doc_id": doc_id, "session_id": session_id, "parents": records}
    _doc_parents_path(session_id, doc_id).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_parent(session_id: str, doc_id: str, parent_index: int) -> Document | None:
    """
    Load a single parent chunk by its index.

    Returns None if the parent store doesn't exist for this doc (e.g. documents
    indexed before the parent-child architecture was introduced), allowing a
    graceful fallback to the child chunk itself.
    """
    path = _doc_parents_path(session_id, doc_id)
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    for record in payload.get("parents", []):
        if int(record.get("index", -1)) == parent_index:
            return Document(
                page_content=str(record.get("content", "")),
                metadata=record.get("metadata", {}),
            )

    return None


def load_all_parents(session_id: str, doc_id: str) -> list[Document]:
    """Load every parent chunk for a document (used by summary index builder)."""
    path = _doc_parents_path(session_id, doc_id)
    if not path.exists():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    docs: list[Document] = []
    for record in sorted(payload.get("parents", []), key=lambda r: int(r.get("index", 0))):
        docs.append(
            Document(
                page_content=str(record.get("content", "")),
                metadata=record.get("metadata", {}),
            )
        )
    return docs
