from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_core.documents import Document

from chains.llm_builder import build_llm
from config import PATHS
from indexing.vector_store import VectorStoreManager


@dataclass
class SummaryIndexRecord:
    doc_id: str
    session_id: str
    source: str
    version: int
    summary: str
    created_at: str


def _summary_cache_dir(session_id: str) -> Path:
    return PATHS.summary_index_dir / session_id


def _summary_cache_path(session_id: str, doc_id: str) -> Path:
    return _summary_cache_dir(session_id) / f"{doc_id}.json"


def _load_cached_summary(session_id: str, doc_id: str) -> list[Document]:
    cache_path = _summary_cache_path(session_id, doc_id)
    if not cache_path.exists():
        return []

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    documents: list[Document] = []
    for item in payload.get("documents", []):
        if not isinstance(item, dict):
            continue
        documents.append(
            Document(
                page_content=str(item.get("summary", "")),
                metadata={
                    "doc_id": str(item.get("doc_id", doc_id)),
                    "session_id": str(item.get("session_id", session_id)),
                    "source": str(item.get("source", "unknown")),
                    "version": int(item.get("version", 1)),
                    "doc_type": "summary",
                    "summary_type": "document",
                    "chunk_id": f"{item.get('doc_id', doc_id)}_summary",
                },
            )
        )
    return documents


def _persist_summary(session_id: str, doc_id: str, records: list[SummaryIndexRecord]) -> None:
    cache_dir = _summary_cache_dir(session_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _summary_cache_path(session_id, doc_id)
    payload = {"documents": [asdict(record) for record in records]}
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _summarize_text(text: str, *, model_name: str | None = None) -> str:
    if not text.strip():
        return "No text available for summarization."

    prompt = (
        "Summarize the following document version into a concise, retrieval-friendly digest. "
        "Use short bullets for facts, decisions, entities, and key risks. "
        "Do not add disclaimers or extra headings.\n\n"
        f"Document:\n{text}"
    )
    try:
        llm = build_llm(model_name=model_name, provider="groq", temperature=0)
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception:
        sentences = [segment.strip() for segment in text.replace("\n", " ").split(". ") if segment.strip()]
        if not sentences:
            return text[:1500]
        return ". ".join(sentences[:8])[:2000]


def load_summary_documents(session_id: str, doc_id: str | None = None) -> list[Document]:
    """Load cached summary documents for a session or a single doc version."""
    if doc_id:
        return _load_cached_summary(session_id, doc_id)

    cache_dir = _summary_cache_dir(session_id)
    if not cache_dir.exists():
        return []

    documents: list[Document] = []
    for cache_file in sorted(cache_dir.glob("*.json")):
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for item in payload.get("documents", []):
            if not isinstance(item, dict):
                continue
            documents.append(
                Document(
                    page_content=str(item.get("summary", "")),
                    metadata={
                        "doc_id": str(item.get("doc_id", cache_file.stem)),
                        "session_id": str(item.get("session_id", session_id)),
                        "source": str(item.get("source", "unknown")),
                        "version": int(item.get("version", 1)),
                        "doc_type": "summary",
                        "summary_type": "document",
                        "chunk_id": f"{item.get('doc_id', cache_file.stem)}_summary",
                    },
                )
            )
    return documents


def build_summary_index(session_id: str, doc_id: str, model_name: str | None = None) -> list[Document]:
    """Build and cache a summary document for a session document version."""
    manager = VectorStoreManager(session_id=session_id)
    documents = manager.documents_for_doc_id(doc_id)
    if not documents:
        return []

    cached_documents = _load_cached_summary(session_id, doc_id)
    if cached_documents:
        return cached_documents

    source_name = str(documents[0].metadata.get("source", doc_id))
    version = int(documents[0].metadata.get("version", 1))
    combined_text = "\n\n".join(doc.page_content for doc in documents)
    summary_text = _summarize_text(combined_text, model_name=model_name)

    record = SummaryIndexRecord(
        doc_id=doc_id,
        session_id=session_id,
        source=source_name,
        version=version,
        summary=summary_text,
        created_at=str(documents[0].metadata.get("created_at", "")),
    )
    _persist_summary(session_id, doc_id, [record])

    return [
        Document(
            page_content=summary_text,
            metadata={
                "doc_id": doc_id,
                "session_id": session_id,
                "source": source_name,
                "version": version,
                "doc_type": "summary",
                "summary_type": "document",
                "chunk_id": f"{doc_id}_summary",
            },
        )
    ]
