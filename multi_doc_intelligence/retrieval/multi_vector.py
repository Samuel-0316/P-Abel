from __future__ import annotations

from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from indexing.summary_index import load_summary_documents
from indexing.vector_store import VectorStoreManager


@dataclass
class MultiVectorConfig:
    enable_summary_vectors: bool = True
    enable_table_vectors: bool = True
    enable_raw_vectors: bool = True
    summary_top_k: int = 3  # Reduced from 5 — keeps context tight


def build_multi_vector_config() -> MultiVectorConfig:
    """Build the default multi-vector retrieval configuration."""
    return MultiVectorConfig()


def _summary_store(
    session_id: str,
    summary_documents: list[Document],
    vector_manager: VectorStoreManager,
) -> FAISS | None:
    if not summary_documents:
        return None
    return FAISS.from_documents(summary_documents, vector_manager.embeddings)


def collect_summary_vector_candidates(
    *,
    session_id: str,
    question: str,
    top_k: int = 3,
) -> list[Document]:
    """
    Search cached summary vectors for the given question.

    IMPORTANT: This function never calls the LLM or builds new summaries.
    Summaries are built once at document ingest time (see upload_page.py)
    and cached to disk. This keeps retrieval LLM-free.

    If no cached summaries exist yet, returns an empty list gracefully.
    """
    vector_manager = VectorStoreManager(session_id=session_id)

    # Load from disk cache only — no LLM call
    summary_documents = load_summary_documents(session_id=session_id)
    if not summary_documents:
        return []

    store = _summary_store(session_id, summary_documents, vector_manager)
    if store is None:
        return []

    return store.similarity_search(question, k=top_k)
