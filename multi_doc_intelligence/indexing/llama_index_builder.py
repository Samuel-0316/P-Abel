from __future__ import annotations

from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.faiss import FaissVectorStore

from config import PATHS


def _session_index_dir(session_id: str) -> str:
    return str(PATHS.faiss_index_dir / session_id)


def build_llamaindex_index(session_id: str) -> VectorStoreIndex:
    """Load persisted FAISS and wrap it in a LlamaIndex VectorStoreIndex."""
    vector_store = FaissVectorStore.from_persist_dir(_session_index_dir(session_id))
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


def build_llamaindex_retriever(session_id: str) -> Any:
    """Expose a LlamaIndex retriever for a session-scoped persisted FAISS index."""
    index = build_llamaindex_index(session_id=session_id)
    return index.as_retriever(similarity_top_k=8)


def build_llamaindex_query_engine(session_id: str) -> Any:
    """Expose a LlamaIndex query engine for response synthesis over retrieved nodes."""
    index = build_llamaindex_index(session_id=session_id)
    return index.as_query_engine(similarity_top_k=8)
