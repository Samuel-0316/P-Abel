"""
Hierarchical document chunking — the foundation of the parent-child RAG architecture.

Two-level split strategy
------------------------
  Parent chunks  (~1500 chars)  — full semantic sections, sent to the LLM
  Child  chunks  (~200 chars)   — precise units, indexed in FAISS for retrieval

At query time the retriever matches child chunks (precision), then expands
each hit to its parent (context completeness) before sending to the LLM.
This is the same pattern used by LangChain's ParentDocumentRetriever and
LlamaIndex's Small-to-Big Retrieval.

Excel documents use a single-level strategy (rows/cells are already atomic).
"""
from __future__ import annotations

from typing import Iterable

from langchain_core.documents import Document

from config import (
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    EXCEL_CHUNK_OVERLAP,
    EXCEL_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
)

# Sentence-aware separators — split on structural boundaries first,
# falling back to finer boundaries only when needed.
SEMANTIC_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]
EXCEL_SEPARATORS = ["\n", ", ", " "]


def _make_splitter(chunk_size: int, chunk_overlap: int, separators: list[str]):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        keep_separator=True,
    )


def chunk_documents(documents: Iterable[Document]) -> list[Document]:
    """
    Single-level chunking (legacy path, used for Excel).

    For PDF/DOCX/TXT prefer chunk_documents_hierarchical().
    """
    chunks: list[Document] = []
    for doc in documents:
        doc_type = doc.metadata.get("doc_type")
        if doc_type == "excel":
            splitter = _make_splitter(EXCEL_CHUNK_SIZE, EXCEL_CHUNK_OVERLAP, EXCEL_SEPARATORS)
        else:
            splitter = _make_splitter(CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP, SEMANTIC_SEPARATORS)

        current_chunks = splitter.split_documents([doc])
        for i, chunk in enumerate(current_chunks):
            chunk.metadata.setdefault("source", doc.metadata.get("source", "unknown"))
            chunk.metadata["doc_type"] = doc_type or "unknown"
            chunk.metadata["chunk_order"] = i
        chunks.extend(current_chunks)

    return chunks


def chunk_documents_hierarchical(
    documents: Iterable[Document],
) -> tuple[list[Document], list[Document]]:
    """
    Two-level hierarchical chunking.

    Returns
    -------
    children : list[Document]
        Small chunks (CHILD_CHUNK_SIZE chars) to be indexed in FAISS.
        Each child carries ``parent_index`` in its metadata so the retriever
        can look up the parent after embedding search.

    parents : list[Document]
        Large chunks (PARENT_CHUNK_SIZE chars) to be saved in the parent store.
        Each parent carries ``parent_index`` as its position key.

    Note: doc_id/session_id are NOT set here — they are applied by
    VectorStoreManager.ingest_document() after this function returns.
    The upload page saves parents using the doc_id from that return value.
    """
    parent_splitter = _make_splitter(
        PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP, SEMANTIC_SEPARATORS
    )
    child_splitter = _make_splitter(
        CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP, SEMANTIC_SEPARATORS
    )

    all_children: list[Document] = []
    all_parents: list[Document] = []
    global_parent_idx = 0

    for doc in documents:
        doc_type = doc.metadata.get("doc_type")
        source = doc.metadata.get("source", "unknown")

        if doc_type == "excel":
            # Excel: use single-level chunking — rows are already atomic units
            splitter = _make_splitter(EXCEL_CHUNK_SIZE, EXCEL_CHUNK_OVERLAP, EXCEL_SEPARATORS)
            chunks = splitter.split_documents([doc])
            for i, chunk in enumerate(chunks):
                chunk.metadata.setdefault("source", source)
                chunk.metadata["doc_type"] = "excel"
                chunk.metadata["chunk_order"] = i
                # No parent for Excel chunks — parent_index absent means
                # the retriever falls back to the child chunk itself.
            all_children.extend(chunks)
            continue

        # Split into parent sections first
        parents = parent_splitter.split_documents([doc])
        for parent in parents:
            p_idx = global_parent_idx
            global_parent_idx += 1

            parent.metadata.setdefault("source", source)
            parent.metadata["doc_type"] = doc_type or "unknown"
            parent.metadata["parent_index"] = p_idx
            all_parents.append(parent)

            # Split each parent into children
            children = child_splitter.split_documents([parent])
            for c_order, child in enumerate(children):
                child.metadata.setdefault("source", source)
                child.metadata["doc_type"] = doc_type or "unknown"
                child.metadata["parent_index"] = p_idx   # key to look up parent
                child.metadata["chunk_order"] = c_order
            all_children.extend(children)

    return all_children, all_parents
