"""
Hybrid retriever with parent-child expansion.

Retrieval pipeline
------------------
  1. HyDE query expansion (optional, skipped on Ollama)
  2. FAISS semantic search on child chunks  (k=10)
  3. BM25 keyword search  on child chunks  (k=10)
  4. Summary vector search (cache-only)     (k=3)
  5. RRF fusion → top-12 child candidates
  6. Cross-encoder reranking → top-5 best children
  7. Parent expansion: each child's parent_index + doc_id → load full section
  8. Deduplicate parents (multiple children may share one parent)
  9. Return parent sections (or child chunks as fallback for old/Excel docs)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from config import USE_HYDE
from indexing.parent_store import load_parent
from indexing.vector_store import VectorStoreManager
from retrieval.hyde import LLMProtocol, build_hyde_query
from retrieval.multi_vector import build_multi_vector_config, collect_summary_vector_candidates
from retrieval.reranker import rerank_with_cross_encoder, rrf_fuse

logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    question: str
    hyde_query: str
    documents: list[Document]   # parent sections (or child fallbacks)


class HybridRetriever:
    """
    Session-scoped hybrid retriever with parent-child expansion.

    Child chunks (small, ~200 chars) are retrieved for precision.
    Parent sections (large, ~1500 chars) are returned to the LLM for context.
    This resolves the "compensation spread over 5 pages" problem — retrieving
    any child chunk in a section automatically surfaces the full section.
    """

    def __init__(
        self,
        *,
        session_id: str,
        llm: LLMProtocol,
        child_semantic_k: int = 10,
        child_bm25_k: int = 10,
        rrf_top_n: int = 12,       # candidates after RRF (before cross-encoder)
        rerank_top_n: int = 5,     # children kept after cross-encoder
        use_hyde: bool | None = None,
    ):
        self.session_id = session_id
        self.llm = llm
        self.child_semantic_k = child_semantic_k
        self.child_bm25_k = child_bm25_k
        self.rrf_top_n = rrf_top_n
        self.rerank_top_n = rerank_top_n
        self.use_hyde = USE_HYDE if use_hyde is None else use_hyde
        self.vector_manager = VectorStoreManager(session_id=session_id)
        self.multi_vector_config = build_multi_vector_config()

    def _build_bm25_retriever(self) -> BM25Retriever | None:
        docs = self.vector_manager.all_session_documents()
        if not docs:
            return None
        retriever = BM25Retriever.from_documents(docs)
        retriever.k = self.child_bm25_k
        return retriever

    def _expand_to_parents(self, children: list[Document]) -> list[Document]:
        """
        For each child chunk, load its parent section from disk.

        Multiple children that share the same parent are deduplicated —
        the parent section is included only once even if 3 children from it
        were retrieved. This keeps context tight while ensuring completeness.

        Falls back to the child chunk itself for:
        - Excel documents (no parent stored)
        - Documents indexed before the parent-child architecture
        """
        seen_parent_keys: set[str] = set()
        result: list[Document] = []

        for child in children:
            doc_id = str(child.metadata.get("doc_id", ""))
            parent_index = child.metadata.get("parent_index")

            # No parent_index → old-format doc or Excel; use child directly
            if parent_index is None or not doc_id:
                chunk_id = str(child.metadata.get("chunk_id", id(child)))
                if chunk_id not in seen_parent_keys:
                    seen_parent_keys.add(chunk_id)
                    result.append(child)
                continue

            parent_key = f"{doc_id}__p{parent_index}"
            if parent_key in seen_parent_keys:
                continue   # already included this section
            seen_parent_keys.add(parent_key)

            parent = load_parent(
                session_id=self.session_id,
                doc_id=doc_id,
                parent_index=int(parent_index),
            )
            if parent is not None:
                result.append(parent)
            else:
                # Parent store missing (old index) — fall back to child
                result.append(child)

        return result

    def retrieve(self, question: str) -> RetrievedContext:
        # Step 1: HyDE (skipped on Ollama path)
        try:
            hyde_query = build_hyde_query(question, self.llm, use_hyde=self.use_hyde)
        except Exception:
            hyde_query = question

        # Step 2: Semantic search on child chunks (no score filter — cross-encoder handles precision)
        raw_semantic = self.vector_manager.similarity_search_with_score(
            hyde_query, k=self.child_semantic_k
        )
        semantic_docs = [doc for doc, _score in raw_semantic]

        # Step 3: BM25 keyword search on child chunks
        bm25 = self._build_bm25_retriever()
        bm25_docs = bm25.invoke(question) if bm25 is not None else []

        # Step 4: Summary vector search (cache-only, no LLM call)
        summary_docs: list[Document] = []
        if self.multi_vector_config.enable_summary_vectors:
            summary_docs = collect_summary_vector_candidates(
                session_id=self.session_id,
                question=question,
                top_k=self.multi_vector_config.summary_top_k,
            )

        # Step 5: RRF fusion over all three rankings → top-12 candidates
        fused_children = rrf_fuse(
            [semantic_docs, bm25_docs, summary_docs],
            top_n=self.rrf_top_n,
        )

        # Step 6: Cross-encoder reranking → top-5 most relevant children
        best_children = rerank_with_cross_encoder(
            question, fused_children, top_n=self.rerank_top_n
        )

        # Step 7+8: Expand each child to its parent section; deduplicate
        final_docs = self._expand_to_parents(best_children)

        return RetrievedContext(
            question=question,
            hyde_query=hyde_query,
            documents=final_docs,
        )
