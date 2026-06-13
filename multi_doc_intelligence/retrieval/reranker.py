"""
Reranking utilities for the hybrid retrieval pipeline.

Two stages:
  1. RRF fusion   — combines semantic + BM25 + summary rankings (fast, no model)
  2. Cross-encoder reranker — scores query-chunk pairs precisely (small model, CPU-friendly)

The cross-encoder uses 'cross-encoder/ms-marco-MiniLM-L-6-v2' (~22 MB),
auto-downloaded via sentence-transformers (already in requirements.txt).
Scoring 15 candidates takes ~100-200ms on CPU — hidden by LLM response time.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy cross-encoder singleton
# ---------------------------------------------------------------------------

_CE_MODEL: Any = None
_CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_cross_encoder():
    global _CE_MODEL
    if _CE_MODEL is not None:
        return _CE_MODEL
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
        _CE_MODEL = CrossEncoder(_CE_MODEL_NAME)
        logger.info("Cross-encoder loaded: %s", _CE_MODEL_NAME)
    except Exception as exc:
        logger.warning("Cross-encoder unavailable (%s), using RRF scores only.", exc)
        _CE_MODEL = None
    return _CE_MODEL


# ---------------------------------------------------------------------------
# Stage 1: RRF fusion
# ---------------------------------------------------------------------------

def rrf_fuse(rankings: list[list[Document]], k: int = 60, top_n: int = 12) -> list[Document]:
    """
    Reciprocal Rank Fusion over multiple ranked result lists.

    top_n is set higher here (12) than the final fused_k sent to the LLM,
    because the cross-encoder will then further reduce this to the best 5.
    """
    score_by_chunk_id: dict[str, float] = defaultdict(float)
    doc_by_chunk_id: dict[str, Document] = {}

    for ranked_docs in rankings:
        for rank, doc in enumerate(ranked_docs, start=1):
            chunk_id = str(doc.metadata.get("chunk_id", f"fallback_{id(doc)}"))
            score_by_chunk_id[chunk_id] += 1.0 / (k + rank)
            doc_by_chunk_id[chunk_id] = doc

    fused_ids = sorted(score_by_chunk_id, key=score_by_chunk_id.__getitem__, reverse=True)[:top_n]
    return [doc_by_chunk_id[cid] for cid in fused_ids]


# ---------------------------------------------------------------------------
# Stage 2: Cross-encoder reranking
# ---------------------------------------------------------------------------

def rerank_with_cross_encoder(
    query: str,
    docs: list[Document],
    top_n: int = 5,
) -> list[Document]:
    """
    Re-score query-document pairs with a cross-encoder and return top_n.

    Falls back to the original RRF ordering if the cross-encoder is unavailable.
    This gives much higher precision than cosine similarity alone, especially
    for multi-part queries where relevant chunks may have lower embedding scores.
    """
    if not docs:
        return docs

    ce = _get_cross_encoder()
    if ce is None:
        # Graceful fallback: return the top_n from RRF ordering as-is
        return docs[:top_n]

    try:
        pairs = [(query, doc.page_content) for doc in docs]
        scores = ce.predict(pairs)
        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_n]]
    except Exception as exc:
        logger.warning("Cross-encoder scoring failed (%s), falling back to RRF order.", exc)
        return docs[:top_n]


def rerank_with_scores(
    query: str,
    docs: list[Document],
    top_n: int = 5,
) -> list[tuple[float, Document]]:
    """
    Like rerank_with_cross_encoder but also returns the cross-encoder scores.

    Returns a list of (score, Document) tuples sorted descending by score.
    The score indicates how relevant the document is to the query —
    used by the relevance gate to detect off-topic questions.
    """
    if not docs:
        return []

    ce = _get_cross_encoder()
    if ce is None:
        # No cross-encoder — return with fake scores (high enough to pass any gate)
        return [(1.0, doc) for doc in docs[:top_n]]

    try:
        pairs = [(query, doc.page_content) for doc in docs]
        scores = ce.predict(pairs)
        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [(float(s), d) for s, d in scored[:top_n]]
    except Exception as exc:
        logger.warning("Cross-encoder scoring failed (%s), falling back to RRF order.", exc)
        return [(1.0, doc) for doc in docs[:top_n]]

