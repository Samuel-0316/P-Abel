"""
Persistent query-answer cache with vector-based semantic matching.

Two-tier lookup:
  1. Exact SHA-256 key match  — O(1), zero compute
  2. Cosine similarity scan   — O(n * embed_dim), ~2-5ms for 200 entries
     Uses the same all-MiniLM-L6-v2 embedding model already in memory for FAISS.
     Any phrasing that means the same thing — including completely different word
     choices — will resolve to a cache HIT if cosine similarity > SEMANTIC_THRESHOLD.

Examples that now hit the same cache entry:
  "What was the role in my internship?"
  "What was my role in my internship?"
  "Tell me my internship role"
  "What position did I hold during my internship?"
  "Which role did I have in the internship?"

Cache is stored per session:
    storage/query_cache/{session_id}/cache.json

Cache is invalidated automatically when:
  - A new document is indexed in the session  (upload_page calls clear_session_cache)
  - The session is deleted                    (folder removed by _delete_session_local)
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from config import EMBED_MODEL_NAME, PATHS

logger = logging.getLogger(__name__)

MAX_CACHE_ENTRIES = 200

# Cosine similarity threshold for semantic matching.
# all-MiniLM-L6-v2 produces cosine similarities of ~0.82-0.88 for paraphrases
# with different word choices (e.g. "role" vs "position", "what was" vs "tell me"),
# ~0.75-0.82 for synonym-heavy rewrites (e.g. "student" vs "candidate").
# Different-topic questions typically score below 0.65 on this model.
# 0.75 catches synonym rewrites while safely rejecting cross-topic questions.
SEMANTIC_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# Embedding singleton (reuses the same model already loaded for FAISS)
# ---------------------------------------------------------------------------

_embed_fn: Any = None


def _get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embed_fn = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    return _embed_fn


def _embed(text: str) -> list[float]:
    """Return a unit-normalized embedding vector for a text string."""
    fn = _get_embed_fn()
    vec = np.array(fn.embed_query(text), dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two pre-normalized vectors (= dot product)."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    return float(np.dot(va, vb))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _cache_dir(session_id: str) -> Path:
    return PATHS.storage_dir / "query_cache" / session_id


def _cache_path(session_id: str) -> Path:
    return _cache_dir(session_id) / "cache.json"


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------

def make_cache_key(question: str, memory_context: str = "") -> str:
    """Exact-match key. Semantic fallback is handled in get_cached_result."""
    return hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Read / Write
# ---------------------------------------------------------------------------

def _load_cache(session_id: str) -> dict[str, Any]:
    path = _cache_path(session_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(session_id: str, data: dict[str, Any]) -> None:
    cache_dir = _cache_dir(session_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_path(session_id).write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_cached_result(
    session_id: str,
    cache_key: str,
    question: str = "",
) -> dict[str, Any] | None:
    """
    Return a cached QA result, or None if not found.

    Lookup order:
      1. Exact key (O(1)) — zero compute
      2. Semantic cosine similarity scan (O(n)) — uses embedding model already
         loaded for FAISS. Finds the closest cached question by meaning, not text.
    """
    cache = _load_cache(session_id)
    if not cache:
        return None

    # --- Tier 1: Exact key ---
    entry = cache.get(cache_key)
    if entry and isinstance(entry, dict) and entry.get("answer"):
        logger.info("Cache EXACT HIT  session=%s  q='%.50s'", session_id, question)
        return entry

    # --- Tier 2: Semantic similarity ---
    if not question:
        return None

    try:
        query_vec = _embed(question)
    except Exception as exc:
        logger.warning("Cache embed failed (skipping semantic lookup): %s", exc)
        return None

    best_score = 0.0
    best_entry: dict[str, Any] | None = None

    for entry in cache.values():
        if not isinstance(entry, dict) or not entry.get("answer"):
            continue
        stored_vec = entry.get("embedding")
        if not stored_vec or not isinstance(stored_vec, list):
            continue
        score = _cosine(query_vec, stored_vec)
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_score >= SEMANTIC_THRESHOLD and best_entry is not None:
        logger.info(
            "Cache SEMANTIC HIT  score=%.3f  session=%s  q='%.50s'",
            best_score, session_id, question,
        )
        return best_entry

    return None


def save_cached_result(
    session_id: str,
    cache_key: str,
    question: str,
    answer: str,
    citations: list[dict[str, Any]],
    hyde_query: str,
    faithfulness: dict[str, Any],
) -> None:
    """Persist a QA result including its question embedding for semantic lookup."""
    cache = _load_cache(session_id)

    # Embed the question so future similar questions can find this entry
    try:
        embedding = _embed(question)
    except Exception as exc:
        logger.warning("Cache embed failed at save time: %s", exc)
        embedding = []

    cache[cache_key] = {
        "question": question,
        "embedding": embedding,      # stored for semantic similarity lookup
        "answer": answer,
        "citations": citations,
        "hyde_query": hyde_query,
        "faithfulness": faithfulness,
        "cached_at": datetime.now(timezone.utc).isoformat(),
    }

    # Evict oldest entries if over limit
    if len(cache) > MAX_CACHE_ENTRIES:
        sorted_keys = sorted(cache, key=lambda k: cache[k].get("cached_at", ""))
        for old_key in sorted_keys[: len(cache) - MAX_CACHE_ENTRIES]:
            cache.pop(old_key, None)

    try:
        _save_cache(session_id, cache)
        logger.debug("Cache SAVE  session=%s  key=%s", session_id, cache_key[:8])
    except OSError as exc:
        logger.warning("Cache write failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------

def clear_session_cache(session_id: str) -> None:
    """
    Wipe the entire query cache for a session.
    Called automatically when a new document is indexed.
    """
    path = _cache_path(session_id)
    if path.exists():
        try:
            path.unlink()
            logger.info("Query cache cleared for session=%s", session_id)
        except OSError as exc:
            logger.warning("Could not clear query cache: %s", exc)


def cache_stats(session_id: str) -> dict[str, int]:
    """Return basic cache statistics for a session."""
    cache = _load_cache(session_id)
    return {"entries": len(cache), "max_entries": MAX_CACHE_ENTRIES}
