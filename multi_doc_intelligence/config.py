from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
FAISS_INDEX_DIR = STORAGE_DIR / "faiss_index"
SUMMARY_INDEX_DIR = STORAGE_DIR / "summary_index"
PARENTS_DIR = STORAGE_DIR / "parents"        # parent chunk store (new)
THREADS_DIR = STORAGE_DIR / "threads"
UPLOAD_DIR = STORAGE_DIR / "uploads"
DOC_REGISTRY_PATH = STORAGE_DIR / "doc_registry.json"

# ---------------------------------------------------------------------------
# Hierarchical chunking sizes
# Child chunks: small & precise → indexed in FAISS for embedding search
# Parent chunks: full sections  → fetched at retrieval time for LLM context
# ---------------------------------------------------------------------------
PARENT_CHUNK_SIZE = 1500    # ~375 tokens — a full document section
PARENT_CHUNK_OVERLAP = 100  # slight overlap so section boundaries don't cut sentences
CHILD_CHUNK_SIZE = 200      # ~50 tokens — precise embedding target
CHILD_CHUNK_OVERLAP = 20    # minimal overlap

# Legacy (kept for Excel which doesn't use the parent-child pattern)
CHUNK_SIZE = CHILD_CHUNK_SIZE
CHUNK_OVERLAP = CHILD_CHUNK_OVERLAP
EXCEL_CHUNK_SIZE = 256
EXCEL_CHUNK_OVERLAP = 32

# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("DEFAULT_OLLAMA_MODEL", "phi3.5:latest")

# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Retrieval efficiency
# ---------------------------------------------------------------------------
USE_HYDE = os.getenv("USE_HYDE", "true").lower() == "true"

# Context budget for the LLM prompt.
# Parent sections are ~1500 chars each; 4 parents = 6000 chars ≈ 1500 tokens.
# Gemini Flash (1M ctx) and phi3.5 (4K ctx) both handle this comfortably.
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))


@dataclass(frozen=True)
class AppPaths:
    base_dir: Path = BASE_DIR
    storage_dir: Path = STORAGE_DIR
    faiss_index_dir: Path = FAISS_INDEX_DIR
    summary_index_dir: Path = SUMMARY_INDEX_DIR
    parents_dir: Path = PARENTS_DIR
    threads_dir: Path = THREADS_DIR
    upload_dir: Path = UPLOAD_DIR
    doc_registry_path: Path = DOC_REGISTRY_PATH


PATHS = AppPaths()


def ensure_storage_dirs() -> None:
    """Create required on-disk folders if they do not exist."""
    PATHS.storage_dir.mkdir(parents=True, exist_ok=True)
    PATHS.faiss_index_dir.mkdir(parents=True, exist_ok=True)
    PATHS.summary_index_dir.mkdir(parents=True, exist_ok=True)
    PATHS.parents_dir.mkdir(parents=True, exist_ok=True)
    PATHS.threads_dir.mkdir(parents=True, exist_ok=True)
    PATHS.upload_dir.mkdir(parents=True, exist_ok=True)
