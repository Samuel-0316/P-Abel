from __future__ import annotations

from pathlib import Path
import sys

from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PATHS, ensure_storage_dirs
from indexing.vector_store import VectorStoreManager


def _print(title: str, value: object) -> None:
    print(f"[faiss-smoke] {title}: {value}")


def main() -> None:
    ensure_storage_dirs()

    session_id = "faiss-smoke-session"
    manager = VectorStoreManager(session_id=session_id)

    # 1) deterministic duplicate handling
    test_file = PATHS.upload_dir / "faiss_smoke.txt"
    test_file.write_text("local faiss smoke document alpha beta gamma", encoding="utf-8")

    chunks = [
        Document(
            page_content="local faiss smoke document alpha beta gamma",
            metadata={"source": test_file.name, "doc_type": "txt"},
        )
    ]

    first = manager.ingest_document(file_path=test_file, chunks=chunks)
    second = manager.ingest_document(file_path=test_file, chunks=chunks)

    _print("first_ingest_status", first.get("status"))
    _print("second_ingest_status", second.get("status"))
    _print("registry_docs", len(manager.list_session_registry_documents()))

    # 2) retrieval availability
    hits = manager.similarity_search("alpha", k=3)
    _print("search_hits", len(hits))

    # 3) broken artifact self-repair behavior
    broken_session = "faiss-smoke-broken"
    broken_dir = PATHS.faiss_index_dir / broken_session
    broken_dir.mkdir(parents=True, exist_ok=True)
    (broken_dir / "index.faiss").write_bytes(b"broken")
    pkl_file = broken_dir / "index.pkl"
    if pkl_file.exists():
        pkl_file.unlink()

    broken_manager = VectorStoreManager(session_id=broken_session)
    safe_hits = broken_manager.similarity_search("hello", k=2)
    _print("broken_session_safe_hits", len(safe_hits))

    repair_probe = (broken_dir / "index.faiss").exists() or (broken_dir / "index.pkl").exists()
    _print("broken_artifacts_present_after_repair", repair_probe)


if __name__ == "__main__":
    main()
