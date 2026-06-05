from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import DOC_REGISTRY_PATH, EMBED_MODEL_NAME, PATHS, ensure_storage_dirs


_EMBEDDINGS_CACHE: dict[str, Any] = {}


def _get_embeddings(model_name: str):
    cached = _EMBEDDINGS_CACHE.get(model_name)
    if cached is not None:
        return cached

    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    _EMBEDDINGS_CACHE[model_name] = embeddings
    return embeddings


class VectorStoreManager:
    """Persist and query a FAISS index with session/document metadata."""

    def __init__(self, session_id: str):
        if not session_id:
            raise ValueError("session_id is required")

        self.session_id = session_id
        self.index_dir = PATHS.faiss_index_dir / session_id
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = _get_embeddings(EMBED_MODEL_NAME)

    def _registry_payload(self) -> dict[str, Any]:
        if not DOC_REGISTRY_PATH.exists():
            return {"sessions": {}}
        try:
            return json.loads(DOC_REGISTRY_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Keep local-first behavior resilient even if registry file is malformed.
            return {"sessions": {}}

    def _write_registry(self, payload: dict[str, Any]) -> None:
        DOC_REGISTRY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _session_bucket(self, payload: dict[str, Any]) -> dict[str, Any]:
        sessions = payload.setdefault("sessions", {})
        return sessions.setdefault(self.session_id, {"session_name": self.session_id, "documents": []})

    def _ensure_session_name(self, filename: str | None = None) -> str:
        payload = self._registry_payload()
        session_bucket = self._session_bucket(payload)
        session_name = str(session_bucket.get("session_name", "")).strip()
        if not session_name:
            session_name = Path(filename).stem if filename else self.session_id
            session_bucket["session_name"] = session_name
            self._write_registry(payload)
        return session_name

    def session_display_name(self) -> str:
        payload = self._registry_payload()
        session_bucket = payload.get("sessions", {}).get(self.session_id, {})
        session_name = str(session_bucket.get("session_name", "")).strip()
        if session_name:
            return session_name

        documents = session_bucket.get("documents", [])
        if isinstance(documents, list):
            for item in documents:
                if isinstance(item, dict):
                    filename = str(item.get("filename", "")).strip()
                    if filename:
                        return Path(filename).stem

        return self.session_id

    def _next_version(self, filename: str, file_hash: str) -> int:
        payload = self._registry_payload()
        session_bucket = self._session_bucket(payload)
        versions = [
            item["version"]
            for item in session_bucket["documents"]
            if item.get("filename") == filename and item.get("file_hash") != file_hash
        ]
        return max(versions, default=0) + 1

    def _find_existing_by_hash(self, filename: str, file_hash: str) -> dict[str, Any] | None:
        payload = self._registry_payload()
        session_bucket = payload.get("sessions", {}).get(self.session_id, {})
        documents = session_bucket.get("documents", [])
        if not isinstance(documents, list):
            return None

        matches = [
            item
            for item in documents
            if isinstance(item, dict)
            and item.get("filename") == filename
            and item.get("file_hash") == file_hash
            and item.get("doc_id")
        ]
        if not matches:
            return None

        matches.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        return matches[0]

    def _clear_session_registry(self) -> None:
        payload = self._registry_payload()
        session_bucket = self._session_bucket(payload)
        session_name = str(session_bucket.get("session_name", self.session_id)).strip() or self.session_id
        payload.setdefault("sessions", {})[self.session_id] = {
            "session_name": session_name,
            "documents": [],
        }
        self._write_registry(payload)

    def check_session_index_health(self) -> dict[str, str]:
        """Inspect current session FAISS artifacts without mutating on-disk state."""
        faiss_file = self.index_dir / "index.faiss"
        pkl_file = self.index_dir / "index.pkl"

        if not faiss_file.exists() and not pkl_file.exists():
            return {"status": "clean", "action": "none"}

        if faiss_file.exists() != pkl_file.exists():
            return {"status": "broken", "action": "incomplete_index"}

        try:
            FAISS.load_local(
                folder_path=str(self.index_dir),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
            return {"status": "healthy", "action": "none"}
        except Exception:
            return {"status": "broken", "action": "corrupt_index"}

    def repair_session_index(self) -> dict[str, str]:
        """Attempt to repair broken local FAISS index artifacts for this session."""
        faiss_file = self.index_dir / "index.faiss"
        pkl_file = self.index_dir / "index.pkl"

        if not faiss_file.exists() and not pkl_file.exists():
            return {"status": "clean", "action": "none"}

        if faiss_file.exists() != pkl_file.exists():
            if faiss_file.exists():
                faiss_file.unlink(missing_ok=True)
            if pkl_file.exists():
                pkl_file.unlink(missing_ok=True)
            self._clear_session_registry()
            return {"status": "repaired", "action": "reset_incomplete_index"}

        try:
            FAISS.load_local(
                folder_path=str(self.index_dir),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
            return {"status": "healthy", "action": "none"}
        except Exception:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            quarantine_dir = self.index_dir / f"corrupt_{timestamp}"
            quarantine_dir.mkdir(parents=True, exist_ok=True)

            if faiss_file.exists():
                faiss_file.rename(quarantine_dir / "index.faiss")
            if pkl_file.exists():
                pkl_file.rename(quarantine_dir / "index.pkl")

            self._clear_session_registry()
            return {"status": "repaired", "action": "quarantined_corrupt_index"}

    def _append_registry(
        self,
        *,
        doc_id: str,
        filename: str,
        file_hash: str,
        version: int,
        chunk_count: int,
    ) -> None:
        payload = self._registry_payload()
        session_bucket = self._session_bucket(payload)
        session_bucket["session_name"] = str(session_bucket.get("session_name", "")).strip() or Path(filename).stem
        session_bucket["documents"].append(
            {
                "doc_id": doc_id,
                "filename": filename,
                "file_hash": file_hash,
                "version": version,
                "chunk_count": chunk_count,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._write_registry(payload)

    def _load_faiss(self) -> FAISS | None:
        faiss_file = self.index_dir / "index.faiss"
        pkl_file = self.index_dir / "index.pkl"

        if not faiss_file.exists() and not pkl_file.exists():
            return None

        if faiss_file.exists() != pkl_file.exists():
            raise RuntimeError(
                "FAISS index files are incomplete for this session. "
                "Expected both index.faiss and index.pkl."
            )

        if not faiss_file.exists():
            return None
        try:
            return FAISS.load_local(
                folder_path=str(self.index_dir),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load FAISS index for session '{self.session_id}': {exc}") from exc

    def _save_faiss(self, store: FAISS) -> None:
        store.save_local(folder_path=str(self.index_dir))
        faiss_file = self.index_dir / "index.faiss"
        pkl_file = self.index_dir / "index.pkl"
        if not faiss_file.exists() or not pkl_file.exists():
            raise RuntimeError("FAISS index save was incomplete (index.faiss/index.pkl missing after save).")

    @staticmethod
    def _hash_file(file_path: str | Path) -> str:
        path = Path(file_path)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def ingest_document(self, *, file_path: str | Path, chunks: list[Document]) -> dict[str, Any]:
        ensure_storage_dirs()
        if not chunks:
            raise ValueError("No chunks to ingest")

        source_path = Path(file_path)
        file_hash = self._hash_file(source_path)
        self._ensure_session_name(source_path.name)

        existing_same_hash = self._find_existing_by_hash(source_path.name, file_hash)
        if existing_same_hash is not None:
            return {
                "doc_id": str(existing_same_hash.get("doc_id", "")),
                "session_id": self.session_id,
                "version": int(existing_same_hash.get("version", 1)),
                "chunk_count": int(existing_same_hash.get("chunk_count", 0)),
                "index_dir": str(self.index_dir),
                "status": "skipped",
                "reason": "duplicate_content",
            }

        version = self._next_version(source_path.name, file_hash)
        doc_id = f"{source_path.stem}_v{version}_{file_hash[:8]}"

        for idx, chunk in enumerate(chunks):
            chunk.metadata.setdefault("source", source_path.name)
            chunk.metadata["doc_id"] = doc_id
            chunk.metadata["version"] = version
            chunk.metadata["session_id"] = self.session_id
            chunk.metadata["chunk_id"] = f"{doc_id}_c{idx:05d}"

        repair_action = "none"
        try:
            existing = self._load_faiss()
        except RuntimeError:
            repair = self.repair_session_index()
            repair_action = repair.get("action", "none")
            existing = None

        if existing is None:
            store = FAISS.from_documents(chunks, self.embeddings)
        else:
            existing.add_documents(chunks)
            store = existing

        self._save_faiss(store)
        self._append_registry(
            doc_id=doc_id,
            filename=source_path.name,
            file_hash=file_hash,
            version=version,
            chunk_count=len(chunks),
        )

        return {
            "doc_id": doc_id,
            "session_id": self.session_id,
            "version": version,
            "chunk_count": len(chunks),
            "index_dir": str(self.index_dir),
            "status": "indexed",
            "repair_action": repair_action,
        }

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        try:
            store = self._load_faiss()
        except RuntimeError:
            self.repair_session_index()
            return []
        if store is None:
            return []
        return store.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        try:
            store = self._load_faiss()
        except RuntimeError:
            self.repair_session_index()
            return []
        if store is None:
            return []
        return store.similarity_search_with_score(query, k=k)

    def all_session_documents(self) -> list[Document]:
        """Return all documents currently present in the session FAISS docstore."""
        try:
            store = self._load_faiss()
        except RuntimeError:
            self.repair_session_index()
            return []
        if store is None:
            return []

        docstore = getattr(store, "docstore", None)
        if docstore is None:
            return []

        internal = getattr(docstore, "_dict", None)
        if not isinstance(internal, dict):
            return []

        docs: list[Document] = []
        for value in internal.values():
            if isinstance(value, Document):
                docs.append(value)
        return docs

    def documents_for_doc_id(self, doc_id: str) -> list[Document]:
        docs = [doc for doc in self.all_session_documents() if str(doc.metadata.get("doc_id", "")) == doc_id]
        docs = sorted(docs, key=lambda d: str(d.metadata.get("chunk_id", "")))
        unique_docs: list[Document] = []
        seen_chunk_ids: set[str] = set()
        for doc in docs:
            chunk_id = str(doc.metadata.get("chunk_id", ""))
            if chunk_id and chunk_id in seen_chunk_ids:
                continue
            if chunk_id:
                seen_chunk_ids.add(chunk_id)
            unique_docs.append(doc)
        return unique_docs

    def list_session_registry_documents(self) -> list[dict[str, Any]]:
        payload = self._registry_payload()
        session_bucket = payload.get("sessions", {}).get(self.session_id, {})
        documents = session_bucket.get("documents", [])
        if not isinstance(documents, list):
            return []

        latest_by_doc_id: dict[str, dict[str, Any]] = {}
        for item in documents:
            if not isinstance(item, dict):
                continue
            doc_id = str(item.get("doc_id", "")).strip()
            if not doc_id:
                continue
            existing = latest_by_doc_id.get(doc_id)
            if existing is None or str(item.get("created_at", "")) > str(existing.get("created_at", "")):
                latest_by_doc_id[doc_id] = item

        return sorted(latest_by_doc_id.values(), key=lambda item: str(item.get("created_at", "")), reverse=True)
