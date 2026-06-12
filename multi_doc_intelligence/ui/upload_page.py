from __future__ import annotations

import streamlit as st

from config import PATHS
from indexing.parent_store import save_parents
from indexing.query_cache import clear_session_cache
from indexing.summary_index import build_summary_index
from indexing.vector_store import VectorStoreManager
from ingestion.chunker import chunk_documents, chunk_documents_hierarchical
from ingestion.loaders import load_documents


def render_upload_page(session_id: str) -> None:
    st.subheader("Upload Documents")

    with st.expander("Session FAISS Health", expanded=False):
        manager = VectorStoreManager(session_id=session_id)
        col_check, col_repair = st.columns(2)
        with col_check:
            if st.button("Check index health", key="check_faiss_health_btn"):
                health = manager.check_session_index_health()
                status = str(health.get("status", "unknown"))
                action = str(health.get("action", "none"))
                if status in {"healthy", "clean"}:
                    st.success(f"Session index is {status} (action={action}).")
                else:
                    st.warning(f"Session index is {status} (action={action}).")

        with col_repair:
            if st.button("Repair index", key="repair_faiss_health_btn"):
                repair = manager.repair_session_index()
                repaired_status = str(repair.get("status", "unknown"))
                repaired_action = str(repair.get("action", "none"))
                if repaired_status == "repaired":
                    st.warning(f"Repair applied: {repaired_action}")
                else:
                    st.info(f"No repair needed (status={repaired_status}, action={repaired_action}).")

    files = st.file_uploader(
        "Upload PDF, DOCX, TXT, or XLSX files",
        type=["pdf", "docx", "txt", "xlsx"],
        accept_multiple_files=True,
    )

    if not files:
        return

    if not st.button("Index uploaded files", key="index_uploaded_files_btn"):
        st.caption("Select files, then click 'Index uploaded files' to process once.")
        return

    manager = VectorStoreManager(session_id=session_id)

    for uploaded in files:
        target = PATHS.upload_dir / uploaded.name
        target.write_bytes(uploaded.getbuffer())

        with st.status(f"Indexing {uploaded.name}", expanded=False) as status:
            docs = load_documents(target)

            # Detect doc type to decide chunking strategy
            doc_type = docs[0].metadata.get("doc_type", "unknown") if docs else "unknown"
            is_excel = doc_type == "excel"

            if is_excel:
                # Excel uses single-level chunking (rows are already atomic)
                children = chunk_documents(docs)
                parents = []
            else:
                # PDF/DOCX/TXT: hierarchical parent-child chunking
                children, parents = chunk_documents_hierarchical(docs)

            # Ingest child chunks into FAISS — doc_id is assigned here
            result = manager.ingest_document(file_path=target, chunks=children)

            if result.get("status") == "skipped":
                status.update(
                    label=f"Skipped {uploaded.name} (duplicate content)",
                    state="complete",
                )
            else:
                doc_id = str(result.get("doc_id", ""))
                chunk_count = int(result.get("chunk_count", 0))

                # Save parent sections to disk (keyed by doc_id now known)
                if parents and doc_id:
                    status.update(
                        label=f"Saving {len(parents)} parent sections…",
                        state="running",
                    )
                    # Stamp doc_id onto parent metadata before saving
                    for p in parents:
                        p.metadata["doc_id"] = doc_id
                        p.metadata["session_id"] = session_id
                    try:
                        save_parents(session_id=session_id, doc_id=doc_id, parents=parents)
                    except Exception as exc:
                        st.warning(f"Parent store save failed (non-fatal): {exc}")

                # Build summary index at ingest time (cached, not rebuilt at query time)
                if doc_id:
                    status.update(label="Building document summary…", state="running")
                    try:
                        build_summary_index(
                            session_id=session_id,
                            doc_id=doc_id,
                            model_name=None,
                        )
                    except Exception:
                        pass  # Non-fatal

                parent_info = f", {len(parents)} sections" if parents else ""
                status.update(
                    label=f"Indexed {uploaded.name} — {chunk_count} chunks{parent_info}",
                    state="complete",
                )

        if result.get("status") == "skipped":
            st.info(f"{uploaded.name}: duplicate content already indexed as {result['doc_id']}")
        else:
            parent_note = f" · {len(parents)} parent sections stored" if parents else ""
            st.success(f"{uploaded.name}: {result['chunk_count']} child chunks indexed{parent_note}")
            # Invalidate query cache — new document changes what's retrievable
            clear_session_cache(session_id)

        repair_action = str(result.get("repair_action", "none"))
        if repair_action != "none":
            st.warning(
                f"Index auto-repair applied ({repair_action}) before indexing {uploaded.name}."
            )
