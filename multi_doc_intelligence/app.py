from __future__ import annotations

import io
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import streamlit as st

from config import DOC_REGISTRY_PATH, PATHS, ensure_storage_dirs
from ui.chat_page import render_chat_page
from ui.upload_page import render_upload_page


def _normalize_session_id(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = value[0] if value else ""
    normalized = str(value).strip()
    if not normalized or normalized.lower() == "none":
        return ""
    return normalized


def _load_registry_payload() -> dict:
    if not DOC_REGISTRY_PATH.exists():
        return {"sessions": {}}
    try:
        return json.loads(DOC_REGISTRY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"sessions": {}}


def _write_registry_payload(payload: dict) -> None:
    DOC_REGISTRY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _cleanup_unreferenced_uploads() -> None:
    payload = _load_registry_payload()
    sessions = payload.get("sessions", {}) if isinstance(payload, dict) else {}
    referenced_files: set[str] = set()

    if isinstance(sessions, dict):
        for meta in sessions.values():
            if not isinstance(meta, dict):
                continue
            documents = meta.get("documents", [])
            if not isinstance(documents, list):
                continue
            for item in documents:
                if not isinstance(item, dict):
                    continue
                filename = str(item.get("filename", "")).strip()
                if filename:
                    referenced_files.add(filename)

    if not PATHS.upload_dir.exists():
        return

    for upload_file in PATHS.upload_dir.iterdir():
        if not upload_file.is_file():
            continue
        if upload_file.name not in referenced_files:
            upload_file.unlink(missing_ok=True)


def _delete_session_local(session_id: str) -> tuple[bool, str]:
    normalized_session_id = _normalize_session_id(session_id)
    if not normalized_session_id:
        return False, "Invalid session id."

    payload = _load_registry_payload()
    sessions = payload.get("sessions", {})
    if not isinstance(sessions, dict):
        sessions = {}
        payload["sessions"] = sessions

    sessions.pop(normalized_session_id, None)
    _write_registry_payload(payload)

    for folder in (
        PATHS.faiss_index_dir / normalized_session_id,
        PATHS.summary_index_dir / normalized_session_id,
        PATHS.threads_dir / normalized_session_id,
    ):
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)

    _cleanup_unreferenced_uploads()
    return True, "Session deleted from local storage."


def _add_directory_to_zip(zf: zipfile.ZipFile, base_dir: Path, archive_prefix: str) -> None:
    if not base_dir.exists() or not base_dir.is_dir():
        return

    for item in base_dir.rglob("*"):
        if item.is_file():
            relative = item.relative_to(base_dir)
            zf.write(item, arcname=f"{archive_prefix}/{relative.as_posix()}")


def _build_session_export(session_id: str) -> tuple[str, bytes]:
    normalized_session_id = _normalize_session_id(session_id)
    if not normalized_session_id:
        raise ValueError("Invalid session id")

    payload = _load_registry_payload()
    sessions = payload.get("sessions", {}) if isinstance(payload, dict) else {}
    session_meta = sessions.get(normalized_session_id, {}) if isinstance(sessions, dict) else {}
    export_name = f"session_{normalized_session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        manifest = {
            "session_id": normalized_session_id,
            "exported_at": datetime.now().isoformat(),
            "session": session_meta if isinstance(session_meta, dict) else {},
        }
        zf.writestr("session/manifest.json", json.dumps(manifest, indent=2))

        if DOC_REGISTRY_PATH.exists():
            zf.write(DOC_REGISTRY_PATH, arcname="session/doc_registry.json")

        _add_directory_to_zip(
            zf,
            PATHS.faiss_index_dir / normalized_session_id,
            "session/faiss_index",
        )
        _add_directory_to_zip(
            zf,
            PATHS.summary_index_dir / normalized_session_id,
            "session/summary_index",
        )
        _add_directory_to_zip(
            zf,
            PATHS.threads_dir / normalized_session_id,
            "session/threads",
        )

        referenced_files: set[str] = set()
        if isinstance(session_meta, dict):
            documents = session_meta.get("documents", [])
            if isinstance(documents, list):
                for item in documents:
                    if not isinstance(item, dict):
                        continue
                    filename = str(item.get("filename", "")).strip()
                    if filename:
                        referenced_files.add(filename)

        for filename in sorted(referenced_files):
            upload_path = PATHS.upload_dir / filename
            if upload_path.exists() and upload_path.is_file():
                zf.write(upload_path, arcname=f"session/uploads/{filename}")

    return export_name, buffer.getvalue()


def _saved_sessions() -> list[dict[str, str]]:
    if not DOC_REGISTRY_PATH.exists():
        return []

    try:
        payload = json.loads(DOC_REGISTRY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    sessions = payload.get("sessions", {})
    if not isinstance(sessions, dict):
        return []

    saved: list[dict[str, str]] = []
    for session_id, meta in sessions.items():
        if not isinstance(meta, dict):
            continue
        documents = meta.get("documents", [])
        if not isinstance(documents, list) or not documents:
            continue

        session_name = str(meta.get("session_name", "")).strip()
        if not session_name:
            first_doc = next((item for item in documents if isinstance(item, dict)), {})
            session_name = str(first_doc.get("filename", session_id)).strip()
            if session_name.endswith(".pdf") or session_name.endswith(".docx") or session_name.endswith(".txt") or session_name.endswith(".xlsx"):
                session_name = session_name.rsplit(".", 1)[0]

        saved.append({"session_id": str(session_id), "session_name": session_name})

    saved.sort(key=lambda item: item["session_name"].lower())
    return saved


def _session_display_name_map(saved_sessions: list[dict[str, str]]) -> dict[str, str]:
    display_name_by_id: dict[str, str] = {}
    name_counts: dict[str, int] = {}

    for item in saved_sessions:
        session_id = item["session_id"]
        base_name = item["session_name"] or "Session"
        count = name_counts.get(base_name, 0) + 1
        name_counts[base_name] = count
        display_name = base_name if count == 1 else f"{base_name} ({count})"
        display_name_by_id[session_id] = display_name

    return display_name_by_id


def _current_session_name(session_id: str, saved_sessions: list[dict[str, str]]) -> str:
    for item in saved_sessions:
        if item["session_id"] == session_id:
            return item["session_name"]
    return session_id


def _rename_session(session_id: str, new_name: str) -> bool:
    clean_name = new_name.strip()
    if not clean_name:
        return False

    payload = _load_registry_payload()
    sessions = payload.setdefault("sessions", {})
    session_bucket = sessions.setdefault(session_id, {"session_name": clean_name, "documents": []})
    if not isinstance(session_bucket, dict):
        return False

    session_bucket["session_name"] = clean_name
    if "documents" not in session_bucket or not isinstance(session_bucket["documents"], list):
        session_bucket["documents"] = []

    _write_registry_payload(payload)
    return True


def main() -> None:
    ensure_storage_dirs()
    st.set_page_config(page_title="Multi-Document Intelligence", page_icon="📚", layout="wide")
    st.title("Multi-Document Intelligence Dashboard")
    st.caption("Cloud-based document intelligence with Groq API and LCEL orchestration.")

    if "session_id" not in st.session_state:
        sid_from_query = _normalize_session_id(st.query_params.get("sid"))
        st.session_state.session_id = sid_from_query or uuid4().hex[:12]
    else:
        st.session_state.session_id = _normalize_session_id(st.session_state.session_id) or uuid4().hex[:12]

    if _normalize_session_id(st.query_params.get("sid")) != st.session_state.session_id:
        st.query_params["sid"] = st.session_state.session_id


    with st.sidebar:
        st.subheader("Session")

        if st.button("New session", key="create_new_session_btn", use_container_width=True):
            st.session_state.session_id = uuid4().hex[:12]
            st.session_state.pop("threads", None)
            st.session_state.pop("active_thread", None)
            st.session_state.pop("session_export_name", None)
            st.session_state.pop("session_export_bytes", None)
            st.query_params["sid"] = st.session_state.session_id
            st.rerun()

        saved_sessions = _saved_sessions()
        display_name_by_id = _session_display_name_map(saved_sessions)

        session_options = [_normalize_session_id(item["session_id"]) for item in saved_sessions]
        session_options = [sid for sid in session_options if sid]
        if st.session_state.session_id not in session_options:
            session_options.insert(0, st.session_state.session_id)
            display_name_by_id.setdefault(st.session_state.session_id, "Current Session")

        selected_index = session_options.index(st.session_state.session_id) if st.session_state.session_id in session_options else 0
        selected_session_id = st.selectbox(
            "Open session",
            options=session_options,
            index=selected_index,
            format_func=lambda sid: display_name_by_id.get(sid, sid),
        )

        normalized_selected_session_id = _normalize_session_id(selected_session_id)
        if normalized_selected_session_id and normalized_selected_session_id != st.session_state.session_id:
            st.session_state.session_id = normalized_selected_session_id
            st.session_state.pop("threads", None)
            st.session_state.pop("active_thread", None)
            st.session_state.pop("session_export_name", None)
            st.session_state.pop("session_export_bytes", None)
            st.query_params["sid"] = st.session_state.session_id
            st.rerun()

        st.markdown("#### Session Lifecycle")
        if st.button("Prepare export", key="prepare_session_export_btn", use_container_width=True):
            try:
                export_name, export_bytes = _build_session_export(st.session_state.session_id)
                st.session_state.session_export_name = export_name
                st.session_state.session_export_bytes = export_bytes
                st.success("Session export is ready.")
            except Exception as exc:
                st.error(f"Failed to prepare export: {exc}")

        export_name = str(st.session_state.get("session_export_name", "")).strip()
        export_bytes = st.session_state.get("session_export_bytes")
        if export_name and isinstance(export_bytes, (bytes, bytearray)):
            st.download_button(
                "Download session export",
                data=bytes(export_bytes),
                file_name=export_name,
                mime="application/zip",
                use_container_width=True,
                key="download_session_export_btn",
            )

        confirm_delete = st.checkbox("Confirm delete selected session", key=f"confirm_delete_session_{st.session_state.session_id}")
        if st.button("Delete selected session", key="delete_selected_session_btn", use_container_width=True):
            if not confirm_delete:
                st.warning("Please confirm delete before continuing.")
            else:
                deleted_session_id = st.session_state.session_id
                ok, message = _delete_session_local(deleted_session_id)
                if ok:
                    st.session_state.pop("threads", None)
                    st.session_state.pop("active_thread", None)
                    st.session_state.pop("session_export_name", None)
                    st.session_state.pop("session_export_bytes", None)
                    st.session_state.session_id = uuid4().hex[:12]
                    st.query_params["sid"] = st.session_state.session_id
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        st.markdown("#### Rename Session")
        current_name = _current_session_name(st.session_state.session_id, saved_sessions)
        rename_key = f"rename_session_name_input_{st.session_state.session_id}"
        new_name = st.text_input("Session name", value=current_name, key=rename_key)
        if st.button("Save session name", key="save_session_name_btn"):
            if _rename_session(st.session_state.session_id, new_name):
                st.success("Session name updated.")
                st.rerun()
            else:
                st.warning("Please enter a valid session name.")

        page = st.radio("Page", options=["Upload", "Chat"])

    if page == "Upload":
        render_upload_page(session_id=st.session_state.session_id)
    else:
        render_chat_page(
            session_id=st.session_state.session_id,
            model_name=None,
        )


if __name__ == "__main__":
    main()
