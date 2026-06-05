from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import streamlit as st

from chains.qa_chain import QAResult, ask_question, clear_thread_memory, hydrate_thread_memory
from config import PATHS


def _session_threads_dir(session_id: str) -> Path:
    return PATHS.threads_dir / session_id


def _threads_file_path(session_id: str) -> Path:
    return _session_threads_dir(session_id) / "threads.json"


def _qa_result_to_meta(result: QAResult) -> dict[str, Any]:
    return {
        "citations": result.citations,
        "hyde_query": result.hyde_query,
        "faithfulness": {
            "faithful": result.faithfulness.faithful,
            "confidence": result.faithfulness.confidence,
            "reason": result.faithfulness.reason,
        },
        "error": result.error,
    }


def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "role": str(message.get("role", "assistant")),
        "content": str(message.get("content", "")),
    }
    if "meta" in message:
        meta = message.get("meta")
        if isinstance(meta, QAResult):
            normalized["meta"] = _qa_result_to_meta(meta)
        elif isinstance(meta, dict):
            normalized["meta"] = meta
    return normalized


def _load_threads_from_disk(session_id: str) -> tuple[dict[str, list[dict[str, Any]]], str]:
    file_path = _threads_file_path(session_id)
    if not file_path.exists():
        return {"thread-1": []}, "thread-1"

    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"thread-1": []}, "thread-1"

    raw_threads = payload.get("threads", {})
    if not isinstance(raw_threads, dict):
        return {"thread-1": []}, "thread-1"

    normalized_threads: dict[str, list[dict[str, Any]]] = {}
    for key, value in raw_threads.items():
        thread_id = str(key)
        messages = value if isinstance(value, list) else []
        normalized_threads[thread_id] = [
            _normalize_message(item) for item in messages if isinstance(item, dict)
        ]

    if not normalized_threads:
        normalized_threads = {"thread-1": []}

    active_thread = str(payload.get("active_thread", ""))
    if active_thread not in normalized_threads:
        active_thread = next(iter(normalized_threads.keys()))

    return normalized_threads, active_thread


def _save_threads_to_disk(session_id: str) -> None:
    threads = st.session_state.get("threads", {})
    if not isinstance(threads, dict):
        return

    normalized_threads: dict[str, list[dict[str, Any]]] = {}
    for key, value in threads.items():
        thread_id = str(key)
        messages = value if isinstance(value, list) else []
        normalized_threads[thread_id] = [
            _normalize_message(item) for item in messages if isinstance(item, dict)
        ]

    payload = {
        "active_thread": str(st.session_state.get("active_thread", "thread-1")),
        "threads": normalized_threads,
    }

    session_dir = _session_threads_dir(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    _threads_file_path(session_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _ensure_chat_state(session_id: str) -> None:
    if "threads" not in st.session_state:
        threads, active_thread = _load_threads_from_disk(session_id)
        st.session_state.threads = threads
        st.session_state.active_thread = active_thread

        for thread_id, messages in threads.items():
            hydrate_thread_memory(thread_id, messages)

    if "active_thread" not in st.session_state:
        st.session_state.active_thread = "thread-1"


def _delete_or_reset_active_thread(session_id: str, thread_id: str) -> None:
    threads = st.session_state.threads
    if len(threads) <= 1:
        threads[thread_id] = []
        st.session_state._next_active_thread = thread_id
        clear_thread_memory(thread_id)
    else:
        threads.pop(thread_id, None)
        clear_thread_memory(thread_id)
        st.session_state._next_active_thread = sorted(threads.keys())[0]

    _save_threads_to_disk(session_id)


def _clean_faithfulness_reason(reason: str) -> str:
    text = reason.strip()
    if not text:
        return "Faithfulness score generated from retrieved context."

    lowered = text.lower()
    if "specifically documents" in lowered or "provided context documents" in lowered:
        return "The answer is grounded in the retrieved evidence from your indexed source files."

    # Collapse noisy patterns like "documents 2, 3, 5" while preserving the main message.
    cleaned = re.sub(r"\(\s*specifically\s+documents?[^\)]*\)", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bdocuments?\s+(\d+[\s,]*)+", "retrieved evidence ", cleaned, flags=re.IGNORECASE)
    return cleaned.strip() or "Faithfulness score generated from retrieved context."


def _render_citations(result: QAResult | dict[str, Any]) -> None:
    if isinstance(result, QAResult):
        faithful = result.faithfulness.faithful
        confidence = result.faithfulness.confidence
        reason = result.faithfulness.reason
        citations = result.citations
    else:
        faithfulness = result.get("faithfulness", {}) if isinstance(result, dict) else {}
        faithful = bool(faithfulness.get("faithful", False))
        confidence = float(faithfulness.get("confidence", 0.0))
        reason = str(faithfulness.get("reason", "Unavailable"))
        citations = result.get("citations", []) if isinstance(result, dict) else []

    grouped_citations: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for cite in citations:
        if not isinstance(cite, dict):
            continue

        source = str(cite.get("source", "unknown")).strip() or "unknown"

        raw_page = str(cite.get("page", "")).strip()
        page = ""
        if raw_page and raw_page.lower() not in {"n/a", "none"}:
            try:
                page_number = int(raw_page)
                # Backward compatibility: treat only page 0 as legacy zero-based.
                page = "1" if page_number == 0 else str(page_number)
            except ValueError:
                page = raw_page

        sheet = str(cite.get("sheet", "")).strip()
        if sheet.lower() in {"n/a", "none"}:
            sheet = ""

        cell_range = str(cite.get("cell_range", "")).strip()
        if cell_range.lower() in {"n/a", "none"}:
            cell_range = ""

        key = (source, page, sheet, cell_range)
        group = grouped_citations.setdefault(
            key,
            {
                "source": source,
                "page": page,
                "sheet": sheet,
                "cell_range": cell_range,
                "hits": 0,
                "excerpt": "",
            },
        )

        group["hits"] = int(group["hits"]) + 1
        if not group["excerpt"]:
            excerpt = str(cite.get("excerpt", "")).strip()
            if excerpt:
                group["excerpt"] = excerpt

    ordered_citations = sorted(
        grouped_citations.values(),
        key=lambda item: (
            str(item.get("source", "")).lower(),
            int(item.get("page")) if str(item.get("page", "")).isdigit() else 10**9,
            str(item.get("sheet", "")).lower(),
            str(item.get("cell_range", "")).lower(),
        ),
    )

    # If a source has at least one concrete location, hide generic location-less rows for that source.
    source_has_location: dict[str, bool] = {}
    for cite in ordered_citations:
        source = str(cite.get("source", "unknown"))
        has_location = bool(str(cite.get("page", "")).strip() or str(cite.get("sheet", "")).strip() or str(cite.get("cell_range", "")).strip())
        source_has_location[source] = source_has_location.get(source, False) or has_location

    filtered_citations: list[dict[str, Any]] = []
    for cite in ordered_citations:
        source = str(cite.get("source", "unknown"))
        has_location = bool(str(cite.get("page", "")).strip() or str(cite.get("sheet", "")).strip() or str(cite.get("cell_range", "")).strip())
        if source_has_location.get(source, False) and not has_location:
            continue
        filtered_citations.append(cite)

    with st.expander("Sources and confidence", expanded=False):
        st.write(f"Faithful: {faithful}")
        st.write(f"Confidence: {confidence:.2f}")
        st.write(f"Reason: {_clean_faithfulness_reason(reason)}")

        if not filtered_citations:
            st.info("No source citations were available for this response.")
            return

        unique_sources = {
            str(cite.get("source", "unknown")).strip() or "unknown"
            for cite in filtered_citations
            if isinstance(cite, dict)
        }
        st.write(f"Sources used: {len(unique_sources)}")

        st.write("Evidence references:")
        for idx, cite in enumerate(filtered_citations, start=1):
            if not isinstance(cite, dict):
                continue

            source = str(cite.get("source", "unknown")).strip() or "unknown"
            location_parts: list[str] = []

            page = str(cite.get("page", "")).strip()
            if page and page.lower() not in {"n/a", "none"}:
                location_parts.append(f"Page {page}")

            sheet = str(cite.get("sheet", "")).strip()
            if sheet and sheet.lower() not in {"n/a", "none"}:
                location_parts.append(f"Sheet {sheet}")

            cell_range = str(cite.get("cell_range", "")).strip()
            if cell_range and cell_range.lower() not in {"n/a", "none"}:
                location_parts.append(f"Cells {cell_range}")

            chunk_id = str(cite.get("chunk_id", "")).strip()
            if chunk_id and chunk_id.lower() not in {"n/a", "none"}:
                location_parts.append(f"Chunk {chunk_id}")

            location_text = " | ".join(location_parts) if location_parts else "General document context"

            st.markdown(f"{idx}. **{source}**")
            st.caption(location_text)

            hits = int(cite.get("hits", 0))
            if hits > 1:
                st.caption(f"Matched chunks: {hits}")


def _render_thread_messages(thread_id: str) -> None:
    for msg in st.session_state.threads.get(thread_id, []):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "meta" in msg:
                _render_citations(msg["meta"])


def render_chat_page(session_id: str, model_name: str) -> None:
    st.subheader("Chat")
    _ensure_chat_state(session_id)

    # Apply any pending thread switch (set before the widget renders to avoid
    # the "cannot modify after widget is instantiated" error).
    if "_next_active_thread" in st.session_state:
        next_thread = st.session_state.pop("_next_active_thread")
        if next_thread in st.session_state.threads:
            st.session_state.active_thread = next_thread

    with st.sidebar:
        st.markdown("### Thread Management")
        selected = st.selectbox("Active thread", options=list(st.session_state.threads.keys()), key="active_thread")
        if st.button("New thread"):
            new_id = f"thread-{len(st.session_state.threads) + 1}"
            st.session_state.threads[new_id] = []
            st.session_state._next_active_thread = new_id
            _save_threads_to_disk(session_id)
            st.rerun()
        if st.button("Delete active thread"):
            _delete_or_reset_active_thread(session_id, selected)
            st.rerun()

    thread_id = selected
    _render_thread_messages(thread_id)

    user_message = st.chat_input("Ask a question grounded in uploaded documents")
    if not user_message:
        return

    st.session_state.threads[thread_id].append({"role": "user", "content": user_message})
    _save_threads_to_disk(session_id)
    with st.chat_message("user"):
        st.write(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and generating grounded answer..."):
            result = ask_question(
                question=user_message,
                thread_id=thread_id,
                session_id=session_id,
                model_name=model_name,
            )
        st.write(result.answer)
        if result.error:
            st.warning(f"⚠️ LLM pipeline note: {result.error}", icon="⚠️")
        _render_citations(result)

    st.session_state.threads[thread_id].append(
        {
            "role": "assistant",
            "content": result.answer,
            "meta": _qa_result_to_meta(result),
        }
    )
    _save_threads_to_disk(session_id)
