from __future__ import annotations

import streamlit as st

from chains.insight_chain import extract_insights
from chains.summarize_chain import summarize_document
from indexing.summary_index import build_summary_index
from indexing.vector_store import VectorStoreManager


def _load_document_text(session_id: str, doc_id: str) -> str:
    manager = VectorStoreManager(session_id=session_id)
    docs = manager.documents_for_doc_id(doc_id)
    if not docs:
        return ""
    return "\n\n".join(doc.page_content for doc in docs)


def _doc_label(doc: dict[str, str]) -> str:
    filename = str(doc.get("filename", "unknown"))
    version = str(doc.get("version", "?"))
    created = str(doc.get("created_at", ""))[:19].replace("T", " ")
    return f"{filename} | v{version} | {created}"

def render_analysis_page(session_id: str, model_name: str) -> None:
    st.subheader("Analysis")
    manager = VectorStoreManager(session_id=session_id)
    documents = manager.list_session_registry_documents()

    if not documents:
        st.info("No indexed documents found for this session. Upload files first.")
        return

    options = { _doc_label(doc): str(doc.get("doc_id", "")) for doc in documents }
    selected_label = st.selectbox("Select document version", list(options.keys()))
    selected_doc_id = options[selected_label]

    if not selected_doc_id:
        st.warning("Selected entry has no doc_id.")
        return

    text = _load_document_text(session_id, selected_doc_id)
    if not text:
        st.warning("No chunk content found for selected document.")
        return

    tab_summary, tab_insights = st.tabs(["Summary", "Insights"])

    with tab_summary:
        if st.button("Generate summary", key="summary_btn"):
            with st.spinner("Running summarization chain..."):
                try:
                    build_summary_index(session_id=session_id, doc_id=selected_doc_id, model_name=model_name)
                except Exception as exc:
                    st.warning(f"Summary index warm-up failed. Continuing with direct summary generation. Details: {exc}")
                summary = summarize_document(text, model_name=model_name)
            st.write(summary)

    with tab_insights:
        if st.button("Extract insights", key="insight_btn"):
            with st.spinner("Running insight extraction chain..."):
                insights = extract_insights(text, model_name=model_name)

            st.markdown("### Decisions")
            for item in insights.get("decisions", []):
                st.write(f"- {item}")

            st.markdown("### Action Items")
            for item in insights.get("action_items", []):
                st.write(f"- {item}")

            st.markdown("### Key Findings")
            for item in insights.get("key_findings", []):
                st.write(f"- {item}")
