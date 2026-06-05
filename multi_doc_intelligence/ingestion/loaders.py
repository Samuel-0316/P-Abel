from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyMuPDFLoader, TextLoader
from langchain_core.documents import Document

from ingestion.excel_parser import parse_excel_to_documents

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".xlsx"}


def load_documents(file_path: str | Path) -> list[Document]:
    """Load a file into LangChain Documents using a format-aware loader."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {ext}")

    if ext == ".pdf":
        docs = PyMuPDFLoader(str(path)).load()
        for doc in docs:
            doc.metadata["source"] = path.name
            doc.metadata["doc_type"] = "pdf"
        return docs

    if ext == ".docx":
        docs = Docx2txtLoader(str(path)).load()
        for doc in docs:
            doc.metadata["source"] = path.name
            doc.metadata["doc_type"] = "word"
        return docs

    if ext == ".txt":
        docs = TextLoader(str(path), encoding="utf-8").load()
        for doc in docs:
            doc.metadata["source"] = path.name
            doc.metadata["doc_type"] = "txt"
        return docs

    return parse_excel_to_documents(path)
