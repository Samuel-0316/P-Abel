from __future__ import annotations

import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from chains.llm_builder import build_llm


def _split_text(text: str, chunk_size: int = 3500) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    return [cleaned[i : i + chunk_size] for i in range(0, len(cleaned), chunk_size)]


def _build_llm(model_name: str | None = None):
    return build_llm(model_name=model_name, provider="gemini", temperature=0)


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "resource_exhausted" in msg
        or "quota" in msg
        or "429" in msg
        or "rate limit" in msg
    )


def _top_sentences(text: str, limit: int = 4) -> list[str]:
    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]
    return sentences[:limit]


def _pick_by_keywords(text: str, keywords: list[str], limit: int = 4) -> list[str]:
    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]
    picked: list[str] = []
    for sentence in sentences:
        lower = sentence.lower()
        if any(keyword in lower for keyword in keywords):
            picked.append(sentence)
        if len(picked) >= limit:
            break
    return picked


def _local_fallback_summary(text: str) -> str:
    overview = _top_sentences(text, limit=3)
    decisions = _pick_by_keywords(text, ["decide", "approved", "selected", "agreed", "final"], limit=4)
    actions = _pick_by_keywords(text, ["action", "todo", "task", "next", "will", "owner", "deadline"], limit=4)
    risks = _pick_by_keywords(text, ["risk", "issue", "blocker", "delay", "concern", "dependency"], limit=4)

    def _section_lines(items: list[str]) -> str:
        if not items:
            return "- Not explicitly identified from local fallback."
        return "\n".join(f"- {item}" for item in items)

    return (
        "Summary generated using local fallback (Gemini quota limit reached).\n\n"
        "Overview\n"
        f"{_section_lines(overview)}\n\n"
        "Key Decisions\n"
        f"{_section_lines(decisions)}\n\n"
        "Action Items\n"
        f"{_section_lines(actions)}\n\n"
        "Risks\n"
        f"{_section_lines(risks)}"
    )


def _single_pass_summary(text: str, model_name: str | None = None) -> str:
    llm = _build_llm(model_name)
    prompt = ChatPromptTemplate.from_template(
        "You are a precise analyst. Summarize the document in concise bullet-style prose with sections: "
        "Overview, Key Decisions, Action Items, and Risks.\n\nDocument:\n{text}"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text})


def _map_reduce_summary(text: str, model_name: str | None = None) -> str:
    llm = _build_llm(model_name)
    map_prompt = ChatPromptTemplate.from_template(
        "Summarize this chunk with focus on factual points, decisions, and actions.\n\nChunk:\n{chunk}"
    )
    reduce_prompt = ChatPromptTemplate.from_template(
        "Combine the partial summaries into a final coherent summary with sections: "
        "Overview, Key Decisions, Action Items, and Risks.\n\nPartial summaries:\n{partials}"
    )

    map_chain = map_prompt | llm | StrOutputParser()
    reduce_chain = reduce_prompt | llm | StrOutputParser()

    chunk_summaries = [map_chain.invoke({"chunk": chunk}) for chunk in _split_text(text)]
    return reduce_chain.invoke({"partials": "\n\n".join(chunk_summaries)})


def summarize_document(text: str, model_name: str | None = None) -> str:
    """Summarize a document using direct or map-reduce LCEL strategy."""
    if not text.strip():
        return "No text available for summarization."

    try:
        if len(text) < 12000:
            return _single_pass_summary(text, model_name=model_name)
        return _map_reduce_summary(text, model_name=model_name)
    except Exception as exc:
        if _is_quota_error(exc):
            return _local_fallback_summary(text)
        return (
            "Summary generation is currently unavailable. "
            f"Model error: {exc}"
        )
