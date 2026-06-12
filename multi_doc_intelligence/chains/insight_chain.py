from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from chains.llm_builder import build_llm


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "resource_exhausted" in msg
        or "quota" in msg
        or "429" in msg
        or "rate limit" in msg
    )


def _pick_by_keywords(text: str, keywords: list[str], limit: int = 5) -> list[str]:
    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]
    picked: list[str] = []
    for sentence in sentences:
        lower = sentence.lower()
        if any(keyword in lower for keyword in keywords):
            picked.append(sentence)
        if len(picked) >= limit:
            break
    return picked


def _fallback_extract_insights(text: str) -> dict[str, Any]:
    decisions = _pick_by_keywords(text, ["decide", "approved", "selected", "agreed", "final"], limit=5)
    action_items = _pick_by_keywords(text, ["action", "todo", "task", "next", "will", "owner", "deadline"], limit=5)
    key_findings = _pick_by_keywords(text, ["found", "result", "finding", "insight", "shows", "observed"], limit=5)

    if not key_findings:
        snippets = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]
        key_findings = snippets[:5]

    return {
        "decisions": decisions,
        "action_items": action_items,
        "key_findings": key_findings,
    }


def _clean_json_payload(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    return cleaned


def _normalize_output(payload: dict[str, Any]) -> dict[str, Any]:
    def to_list(key: str) -> list[str]:
        value = payload.get(key, [])
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    return {
        "decisions": to_list("decisions"),
        "action_items": to_list("action_items"),
        "key_findings": to_list("key_findings"),
    }


def extract_insights(text: str, model_name: str | None = None) -> dict[str, Any]:
    """Extract decisions, action items, and key findings as strict JSON."""
    if not text.strip():
        return {"decisions": [], "action_items": [], "key_findings": []}

    llm = build_llm(model_name=model_name, provider="groq", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "Extract structured insights from the document. Return only JSON with keys "
        "decisions, action_items, key_findings where each value is a list of strings. "
        "No markdown, no extra keys.\n\nDocument:\n{text}"
    )
    chain = prompt | llm | StrOutputParser()
    try:
        raw = chain.invoke({"text": text})
    except Exception as exc:
        if _is_quota_error(exc):
            fallback = _fallback_extract_insights(text)
            fallback["key_findings"] = [
                "Insights generated using local fallback (Groq rate limit reached)."
            ] + fallback["key_findings"]
            return fallback
        return {
            "decisions": [],
            "action_items": [],
            "key_findings": [f"Insight extraction unavailable: {exc}"],
        }

    try:
        parsed = json.loads(_clean_json_payload(raw))
    except json.JSONDecodeError:
        return _fallback_extract_insights(text)

    if not isinstance(parsed, dict):
        return _fallback_extract_insights(text)
    return _normalize_output(parsed)
