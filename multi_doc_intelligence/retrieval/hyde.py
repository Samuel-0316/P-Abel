from __future__ import annotations

from typing import Protocol


class LLMProtocol(Protocol):
    def invoke(self, prompt: str): ...


def _normalize_llm_output(output: object) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output.strip()
    content = getattr(output, "content", None)
    if isinstance(content, str):
        return content.strip()
    return str(output).strip()


def build_hyde_query(question: str, llm: LLMProtocol, *, use_hyde: bool = True) -> str:
    """
    Generate a hypothetical answer to improve retrieval embedding quality (HyDE).

    HyDE improves semantic search quality but costs one extra LLM call per query.
    Set use_hyde=False to skip the LLM call and return the raw question instead —
    recommended when using a slow local model (Ollama on CPU).
    """
    if not use_hyde:
        return question

    prompt = (
        "You are a search query expander. Your job is to rewrite the user's question into an optimized, "
        "comprehensive search query for a document retrieval system.\n\n"
        "RULES:\n"
        "1. Fix abbreviations and slang (e.g., 'edu' -> 'education', 'exp' -> 'experience').\n"
        "2. Add relevant synonyms or related terms (e.g., if asking for education, add 'university, degree, academic').\n"
        "3. Drop conversational filler like 'can please only just give me'.\n"
        "4. DO NOT write a full sentence. Return ONLY the expanded keywords and phrases.\n\n"
        f"User Question: {question}"
    )
    try:
        hypothetical = _normalize_llm_output(llm.invoke(prompt))
        return hypothetical or question
    except Exception:
        return question
