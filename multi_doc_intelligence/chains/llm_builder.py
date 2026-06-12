"""
LLM factory — Groq as primary provider, Ollama as local fallback.

Groq (https://console.groq.com) offers a generous free tier with very fast
inference (~500 tok/s on LPU hardware).  langchain-groq returns a proper
BaseChatModel, so it integrates seamlessly with LCEL chains.

Fallback order:
  1. ChatGroq   — requires GROQ_API_KEY in .env
  2. ChatOllama — local model (provider="ollama")
"""

from __future__ import annotations

import logging
from typing import Any

from config import DEFAULT_OLLAMA_MODEL, GROQ_API_KEY, GROQ_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_cloud_model(name: str | None) -> bool:
    """
    Return True when the name looks like a cloud model identifier
    (Llama, Mixtral, Gemma via Groq) that should NOT be passed to Ollama.
    """
    if not name:
        return False
    n = name.lower()
    return (
        n.startswith("llama")
        or n.startswith("mixtral")
        or n.startswith("gemma")
        or n.startswith("whisper")
        or n.startswith("gemini")   # guard: don't accidentally send old Gemini names to Ollama
        or "-versatile" in n
        or "-instant" in n
        or "-preview" in n
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_llm(
    model_name: str | None = None,
    provider: str = "groq",
    temperature: float = 0,
) -> Any:
    """
    Build and return a LangChain BaseChatModel instance.

    Args:
        model_name:  Model identifier. Uses config default when None.
        provider:    ``"groq"`` (default) or ``"ollama"``.
        temperature: Sampling temperature.

    Returns:
        A LangChain BaseChatModel (ChatGroq or ChatOllama).

    Raises:
        RuntimeError: If the provider cannot be initialised.
    """
    if provider.lower() == "groq":
        key = GROQ_API_KEY.strip()
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file. "
                "Get a free key at https://console.groq.com"
            )
        try:
            from langchain_groq import ChatGroq  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "langchain-groq is not installed. "
                "Run: pip install langchain-groq"
            ) from exc

        return ChatGroq(
            model=model_name or GROQ_MODEL,
            api_key=key,
            temperature=temperature,
        )

    # Ollama fallback — never pass a cloud model name to Ollama
    try:
        from langchain_ollama import ChatOllama  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "langchain-ollama is not installed. Run: pip install langchain-ollama"
        ) from exc

    ollama_model = DEFAULT_OLLAMA_MODEL
    if model_name and not _is_cloud_model(model_name):
        ollama_model = model_name

    return ChatOllama(
        model=ollama_model,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
    )
