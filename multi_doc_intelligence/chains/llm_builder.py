"""
LLM factory using the *already-installed* google-generativeai package.

No extra dependencies are required beyond what requirements.txt already pins.
Builds a proper BaseChatModel so it works seamlessly with ChatPromptTemplate
and LCEL chains (|  pipe operator).

Fallback order:
  1. GeminiChatModel  – requires a valid GOOGLE_API_KEY (starts with AIza…)
  2. ChatOllama       – local model (provider="ollama")
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, List, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from config import DEFAULT_OLLAMA_MODEL, GOOGLE_API_KEY, GEMINI_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _messages_to_text(messages: List[BaseMessage]) -> str:
    """Flatten a list of LangChain messages into a single prompt string."""
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            parts.append(f"System: {msg.content}")
        elif isinstance(msg, HumanMessage):
            parts.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            parts.append(f"Assistant: {msg.content}")
        else:
            parts.append(str(msg.content))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Gemini chat model (uses google-generativeai, already installed)
# ---------------------------------------------------------------------------

class GeminiChatModel(BaseChatModel):
    """LangChain BaseChatModel wrapper around google-generativeai."""

    model: str = Field(default=GEMINI_MODEL)
    temperature: float = Field(default=0.0)
    api_key: str = Field(default="")

    @property
    def _llm_type(self) -> str:
        return "gemini-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        key = self.api_key.strip() or GOOGLE_API_KEY.strip()
        if not key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Add it to your .env file "
                "(it must start with 'AIza…'). "
                "Get one at https://aistudio.google.com/app/apikey"
            )

        try:
            import google.generativeai as genai  # already in requirements
            from google.generativeai import types as gtypes
        except ImportError as exc:
            raise RuntimeError(
                "google-generativeai is not installed. "
                "Run: pip install google-generativeai"
            ) from exc

        prompt = _messages_to_text(messages)

        genai.configure(api_key=key)
        gmodel = genai.GenerativeModel(
            model_name=self.model,
            generation_config=gtypes.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=2048,
            ),
        )
        response = gmodel.generate_content(prompt)
        text = getattr(response, "text", None) or ""
        if not text.strip():
            raise RuntimeError("Gemini returned an empty response.")

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    class Config:
        extra = "allow"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _is_gemini_model(name: str | None) -> bool:
    """Return True when name looks like a Gemini/Gemma model identifier."""
    if not name:
        return False
    n = name.lower()
    return n.startswith("gemini") or n.startswith("models/gemma") or n.startswith("gemma")


def build_llm(
    model_name: str | None = None,
    provider: str = "gemini",
    temperature: float = 0,
) -> Any:
    """
    Build and return a LangChain BaseChatModel instance.

    Args:
        model_name:  Model identifier. Uses config default when None.
        provider:    ``"gemini"`` (default) or ``"ollama"``.
        temperature: Sampling temperature.

    Returns:
        A LangChain BaseChatModel (GeminiChatModel or ChatOllama).

    Raises:
        RuntimeError: If the provider cannot be initialised.
    """
    if provider.lower() == "gemini":
        return GeminiChatModel(
            model=model_name or GEMINI_MODEL,
            api_key=GOOGLE_API_KEY,
            temperature=temperature,
        )

    # Ollama fallback — never pass a Gemini/Gemma model name to Ollama
    try:
        from langchain_ollama import ChatOllama  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "langchain-ollama is not installed. Run: pip install langchain-ollama"
        ) from exc

    ollama_model = DEFAULT_OLLAMA_MODEL
    if model_name and not _is_gemini_model(model_name):
        ollama_model = model_name

    return ChatOllama(
        model=ollama_model,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
    )
