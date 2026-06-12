from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FaithfulnessResult:
    faithful: bool
    confidence: float
    reason: str


def check_faithfulness(answer: str, context: str, model_name: str | None = None) -> FaithfulnessResult:
    """
    Score whether an answer is grounded in the provided context.

    Uses a fast lexical overlap heuristic — no LLM call is made.
    This keeps per-query API usage at exactly 1 call (the answer generation),
    which is critical for free-tier Groq rate limits and CPU-only Ollama speed.
    """
    if not answer.strip() or not context.strip():
        return FaithfulnessResult(faithful=False, confidence=0.0, reason="Missing answer or context.")

    # Meaningful tokens only (length > 4 filters out stop words like "the", "and")
    answer_tokens = {t.lower() for t in answer.split() if len(t) > 4}
    context_tokens = {t.lower() for t in context.split() if len(t) > 4}

    if not answer_tokens:
        return FaithfulnessResult(faithful=False, confidence=0.0, reason="Answer contains no substantive terms.")

    overlap = len(answer_tokens & context_tokens) / max(len(answer_tokens), 1)
    faithful = overlap >= 0.25
    confidence = round(min(max(overlap, 0.0), 1.0), 2)

    if confidence >= 0.6:
        reason = "Answer is well-grounded in the retrieved evidence."
    elif confidence >= 0.25:
        reason = "Answer is partially grounded in the retrieved evidence."
    else:
        reason = "Answer has low overlap with retrieved evidence — verify manually."

    return FaithfulnessResult(faithful=faithful, confidence=confidence, reason=reason)
