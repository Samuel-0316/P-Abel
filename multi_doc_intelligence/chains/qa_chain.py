from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from chains.llm_builder import build_llm
from chains.hallucination import FaithfulnessResult, check_faithfulness
from config import MAX_CONTEXT_CHARS
from retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


def _classify_llm_error(exc: Exception, provider: str) -> str:
    """Return a concise, human-readable error string from a raw LLM exception."""
    msg = str(exc)
    low = msg.lower()
    if provider == "gemini":
        if "429" in msg or "quota" in low or "rate" in low:
            return (
                "Gemini free-tier quota exceeded. "
                "Wait a minute and retry, or enable billing at "
                "https://ai.google.dev/gemini-api/docs/rate-limits"
            )
        if "api_key" in low or "invalid" in low or "401" in msg:
            return "Gemini API key is invalid or not authorised. Check your GOOGLE_API_KEY in .env."
    if provider == "ollama":
        if "system memory" in low or "more memory" in low or "500" in msg:
            return (
                "Ollama model ran out of system RAM. "
                "Switch to a smaller model (e.g. phi3.5:latest) in your .env DEFAULT_OLLAMA_MODEL."
            )
        if "not found" in low or "404" in msg:
            return (
                "Ollama model not found locally. "
                "Run `ollama list` to see installed models and update DEFAULT_OLLAMA_MODEL in .env."
            )
    # Generic fallback — keep it short, no raw JSON blobs
    return msg[:200]


@dataclass
class QAResult:
    answer: str
    citations: list[dict[str, str]]
    hyde_query: str
    faithfulness: FaithfulnessResult
    error: str | None = None          # surfaced to UI when not None


def _format_context(docs: list, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Format retrieved parent sections into a context string.

    Parent sections are already bounded by PARENT_CHUNK_SIZE (~1500 chars each),
    so we don't truncate individual sections. We include as many sections as fit
    within max_chars total, preserving the most relevant ones (they arrive
    already ranked by the cross-encoder).
    """
    if not docs:
        return ""

    blocks: list[str] = []
    total_chars = 0

    for idx, doc in enumerate(docs, start=1):
        source = str(doc.metadata.get("source", "unknown"))
        page = str(doc.metadata.get("page", "n/a"))
        sheet = str(doc.metadata.get("sheet", "n/a"))
        cell_range = str(doc.metadata.get("cell_range", "n/a"))
        content = doc.page_content.strip()
        header = f"[{idx}] source={source} page={page} sheet={sheet} cell_range={cell_range}"
        block = f"{header}\n{content}"

        if total_chars + len(block) > max_chars and blocks:
            # Budget exhausted — stop adding sections but always include at least one
            break
        blocks.append(block)
        total_chars += len(block)

    return "\n\n".join(blocks)


def _extract_citations(docs: list) -> list[dict[str, str]]:
    citations: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for doc in docs:
        metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
        source = str(metadata.get("source", "unknown")).strip() or "unknown"

        raw_page = metadata.get("page")
        page = ""
        if raw_page is not None:
            raw_page_text = str(raw_page).strip()
            if raw_page_text and raw_page_text.lower() not in {"n/a", "none"}:
                try:
                    # PyMuPDF provides 0-based page numbers; convert to 1-based.
                    page = str(int(raw_page_text) + 1)
                except ValueError:
                    page = raw_page_text

        raw_sheet = str(metadata.get("sheet", "")).strip()
        sheet = "" if raw_sheet.lower() in {"", "n/a", "none"} else raw_sheet

        raw_cell_range = str(metadata.get("cell_range", "")).strip()
        cell_range = "" if raw_cell_range.lower() in {"", "n/a", "none"} else raw_cell_range

        raw_chunk_id = str(metadata.get("chunk_id", "")).strip()
        chunk_id = "" if raw_chunk_id.lower() in {"", "n/a", "none"} else raw_chunk_id

        dedupe_key = (source, page, sheet, cell_range)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        excerpt = " ".join(str(doc.page_content).split())[:180]
        citations.append(
            {
                "source": source,
                "page": page,
                "sheet": sheet,
                "cell_range": cell_range,
                "chunk_id": chunk_id,
                "excerpt": excerpt,
            }
        )

    return citations


THREAD_MEMORY: dict[str, list[tuple[str, str]]] = {}


def hydrate_thread_memory(thread_id: str, messages: list[dict[str, Any]]) -> None:
    """Rebuild per-thread QA memory from stored chat messages."""
    turns: list[tuple[str, str]] = []
    pending_user: str | None = None
    for message in messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            turns.append((pending_user, content))
            pending_user = None

    THREAD_MEMORY[thread_id] = turns


def clear_thread_memory(thread_id: str) -> None:
    """Remove per-thread QA memory for a deleted or reset thread."""
    THREAD_MEMORY.pop(thread_id, None)


def _memory_to_text(thread_id: str, max_turns: int = 3) -> str:
    """
    Return the last N conversation turns as plain text.

    Capped at 3 turns (reduced from 5) to keep prompt size small.
    """
    turns = THREAD_MEMORY.get(thread_id, [])[-max_turns:]
    if not turns:
        return ""

    lines: list[str] = []
    for idx, (q, a) in enumerate(turns, start=1):
        lines.append(f"Turn {idx} User: {q}")
        # Truncate long assistant answers in history to save tokens
        a_short = a[:300] + "…" if len(a) > 300 else a
        lines.append(f"Turn {idx} Assistant: {a_short}")
    return "\n".join(lines)


def _store_turn(thread_id: str, question: str, answer: str) -> None:
    history = THREAD_MEMORY.setdefault(thread_id, [])
    history.append((question, answer))


def _fallback_answer_from_context(question: str, docs: list) -> str:
    """Last-resort: return the most relevant raw chunks when no LLM is available."""
    if not docs:
        return "I do not have enough evidence in the indexed documents to answer this question right now."

    snippets: list[str] = []
    for doc in docs[:3]:
        source = str(doc.metadata.get("source", "unknown"))
        excerpt = " ".join(doc.page_content.split())[:280]
        snippets.append(f"- [{source}] {excerpt}")

    return (
        "⚠️ No LLM was reachable. Based on retrieved evidence, here are the most relevant excerpts:\n"
        + "\n".join(snippets)
        + "\n\nPlease check your API key / local model, then re-ask for a synthesized answer."
    )


def _invoke_gemini_chain(
    llm: Any,
    prompt: ChatPromptTemplate,
    inputs: dict,
) -> tuple[str, str | None]:
    """Run the Gemini LCEL chain. Returns (answer_text, error_or_None)."""
    chain = (
        RunnableLambda(lambda data: data)
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = chain.invoke(inputs)
    return answer, None


def _invoke_ollama_chain(
    prompt: ChatPromptTemplate,
    inputs: dict,
    model_name: str | None,
) -> tuple[str, str | None]:
    """Run the Ollama LCEL chain as fallback. Returns (answer_text, error_or_None)."""
    try:
        llm = build_llm(model_name=model_name, provider="ollama", temperature=0)
        chain = (
            RunnableLambda(lambda data: data)
            | prompt
            | llm
            | StrOutputParser()
        )
        answer = chain.invoke(inputs)
        return answer, None
    except Exception as exc:
        logger.warning("Ollama fallback also failed: %s", exc)
        return "", f"Ollama fallback failed: {_classify_llm_error(exc, 'ollama')}"


# ---------------------------------------------------------------------------
# Query-type classification
# ---------------------------------------------------------------------------

# Keywords that signal the user wants a complete, exhaustive answer covering
# all aspects of a topic — not just a quick lookup.
_COMPREHENSIVE_KEYWORDS = {
    "full", "complete", "all", "entire", "breakdown", "summarize", "summarise",
    "summary", "overview", "everything", "detail", "details", "detailed",
    "explain", "describe", "list all", "give me all", "what are all",
    "comprehensive", "total", "overall", "package", "structure",
}


def _classify_query_type(question: str) -> str:
    """
    Return 'comprehensive' or 'lookup' based on the question.

    'comprehensive' queries need wider retrieval and structured output.
    'lookup' queries need precision — a single fact or value.
    """
    lowered = question.lower()
    for kw in _COMPREHENSIVE_KEYWORDS:
        if kw in lowered:
            return "comprehensive"
    return "lookup"


def _build_prompt(
    query_type: str,
    history_block: str,
) -> ChatPromptTemplate:
    """
    Return the appropriate ChatPromptTemplate for the query type.

    Comprehensive prompt: instructs the LLM to be exhaustive, use markdown
    tables for financial/structured data, cover all sections, and note (not
    refuse) when specific values are referenced but not in the retrieved context.

    Lookup prompt: concise, grounded, cites the source section.
    """
    if query_type == "comprehensive":
        system = (
            "You are a thorough document intelligence assistant. "
            "Your job is to synthesize ALL relevant information from the retrieved context "
            "into a complete, well-structured answer.\n\n"
            "FORMATTING RULES:\n"
            "- Use ## section headers to organise the response\n"
            "- Present any financial, salary, or numeric data in markdown tables\n"
            "- Use bullet points for lists of items or benefits\n"
            "- Cover EVERY component mentioned in the context — do not skip any\n"
            "- If a document says 'see Annexure-I' or 'details provided separately', "
            "  explicitly note what is referenced but not in the retrieved text\n"
            "- DO NOT say 'insufficient information' for partial data — instead share "
            "  everything that IS available and flag only specific missing values\n"
            "- Always include a summary table or bullet at the end"
        )
    else:
        system = (
            "You are a precise document assistant. "
            "Answer the question using only the retrieved context. "
            "Be concise and cite which section of the document the answer comes from. "
            "If the specific fact is genuinely absent from the context, say so briefly."
        )

    return ChatPromptTemplate.from_template(
        f"{system}\n\n"
        f"{history_block}"
        "Question:\n{question}\n\n"
        "Retrieved Context:\n{context}\n"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask_question(question: str, thread_id: str, session_id: str, model_name: str | None = None) -> QAResult:
    """
    Conversational QA with per-thread memory and faithfulness scoring.

    LLM call budget per query:
    - Tier 1 (Gemini available):  1 call  — HyDE + answer in same provider
    - Tier 2 (Gemini quota hit):  1 call  — Ollama answer (HyDE skipped)
    - Tier 3 (both unavailable):  0 calls — raw chunk excerpts shown

    HyDE is enabled for Gemini (fast, cheap) and disabled for Ollama
    (too slow on CPU-only hardware without GPU acceleration).
    """
    error_msg: str | None = None
    gemini_available = False

    # --- Attempt Gemini build (validate key + model before retrieval) ---
    try:
        llm = build_llm(model_name=model_name, provider="gemini", temperature=0)
        gemini_available = True
    except RuntimeError as exc:
        logger.warning("Gemini LLM unavailable: %s", exc)
        llm = None
        error_msg = _classify_llm_error(exc, "gemini")

    # --- Retrieval ---
    # HyDE is only used when Gemini is available (fast LLM).
    # With Ollama, skip HyDE to avoid a slow extra inference call.
    try:
        retriever_llm = llm or build_llm(model_name=None, provider="ollama", temperature=0)
    except Exception:
        retriever_llm = None  # type: ignore

    # Classify query type to adapt retrieval width and prompt style.
    # Comprehensive queries ("full breakdown", "complete details") need more
    # sections and structured output; lookup queries need precision.
    query_type = _classify_query_type(question)
    is_comprehensive = query_type == "comprehensive"

    retriever = HybridRetriever(
        session_id=session_id,
        llm=retriever_llm,  # type: ignore[arg-type]
        use_hyde=gemini_available,
        # Wider net for comprehensive queries: more children retrieved,
        # more candidates after RRF, more parent sections to LLM.
        child_semantic_k=15 if is_comprehensive else 10,
        child_bm25_k=15 if is_comprehensive else 10,
        rrf_top_n=18 if is_comprehensive else 12,
        rerank_top_n=8 if is_comprehensive else 5,
    )
    retrieved = retriever.retrieve(question)

    history = _memory_to_text(thread_id)
    # Larger context budget for comprehensive queries
    ctx_budget = MAX_CONTEXT_CHARS * 2 if is_comprehensive else MAX_CONTEXT_CHARS
    context = _format_context(retrieved.documents, max_chars=ctx_budget)

    history_block = f"Conversation History:\n{history}\n\n" if history else ""
    prompt = _build_prompt(query_type, history_block)

    chain_inputs = {
        "question": question,
        "context": context,
    }

    answer = ""

    # Tier 1: Gemini
    if llm is not None:
        try:
            answer, chain_err = _invoke_gemini_chain(llm, prompt, chain_inputs)
            if chain_err:
                error_msg = (error_msg or "") + f" | {chain_err}"
        except Exception as exc:
            logger.warning("Gemini chain failed: %s", exc)
            error_msg = (error_msg or "") + f" | Gemini chain error: {_classify_llm_error(exc, 'gemini')}"

    # Tier 2: Ollama (HyDE already skipped in retriever when gemini_available=False)
    if not answer:
        answer, ollama_err = _invoke_ollama_chain(prompt, chain_inputs, model_name)
        if ollama_err:
            error_msg = (error_msg or "") + f" | {ollama_err}"
        elif answer:
            error_msg = (error_msg or "") + " | Answered via local Ollama model."

    # Tier 3: Raw chunks (last resort)
    if not answer:
        answer = _fallback_answer_from_context(question, retrieved.documents)

    _store_turn(thread_id, question, answer)
    faithfulness = check_faithfulness(answer=answer, context=context, model_name=model_name)

    return QAResult(
        answer=answer,
        citations=_extract_citations(retrieved.documents),
        hyde_query=retrieved.hyde_query,
        faithfulness=faithfulness,
        error=error_msg,
    )
