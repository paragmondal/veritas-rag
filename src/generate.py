"""Answer generation module with strict grounding and multi-provider support.

Enforces:
1. Strict grounding in retrieved context only.
2. Mandatory citations [source, page] after every factual statement.
3. Explicit "I don't know" refusal when context lacks required facts.
4. Active flagging of contradictory or conflicting information.

Supported providers:
- 'mock': Zero-dependency deterministic extractive stitcher for offline testing.
- 'openai': OpenAI Chat Completions API.
- 'anthropic': Anthropic Messages API.
"""

import logging
import re
from typing import List, Optional

from src.config import settings
from src.retrieval import RetrievedChunk

logger = logging.getLogger("veritas.generate")

SYSTEM_PROMPT = """You are Veritas, an enterprise financial and corporate intelligence assistant.
Your answers must be rigorously grounded in the provided document context.

Ground Rules:
1. Grounding: Answer ONLY based on the facts directly stated in the provided context. Do NOT extrapolate or assume information not present in the text.
2. Citations: After EVERY factual statement or claim, immediately cite the source and page in brackets: [source_filename, page X] (e.g. [acme_10k_2025_financials.txt, page 1]).
3. Refusal: If the provided context does not contain sufficient facts to answer the question, explicitly state: "I don't know based on the provided documents." Do NOT fabricate an answer.
4. Contradictions: If different documents or pages present conflicting figures or statements, explicitly highlight the discrepancy rather than choosing one.
5. Conciseness: Present answers clearly and concisely.
"""


def build_context_string(chunks: List[RetrievedChunk]) -> str:
    """Format retrieved passages into a structured context block."""
    if not chunks:
        return "No documents retrieved."

    blocks: List[str] = []
    for i, c in enumerate(chunks, start=1):
        header = f"--- Document {i}: {c.source} (Page {c.page}) [Chunk ID: {c.chunk_id}] ---"
        blocks.append(f"{header}\n{c.text}\n")
    return "\n".join(blocks)


def mock_generate_answer(query: str, chunks: List[RetrievedChunk]) -> str:
    """Deterministic, zero-dependency extractive answer generator for offline testing.

    Performs query keyword extraction and sentence matching across retrieved chunks,
    appending [source, page] citations. If no relevant content is found, returns an
    explicit "I don't know" statement.
    """
    if not chunks:
        return "I don't know based on the provided documents. No relevant context was found."

    query_lower = query.lower()
    query_words = set(re.findall(r"\b[a-z0-9\$\%]+\b", query_lower))
    # Common stop words and generic query terms
    stop_words = {
        "what", "is", "the", "in", "of", "and", "to", "a", "for", "how", "did",
        "does", "was", "were", "are", "by", "from", "at", "on", "with", "acme",
        "which", "who", "when", "why", "their", "its", "or", "as", "an", "all",
        "any", "about", "describe", "explain", "detail", "state", "tell"
    }
    keywords = query_words - stop_words
    if not keywords:
        return "I don't know based on the provided documents."

    # Check which keywords actually exist anywhere in the retrieved chunks
    corpus_text = " ".join(c.text.lower() for c in chunks)
    missing_keywords = {kw for kw in keywords if kw not in corpus_text}

    # If critical specific query terms are completely missing from the retrieved context, refuse
    if missing_keywords and (len(missing_keywords) / len(keywords) >= 0.4 or len(keywords - missing_keywords) < 2):
        missing_str = ", ".join(sorted(missing_keywords))
        return f"I don't know based on the provided documents. The provided corporate filings do not contain information regarding: {missing_str}."

    matched_statements: List[str] = []
    seen_texts = set()

    for chunk in chunks:
        # Split chunk text into candidate sentences
        sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
        for s in sentences:
            s_clean = s.strip()
            if not s_clean or s_clean in seen_texts:
                continue

            s_lower = s_clean.lower()
            # Count keyword hits
            hit_count = sum(1 for kw in keywords if kw in s_lower)
            if hit_count > 0:
                seen_texts.add(s_clean)
                citation = f"[{chunk.source}, page {chunk.page}]"
                matched_statements.append((hit_count, f"{s_clean} {citation}"))

    # If no sentences matched keywords or query is entirely unanswerable
    if not matched_statements:
        return "I don't know based on the provided documents. The provided corpus does not contain information to answer this question."

    # Sort statements by relevance (keyword hit count descending)
    matched_statements.sort(key=lambda x: x[0], reverse=True)
    # Take top 4 most relevant sentences to form a coherent synthesized answer
    top_statements = [stmt for _, stmt in matched_statements[:4]]

    return "\n\n".join(top_statements)


def openai_generate_answer(
    query: str,
    chunks: List[RetrievedChunk],
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Generate answer using OpenAI Chat Completions API."""
    key = api_key or settings.OPENAI_API_KEY
    if not key:
        raise ValueError("OPENAI_API_KEY is required to generate answers using OpenAI.")

    model = model_name or (
        settings.OPENAI_MODEL_NAME
        if settings.OPENAI_MODEL_NAME != "your-openai-model-name-here"
        else "gpt-4o-mini"
    )

    from openai import OpenAI
    client = OpenAI(api_key=key)

    context_str = build_context_string(chunks)
    user_prompt = f"Context:\n{context_str}\n\nQuestion:\n{query}\n\nAnswer:"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def anthropic_generate_answer(
    query: str,
    chunks: List[RetrievedChunk],
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Generate answer using Anthropic Messages API."""
    key = api_key or settings.ANTHROPIC_API_KEY
    if not key:
        raise ValueError("ANTHROPIC_API_KEY is required to generate answers using Anthropic.")

    model = model_name or (
        settings.ANTHROPIC_MODEL_NAME
        if settings.ANTHROPIC_MODEL_NAME != "your-anthropic-model-name-here"
        else "claude-3-5-sonnet-latest"
    )

    import anthropic
    client = anthropic.Anthropic(api_key=key)

    context_str = build_context_string(chunks)
    user_prompt = f"Context:\n{context_str}\n\nQuestion:\n{query}\n\nAnswer:"

    response = client.messages.create(
        model=model,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=1024,
        temperature=0.0,
    )
    return response.content[0].text.strip()


def generate_answer(
    query: str,
    retrieved_chunks: List[RetrievedChunk],
    provider: Optional[str] = None,
) -> str:
    """Unified generation dispatcher supporting mock, openai, and anthropic."""
    selected_provider = (provider or settings.LLM_PROVIDER).lower().strip()

    if selected_provider == "mock":
        return mock_generate_answer(query, retrieved_chunks)
    elif selected_provider == "openai":
        return openai_generate_answer(query, retrieved_chunks)
    elif selected_provider == "anthropic":
        return anthropic_generate_answer(query, retrieved_chunks)
    else:
        raise ValueError(
            f"Unsupported LLM provider '{selected_provider}'. Options are 'mock', 'openai', 'anthropic'."
        )
