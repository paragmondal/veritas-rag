"""Sentence-aware chunking with sliding window and overlap.

DESIGN NOTE:
Token estimation uses a purely dependency-free formula: int(len(text.split()) * 1.3).
We deliberately avoid tiktoken, NLTK (word_tokenize), spaCy, or Hugging Face tokenizers.
Libraries like tiktoken and NLTK require network downloads of vocabulary or model files
at runtime (e.g. tiktoken's cl100k_base.tiktoken or NLTK's punkt tokenizer), which causes
unpredictable failures in air-gapped, containerized, or network-restricted enterprise
environments. A pure regex and whitespace-based approach provides zero-dependency,
deterministic execution everywhere.
"""

import math
import re
from dataclasses import dataclass
from typing import List

from src.config import settings
from src.ingest import RawDoc


@dataclass
class Chunk:
    """Represents a discrete text chunk for indexing and retrieval."""
    chunk_id: str
    doc_id: str
    source: str
    page: int
    chunk_index: int
    text: str
    token_count: int


def estimate_token_count(text: str) -> int:
    """Estimate token count using plain whitespace split without external NLP dependencies.

    Formula: len(text.split()) * 1.3.
    This provides an accurate approximation for English text without requiring
    downloads of vocabulary files like tiktoken or NLTK punkt.
    """
    words = text.split()
    if not words:
        return 0
    return max(1, int(math.ceil(len(words) * 1.3)))


# Regex to handle sentence splitting while preserving abbreviations, decimals, and defined terms.
# Protects common corporate/legal abbreviations: Inc., Corp., Co., Ltd., No., vs., e.g., i.e., U.S., Dr., etc.
_ABBREVIATIONS = (
    r"(?<!\bInc\.)"
    r"(?<!\bCorp\.)"
    r"(?<!\bCo\.)"
    r"(?<!\bLtd\.)"
    r"(?<!\bNo\.)"
    r"(?<!\bvs\.)"
    r"(?<!\be\.g\.)"
    r"(?<!\bi\.e\.)"
    r"(?<!\bU\.S\.)"
    r"(?<!\bJan\.)"
    r"(?<!\bFeb\.)"
    r"(?<!\bMar\.)"
    r"(?<!\bApr\.)"
    r"(?<!\bAug\.)"
    r"(?<!\bSept\.)"
    r"(?<!\bOct\.)"
    r"(?<!\bNov\.)"
    r"(?<!\bDec\.)"
    r"(?<!\bMr\.)"
    r"(?<!\bMrs\.)"
    r"(?<!\bMs\.)"
    r"(?<!\bDr\.)"
)

# Lookbehind for punctuation (. ! ?) followed by whitespace and a capital letter or number or start of line
_SENTENCE_SPLIT_REGEX = re.compile(
    rf"{_ABBREVIATIONS}(?<!\d)(?<=[.!?])\s+(?=[A-Z0-9\"'\(])"
)


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences while respecting abbreviations and numeric decimals.

    Uses standard library `re` only. Does not split mid-sentence or on decimal points.
    """
    text = text.strip()
    if not text:
        return []

    # First split on explicit double-newlines (paragraph breaks)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    all_sentences: List[str] = []

    for para in paragraphs:
        # If paragraph contains single newlines, treat them as line breaks within the paragraph
        clean_para = " ".join(para.split())
        # Split using regex
        raw_sentences = _SENTENCE_SPLIT_REGEX.split(clean_para)
        for s in raw_sentences:
            s_clean = s.strip()
            if s_clean:
                all_sentences.append(s_clean)

    return all_sentences


def chunk_document(
    doc: RawDoc,
    target_tokens: int = settings.CHUNK_SIZE_TARGET,
    overlap_tokens: int = settings.CHUNK_OVERLAP,
) -> List[Chunk]:
    """Chunk a RawDoc into overlapping, sentence-bounded Chunks.

    Guarantees:
    1. Never splits mid-sentence.
    2. Carries approximately `overlap_tokens` worth of trailing sentences into the next chunk.
    3. Retains document metadata (source, page).
    """
    sentences = split_into_sentences(doc.text)
    if not sentences:
        return []

    sentence_tokens = [estimate_token_count(s) for s in sentences]

    chunks: List[Chunk] = []
    start_idx = 0
    chunk_counter = 0

    while start_idx < len(sentences):
        current_sentences: List[str] = []
        current_tokens = 0
        end_idx = start_idx

        # Accumulate sentences until target_tokens is reached
        while end_idx < len(sentences):
            s_tok = sentence_tokens[end_idx]
            # Always take at least one sentence even if it exceeds target_tokens
            if current_sentences and (current_tokens + s_tok > target_tokens):
                break
            current_sentences.append(sentences[end_idx])
            current_tokens += s_tok
            end_idx += 1

        chunk_text = " ".join(current_sentences)
        chunk_id = f"{doc.source}_p{doc.page}_c{chunk_counter}"

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                source=doc.source,
                page=doc.page,
                chunk_index=chunk_counter,
                text=chunk_text,
                token_count=estimate_token_count(chunk_text),
            )
        )
        chunk_counter += 1

        # If we have reached or passed the end, break
        if end_idx >= len(sentences):
            break

        # Calculate overlap: backtrack from end_idx to include ~overlap_tokens
        overlap_accum = 0
        overlap_start = end_idx
        while overlap_start > start_idx:
            prev_tok = sentence_tokens[overlap_start - 1]
            if overlap_accum + prev_tok > overlap_tokens and overlap_accum > 0:
                break
            overlap_accum += prev_tok
            overlap_start -= 1

        # Advance start_idx ensuring forward progress
        if overlap_start <= start_idx:
            start_idx = end_idx
        else:
            start_idx = overlap_start

    return chunks


def chunk_documents(
    docs: List[RawDoc],
    target_tokens: int = settings.CHUNK_SIZE_TARGET,
    overlap_tokens: int = settings.CHUNK_OVERLAP,
) -> List[Chunk]:
    """Chunk a collection of RawDoc instances."""
    all_chunks: List[Chunk] = []
    for doc in docs:
        chunks = chunk_document(doc, target_tokens, overlap_tokens)
        all_chunks.extend(chunks)
    return all_chunks
