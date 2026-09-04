"""Comprehensive test suite for Veritas Enterprise RAG.

Validates:
- Dependency-free token estimation
- Sentence splitting integrity and abbreviation preservation
- Chunk bounds and sentence overlap retention
- Document ingestion and normalization
- Hybrid retrieval with RRF scoring
- End-to-end grounded generation with mock provider
- Unanswerable question refusal ("I don't know")
- Citation deduplication by (source, page)
"""

import math
import pytest
from pathlib import Path

from src.config import settings
from src.chunking import (
    estimate_token_count,
    split_into_sentences,
    chunk_document,
    chunk_documents,
    Chunk,
)
from src.ingest import RawDoc, normalize_whitespace, load_directory
from src.embeddings import TfidfEmbeddingBackend
from src.retrieval import HybridRetriever, RetrievedChunk
from src.rag_pipeline import answer_query, deduplicate_citations, RAGPipeline


# ---------------------------------------------------------------------------
# 1. Token Estimation Tests
# ---------------------------------------------------------------------------

def test_token_estimation():
    """Verify plain whitespace token estimation formula: int(len(text.split()) * 1.3)."""
    assert estimate_token_count("") == 0
    text = "Acme Technologies reported record cloud growth."
    words = text.split()
    expected = int(math.ceil(len(words) * 1.3))
    assert estimate_token_count(text) == expected
    assert estimate_token_count("SingleWord") == 2


# ---------------------------------------------------------------------------
# 2. Sentence Splitting & Integrity Tests
# ---------------------------------------------------------------------------

def test_sentence_integrity_abbreviations():
    """Ensure sentence splitting preserves corporate abbreviations and decimals without mid-sentence cuts."""
    sample = (
        "Acme Technologies Inc. filed Form 10-K with the SEC. "
        "Revenue grew by 22.0% reaching $5,185.0 million in FY2025. "
        "The board appointed Dr. Jane Smith as lead director."
    )
    sentences = split_into_sentences(sample)
    assert len(sentences) == 3
    # Check that "Inc." did not split into a separate sentence
    assert "Acme Technologies Inc. filed Form 10-K with the SEC." in sentences[0]
    # Check that decimals like 22.0% or $5,185.0 did not trigger false sentence breaks
    assert "22.0%" in sentences[1]
    assert "$5,185.0 million" in sentences[1]
    assert "Dr. Jane Smith" in sentences[2]


# ---------------------------------------------------------------------------
# 3. Chunk Size Bounds & Overlap Tests
# ---------------------------------------------------------------------------

def test_chunking_bounds_and_overlap():
    """Verify chunking adheres to sentence boundaries and carries overlap sentences."""
    long_doc_text = " ".join(
        [f"Sentence number {i} describes financial metric {i * 10}." for i in range(1, 40)]
    )
    doc = RawDoc(
        doc_id="test_doc_p1",
        source="test_doc.txt",
        page=1,
        text=long_doc_text,
    )
    chunks = chunk_document(doc, target_tokens=100, overlap_tokens=30)
    assert len(chunks) > 1

    # Check each chunk starts and ends on complete sentences
    for c in chunks:
        assert c.text.endswith(".")
        assert c.source == "test_doc.txt"
        assert c.page == 1

    # Check overlap: last sentence of chunk 0 should be present in chunk 1
    c0_sentences = split_into_sentences(chunks[0].text)
    c1_sentences = split_into_sentences(chunks[1].text)
    assert c0_sentences[-1] in c1_sentences


# ---------------------------------------------------------------------------
# 4. Ingestion Tests
# ---------------------------------------------------------------------------

def test_ingestion_sample_corpus():
    """Verify ingestion loads all sample files and creates structured RawDocs."""
    docs = load_directory(settings.get_raw_dir())
    assert len(docs) >= 4
    sources = {d.source for d in docs}
    assert "acme_10k_2024_financials.txt" in sources
    assert "acme_10k_2025_financials.txt" in sources
    assert "acme_10k_2025_risk_factors.txt" in sources
    assert "acme_governance_charter.txt" in sources

    for doc in docs:
        assert doc.page >= 1
        assert len(doc.text) > 0


def test_normalize_whitespace():
    """Test whitespace normalization without destroying paragraphs."""
    raw = "  Line 1   with   extra spaces. \r\n\r\n\n\n  Line 2 with newlines.  "
    clean = normalize_whitespace(raw)
    assert clean == "Line 1 with extra spaces.\n\nLine 2 with newlines."


# ---------------------------------------------------------------------------
# 5. Hybrid Retrieval & RRF Ranking Tests
# ---------------------------------------------------------------------------

def test_hybrid_retrieval():
    """Verify hybrid retrieval returns top-k chunks with non-zero RRF scores."""
    retriever = HybridRetriever(backend_name="tfidf")
    query = "What is Acme's revenue and gross profit in 2025?"
    results = retriever.retrieve(query, final_top_k=5)
    assert len(results) > 0
    assert len(results) <= 5
    for r in results:
        assert r.score > 0
        assert r.source is not None
        assert r.page >= 1
        assert len(r.text) > 0


# ---------------------------------------------------------------------------
# 6. End-to-End Pipeline & Grounded Refusal Tests
# ---------------------------------------------------------------------------

def test_pipeline_end_to_end():
    """Verify end-to-end pipeline execution produces answer with citations."""
    res = answer_query("What was Acme's revenue in 2025?")
    assert res.answer is not None
    assert len(res.answer) > 0
    assert len(res.citations) > 0
    # Verify citations appear in answer or citation chips
    assert any("acme_10k_2025_financials.txt" in c.source for c in res.citations)


def test_pipeline_unanswerable_refusal():
    """Verify unanswerable question triggers explicit 'I don't know' refusal."""
    res = answer_query("What is Acme's budget for lunar quantum computer exploration in 2039?")
    assert "i don't know" in res.answer.lower()


# ---------------------------------------------------------------------------
# 7. Citation Deduplication Tests (CRITICAL)
# ---------------------------------------------------------------------------

def test_citation_deduplication():
    """Verify citation deduplication produces unique (source, page) pairs preserving highest score."""
    mock_chunks = [
        RetrievedChunk(
            chunk_id="c1",
            doc_id="d1",
            source="acme_10k_2025.txt",
            page=1,
            text="First chunk text for page 1",
            score=0.035,
        ),
        RetrievedChunk(
            chunk_id="c2",
            doc_id="d1",
            source="acme_10k_2025.txt",
            page=1,  # Duplicate source and page!
            text="Second chunk text from same page 1 with lower score",
            score=0.025,
        ),
        RetrievedChunk(
            chunk_id="c3",
            doc_id="d2",
            source="acme_10k_2025.txt",
            page=2,  # Different page
            text="Chunk from page 2",
            score=0.030,
        ),
        RetrievedChunk(
            chunk_id="c4",
            doc_id="d3",
            source="acme_10k_2024.txt",
            page=1,  # Different file
            text="Chunk from 2024 file",
            score=0.028,
        ),
    ]

    deduped = deduplicate_citations(mock_chunks)
    # Total input chunks: 4, but unique (source, page) pairs: 3
    assert len(deduped) == 3

    # Ensure unique (source, page)
    pairs = [(c.source, c.page) for c in deduped]
    assert len(pairs) == len(set(pairs))

    # Ensure highest score was kept for the duplicate (acme_10k_2025.txt, 1)
    p1_citations = [c for c in deduped if c.source == "acme_10k_2025.txt" and c.page == 1]
    assert len(p1_citations) == 1
    assert p1_citations[0].score == 0.035
