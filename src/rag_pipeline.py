"""End-to-end RAG pipeline orchestrator for Veritas.

Unifies retrieval, citation deduplication, and generation into a single
entrypoint shared across the FastAPI service, CLI, and evaluation harness.
"""

import logging
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from src.config import settings
from src.retrieval import HybridRetriever, RetrievedChunk
from src.generate import generate_answer

logger = logging.getLogger("veritas.rag_pipeline")


@dataclass
class Citation:
    """User-facing citation referencing a specific source and page with an excerpt."""
    source: str
    page: int
    score: float
    excerpt: str


@dataclass
class PipelineResult:
    """Complete result returned by the RAG pipeline."""
    query: str
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[RetrievedChunk]
    latency_seconds: float


def deduplicate_citations(retrieved_chunks: List[RetrievedChunk]) -> List[Citation]:
    """Deduplicate citations by (source, page) preserving highest-scoring occurrences.

    CRITICAL ARCHITECTURAL REQUIREMENT:
    Chunks are sorted in score-descending order. Only the first (highest scoring)
    occurrence for each unique (source, page) pair is kept in the user-facing citation
    list. All chunks continue to be provided to the LLM for full generation context.
    """
    sorted_chunks = sorted(retrieved_chunks, key=lambda c: c.score, reverse=True)
    seen_locations: Set[Tuple[str, int]] = set()
    deduped_citations: List[Citation] = []

    for chunk in sorted_chunks:
        loc = (chunk.source, chunk.page)
        if loc not in seen_locations:
            seen_locations.add(loc)
            # Create a concise excerpt (first 180 chars)
            clean_text = " ".join(chunk.text.split())
            excerpt = clean_text[:180] + "..." if len(clean_text) > 180 else clean_text
            deduped_citations.append(
                Citation(
                    source=chunk.source,
                    page=chunk.page,
                    score=chunk.score,
                    excerpt=excerpt,
                )
            )

    return deduped_citations


class RAGPipeline:
    """Singleton-friendly pipeline orchestrator for hybrid retrieval and generation."""

    def __init__(
        self,
        embedding_backend: Optional[str] = None,
        llm_provider: Optional[str] = None,
    ):
        self.embedding_backend = embedding_backend or settings.EMBEDDING_BACKEND
        self.llm_provider = llm_provider or settings.LLM_PROVIDER
        self.retriever = HybridRetriever(backend_name=self.embedding_backend)

    def answer_query(
        self,
        query: str,
        top_k: int = settings.RETRIEVAL_FINAL_TOP_K,
        provider: Optional[str] = None,
    ) -> PipelineResult:
        """Run end-to-end query answering: retrieval -> deduplication -> generation."""
        start_time = time.perf_counter()

        # 1. Hybrid Retrieval with RRF
        retrieved_chunks = self.retriever.retrieve(
            query=query,
            final_top_k=top_k,
        )

        # 2. Citation Deduplication (User-facing only)
        citations = deduplicate_citations(retrieved_chunks)

        # 3. Grounded Generation (receives all retrieved chunks for complete context)
        chosen_provider = provider or self.llm_provider
        answer = generate_answer(
            query=query,
            retrieved_chunks=retrieved_chunks,
            provider=chosen_provider,
        )

        latency = round(time.perf_counter() - start_time, 4)

        return PipelineResult(
            query=query,
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            latency_seconds=latency,
        )


def answer_query(
    query: str,
    top_k: int = settings.RETRIEVAL_FINAL_TOP_K,
    embedding_backend: Optional[str] = None,
    llm_provider: Optional[str] = None,
) -> PipelineResult:
    """Convenience function instantiating a pipeline and answering a query."""
    pipeline = RAGPipeline(
        embedding_backend=embedding_backend,
        llm_provider=llm_provider,
    )
    return pipeline.answer_query(query=query, top_k=top_k)


if __name__ == "__main__":
    test_query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What was Acme's revenue in 2024 and 2025?"
    )
    print(f"Executing Veritas RAG Pipeline for query: {test_query}\n")
    result = answer_query(test_query)

    print("=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(result.answer)
    print("\n" + "=" * 60)
    print(f"CITATIONS ({len(result.citations)} deduplicated):")
    print("=" * 60)
    for c in result.citations:
        print(f"- {c.source} (page {c.page}) [RRF score: {c.score}]")
        print(f"  Excerpt: {c.excerpt}\n")
    print(f"Latency: {result.latency_seconds}s")
