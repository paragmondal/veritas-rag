"""FastAPI service for Veritas Enterprise Hybrid RAG.

Exposes:
- GET /health: Status and index readiness probe
- POST /query: Hybrid retrieval and grounded generation with citation deduplication
- POST /reindex: Trigger synchronous corpus re-indexing

Enforces:
- Strict CORS with explicit allowed origins (never wildcard with credentials)
- Graceful 503 response if index artifacts are missing
"""

import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings
from src.rag_pipeline import RAGPipeline, PipelineResult
from src.embed_index import build_indexes

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger("veritas.api")

app = FastAPI(
    title="Veritas",
    description="Enterprise Hybrid RAG System over Corporate Filings",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Strict Least-Privilege CORS Configuration
# ---------------------------------------------------------------------------
# CRITICAL: Per CORS specification, allow_credentials=True MUST NOT be combined
# with allow_origins=["*"]. We supply explicit origins (localhost, 127.0.0.1, and env var).
allowed_origins = settings.cors_origins_list
logger.info(f"Configuring CORS with explicit origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question to answer")
    top_k: int = Field(
        default=5, ge=1, le=20, description="Final number of passages for generation"
    )
    embedding_backend: Optional[str] = Field(
        default=None, description="Optional override: 'tfidf' or 'openai'"
    )
    provider: Optional[str] = Field(
        default=None, description="Optional override: 'mock', 'openai', or 'anthropic'"
    )


class CitationResponse(BaseModel):
    source: str
    page: int
    score: float
    excerpt: str


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    source: str
    page: int
    score: float
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[CitationResponse]
    retrieved_chunks: List[RetrievedChunkResponse]
    latency_seconds: float


class HealthResponse(BaseModel):
    status: str
    indexes_ready: bool
    chunks_count: int
    embedding_backend: str
    llm_provider: str


def check_indexes_ready() -> bool:
    """Check if all required persistent index files exist."""
    return (
        settings.get_bm25_path().exists()
        and settings.get_chunks_metadata_path().exists()
        and settings.get_chroma_dir().exists()
    )


def count_indexed_chunks() -> int:
    """Count lines in chunks_metadata.jsonl."""
    meta_path = settings.get_chunks_metadata_path()
    if not meta_path.exists():
        return 0
    with open(meta_path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check returning readiness status and configuration."""
    ready = check_indexes_ready()
    return HealthResponse(
        status="healthy",
        indexes_ready=ready,
        chunks_count=count_indexed_chunks(),
        embedding_backend=settings.EMBEDDING_BACKEND,
        llm_provider=settings.LLM_PROVIDER,
    )


@app.post("/reindex")
def reindex_corpus():
    """Rebuild Chroma, BM25, and metadata indexes from raw files."""
    try:
        count = build_indexes()
        return {
            "status": "success",
            "message": f"Successfully reindexed {count} document chunks.",
            "chunks_count": count,
        }
    except Exception as e:
        logger.error(f"Reindex failure: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reindexing failed: {str(e)}",
        )


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """Execute hybrid retrieval and grounded generation with citation deduplication."""
    if not check_indexes_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Veritas indexes have not been built yet. Please call POST /reindex or run `python -m src.embed_index` first.",
        )

    try:
        pipeline = RAGPipeline(
            embedding_backend=request.embedding_backend,
            llm_provider=request.provider,
        )
        result: PipelineResult = pipeline.answer_query(
            query=request.question,
            top_k=request.top_k,
            provider=request.provider,
        )

        return QueryResponse(
            question=result.query,
            answer=result.answer,
            citations=[
                CitationResponse(
                    source=c.source,
                    page=c.page,
                    score=c.score,
                    excerpt=c.excerpt,
                )
                for c in result.citations
            ],
            retrieved_chunks=[
                RetrievedChunkResponse(
                    chunk_id=c.chunk_id,
                    source=c.source,
                    page=c.page,
                    score=c.score,
                    text=c.text,
                )
                for c in result.retrieved_chunks
            ],
            latency_seconds=result.latency_seconds,
        )
    except Exception as e:
        logger.error(f"Error handling query '{request.question}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution error: {str(e)}",
        )
