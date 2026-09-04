"""Configuration management for Veritas RAG using Pydantic Settings.

Centralizes paths, model configurations, chunking parameters,
and retrieval thresholds with environment variable overrides.
"""

from pathlib import Path
from typing import List, Optional
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General
    ENVIRONMENT: str = Field(default="development", description="Environment mode")
    LOG_LEVEL: str = Field(default="INFO", description="Logging verbosity")

    # Backend selections
    EMBEDDING_BACKEND: str = Field(
        default="tfidf",
        description="Embedding backend: 'tfidf' (offline) or 'openai'",
    )
    LLM_PROVIDER: str = Field(
        default="mock",
        description="LLM Provider: 'mock' (offline), 'openai', or 'anthropic'",
    )

    # Models
    OPENAI_MODEL_NAME: str = Field(
        default="your-openai-model-name-here",
        description="OpenAI LLM model identifier",
    )
    OPENAI_EMBEDDING_MODEL_NAME: str = Field(
        default="your-openai-embedding-model-name-here",
        description="OpenAI text embedding model identifier",
    )
    ANTHROPIC_MODEL_NAME: str = Field(
        default="your-anthropic-model-name-here",
        description="Anthropic LLM model identifier",
    )

    # API Keys
    OPENAI_API_KEY: Optional[str] = Field(
        default=None,
        description="OpenAI API Key for embeddings and/or generation",
    )
    ANTHROPIC_API_KEY: Optional[str] = Field(
        default=None,
        description="Anthropic API Key for generation",
    )

    # Chunking
    CHUNK_SIZE_TARGET: int = Field(
        default=400,
        description="Target chunk size in estimated tokens",
    )
    CHUNK_OVERLAP: int = Field(
        default=60,
        description="Trailing sentence overlap in estimated tokens",
    )

    # Retrieval & RRF
    RETRIEVAL_DENSE_TOP_K: int = Field(
        default=10,
        description="Number of candidate chunks from dense retrieval",
    )
    RETRIEVAL_BM25_TOP_K: int = Field(
        default=10,
        description="Number of candidate chunks from BM25 sparse retrieval",
    )
    RETRIEVAL_FINAL_TOP_K: int = Field(
        default=5,
        description="Final number of top-k passages passed to generation",
    )
    RRF_K: int = Field(
        default=60,
        description="Reciprocal Rank Fusion smoothing constant k",
    )

    # Storage Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_RAW_DIR: Path = Field(default=Path("data/raw"))
    DATA_PROCESSED_DIR: Path = Field(default=Path("data/processed"))
    CHROMA_PERSIST_DIR: Path = Field(default=Path("data/processed/chroma"))
    BM25_INDEX_PATH: Path = Field(default=Path("data/processed/bm25_index.pkl"))
    TFIDF_MODEL_PATH: Path = Field(default=Path("data/processed/tfidf_vectorizer.pkl"))
    CHUNKS_METADATA_PATH: Path = Field(
        default=Path("data/processed/chunks_metadata.jsonl")
    )

    # CORS
    CORS_ALLOWED_ORIGINS: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        description="Comma-separated list of allowed browser origins",
    )

    @computed_field
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse comma-separated CORS origins into a clean, explicit list."""
        origins = {"http://localhost:3000", "http://127.0.0.1:3000"}
        if self.CORS_ALLOWED_ORIGINS:
            for item in self.CORS_ALLOWED_ORIGINS.split(","):
                clean = item.strip()
                if clean and clean != "*":  # Never allow wildcard origin with credentials
                    origins.add(clean)
        return sorted(list(origins))

    def resolve_path(self, path: Path) -> Path:
        """Resolve relative paths relative to BASE_DIR."""
        if path.is_absolute():
            return path
        return (self.BASE_DIR / path).resolve()

    def get_raw_dir(self) -> Path:
        p = self.resolve_path(self.DATA_RAW_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_processed_dir(self) -> Path:
        p = self.resolve_path(self.DATA_PROCESSED_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_chroma_dir(self) -> Path:
        p = self.resolve_path(self.CHROMA_PERSIST_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_bm25_path(self) -> Path:
        p = self.resolve_path(self.BM25_INDEX_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def get_tfidf_path(self) -> Path:
        p = self.resolve_path(self.TFIDF_MODEL_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def get_chunks_metadata_path(self) -> Path:
        p = self.resolve_path(self.CHUNKS_METADATA_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
