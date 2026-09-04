"""Embedding backends for Veritas RAG.

Provides an abstract interface with two concrete implementations:
1. TfidfEmbeddingBackend — scikit-learn TF-IDF, entirely offline, zero network dependencies.
2. OpenAIEmbeddingBackend — OpenAI embeddings API for dense semantic representation.
"""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.config import settings


class EmbeddingBackend(ABC):
    """Abstract interface for document and query embedding backends."""

    @abstractmethod
    def fit(self, texts: List[str]) -> None:
        """Fit the embedding model on a corpus of texts (if applicable)."""
        pass

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a list of document chunk texts."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Compute embedding for a single search query."""
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Persist the model state to disk."""
        pass

    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        """Load model state from disk."""
        pass


class TfidfEmbeddingBackend(EmbeddingBackend):
    """TF-IDF vectorizer backend using scikit-learn.

    Generates L2-normalized dense representations from sparse TF-IDF vectors.
    Requires zero external network or API calls, making it ideal for offline,
    air-gapped, or development use.
    """

    def __init__(self, max_features: int = 1536):
        self.max_features = max_features
        self.vectorizer: Optional[TfidfVectorizer] = None

    def fit(self, texts: List[str]) -> None:
        """Fit the TF-IDF vectorizer on the provided texts."""
        if not texts:
            raise ValueError("Cannot fit TF-IDF vectorizer on empty text collection.")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            sublinear_tf=True,
            stop_words="english",
            token_pattern=r"(?u)\b\w+[\w\.\$\%]*\b",
        )
        self.vectorizer.fit(texts)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Transform texts into L2-normalized float lists."""
        if self.vectorizer is None:
            raise RuntimeError(
                "TfidfEmbeddingBackend has not been fitted or loaded yet."
            )
        if not texts:
            return []

        sparse_matrix = self.vectorizer.transform(texts)
        norm_matrix = normalize(sparse_matrix, norm="l2")
        dense_matrix = norm_matrix.toarray().astype(float)
        return dense_matrix.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query."""
        res = self.embed([query])
        if not res:
            dim = len(self.vectorizer.get_feature_names_out()) if self.vectorizer else 1
            return [0.0] * dim
        return res[0]

    def save(self, path: Union[str, Path]) -> None:
        """Persist vectorizer to disk via pickle."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "max_features": self.max_features}, f)

    def load(self, path: Union[str, Path]) -> None:
        """Load vectorizer from disk."""
        target_path = Path(path)
        if not target_path.exists():
            raise FileNotFoundError(f"TF-IDF model file not found at: {target_path}")
        with open(target_path, "rb") as f:
            data = pickle.load(f)
            self.vectorizer = data["vectorizer"]
            self.max_features = data.get("max_features", self.max_features)


class OpenAIEmbeddingBackend(EmbeddingBackend):
    """OpenAI embedding backend using text-embedding-3-small or user-specified model."""

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        self.model_name = (
            model_name
            or (
                settings.OPENAI_EMBEDDING_MODEL_NAME
                if settings.OPENAI_EMBEDDING_MODEL_NAME != "your-openai-embedding-model-name-here"
                else "text-embedding-3-small"
            )
        )
        self.api_key = api_key or settings.OPENAI_API_KEY
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "OPENAI_API_KEY must be provided to use OpenAIEmbeddingBackend."
                )
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def fit(self, texts: List[str]) -> None:
        """Pretrained embeddings do not require fitting."""
        pass

    def embed(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings in batches via OpenAI API."""
        if not texts:
            return []
        client = self._get_client()
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = client.embeddings.create(model=self.model_name, input=batch)
            sorted_data = sorted(resp.data, key=lambda item: item.index)
            all_embeddings.extend([item.embedding for item in sorted_data])

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query."""
        client = self._get_client()
        resp = client.embeddings.create(model=self.model_name, input=[query])
        return resp.data[0].embedding

    def save(self, path: Union[str, Path]) -> None:
        """No local weights to persist for API backend."""
        pass

    def load(self, path: Union[str, Path]) -> None:
        """No local weights to load for API backend."""
        pass


def get_embedding_backend(backend_name: Optional[str] = None) -> EmbeddingBackend:
    """Factory to retrieve the specified embedding backend."""
    selected = (backend_name or settings.EMBEDDING_BACKEND).lower().strip()
    if selected == "tfidf":
        return TfidfEmbeddingBackend()
    elif selected == "openai":
        return OpenAIEmbeddingBackend()
    else:
        raise ValueError(
            f"Unknown embedding backend '{selected}'. Supported backends are 'tfidf' and 'openai'."
        )
