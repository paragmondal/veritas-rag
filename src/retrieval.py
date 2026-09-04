"""Hybrid retrieval module combining dense vector search and sparse BM25 search.

WHY HYBRID RETRIEVAL BEATS DENSE-ONLY:
Dense semantic embeddings (such as neural embeddings or low-dimensional projections)
map text into continuous geometric spaces. While dense models excel at understanding
synonyms, semantic intent, and thematic similarity, they notoriously suffer from the
"exact token matching" problem. In enterprise contexts—such as financial 10-K filings,
contracts, and regulatory charters—user queries frequently contain exact dollar thresholds
(e.g. '$185.0 million'), specific percentages ('14.5%'), statutory references ('Rule 10A-3'),
or exact proper nouns and ticker symbols.

Dense retrieval frequently fails to distinguish between '$185 million' and '$820 million'
because both occupy nearly identical semantic spaces (both are financial metrics in technology
filings). In contrast, BM25 (sparse lexical search) assigns high inverted-document-frequency
weights to rare, exact tokens, retrieving the exact passage containing the queried numbers.
By combining dense and BM25 scores via Reciprocal Rank Fusion (RRF), Veritas achieves the best
of both worlds: semantic understanding of broad queries and pinpoint accuracy on exact figures.

RECIPROCAL RANK FUSION (RRF):
Unlike weighted linear score combinations (which require fragile score normalization
and hyperparameter tuning across differing score distributions), RRF merges rankings purely
based on ordinal ranks:
    RRF_score(d) = sum_{m in {dense, bm25}} (1 / (k + rank_m(d)))
where k is a smoothing constant (standard default k = 60) and rank_m(d) is the 1-based rank.
"""

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import chromadb

from src.config import settings
from src.embeddings import get_embedding_backend, EmbeddingBackend
from src.embed_index import tokenize_for_bm25, CHROMA_COLLECTION_NAME

logger = logging.getLogger("veritas.retrieval")


@dataclass
class RetrievedChunk:
    """Represents a retrieved passage with scoring and ranking provenance."""
    chunk_id: str
    doc_id: str
    source: str
    page: int
    text: str
    score: float
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None


class HybridRetriever:
    """Orchestrates dense vector search and BM25 sparse search merged with RRF."""

    def __init__(
        self,
        backend_name: Optional[str] = None,
        chroma_dir: Optional[Path] = None,
        bm25_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        k_rrf: int = settings.RRF_K,
    ):
        self.backend_name = backend_name or settings.EMBEDDING_BACKEND
        self.chroma_dir = chroma_dir or settings.get_chroma_dir()
        self.bm25_path = bm25_path or settings.get_bm25_path()
        self.metadata_path = metadata_path or settings.get_chunks_metadata_path()
        self.k_rrf = k_rrf

        self._embedder: Optional[EmbeddingBackend] = None
        self._chroma_client: Optional[chromadb.PersistentClient] = None
        self._chroma_collection = None
        self._bm25_index = None
        self._bm25_chunk_ids: List[str] = []
        self._chunk_metadata_lookup: Dict[str, dict] = {}

        self._load_resources()

    def _load_resources(self) -> None:
        """Load Chroma collection, BM25 index, and chunk metadata."""
        # 1. Chunk metadata lookup
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Chunk metadata missing at {self.metadata_path}. Please run `python -m src.embed_index` first."
            )
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    self._chunk_metadata_lookup[rec["chunk_id"]] = rec

        # 2. Embedding backend
        self._embedder = get_embedding_backend(self.backend_name)
        if self.backend_name == "tfidf":
            tfidf_path = settings.get_tfidf_path()
            if not tfidf_path.exists():
                raise FileNotFoundError(
                    f"TF-IDF model missing at {tfidf_path}. Please run `python -m src.embed_index` first."
                )
            self._embedder.load(tfidf_path)

        # 3. ChromaDB collection
        self._chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))
        try:
            self._chroma_collection = self._chroma_client.get_collection(
                name=CHROMA_COLLECTION_NAME
            )
        except Exception as e:
            raise FileNotFoundError(
                f"Chroma collection '{CHROMA_COLLECTION_NAME}' not found in {self.chroma_dir}: {e}"
            )

        # 4. BM25 index
        if not self.bm25_path.exists():
            raise FileNotFoundError(
                f"BM25 index missing at {self.bm25_path}. Please run `python -m src.embed_index` first."
            )
        with open(self.bm25_path, "rb") as f:
            bm25_data = pickle.load(f)
            self._bm25_index = bm25_data["index"]
            self._bm25_chunk_ids = bm25_data["chunk_ids"]

    def retrieve_dense(self, query: str, top_k: int = settings.RETRIEVAL_DENSE_TOP_K) -> List[str]:
        """Perform dense vector retrieval via ChromaDB."""
        query_vector = self._embedder.embed_query(query)
        results = self._chroma_collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, len(self._chunk_metadata_lookup)),
            include=["documents", "metadatas"],
        )
        if results and "ids" in results and results["ids"]:
            return results["ids"][0]
        return []

    def retrieve_bm25(self, query: str, top_k: int = settings.RETRIEVAL_BM25_TOP_K) -> List[str]:
        """Perform sparse lexical retrieval via BM25."""
        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return []

        scores = self._bm25_index.get_scores(query_tokens)
        # Sort indices by score descending
        sorted_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True,
        )
        top_indices = sorted_indices[: min(top_k, len(self._bm25_chunk_ids))]
        # Only include documents with non-zero BM25 score
        filtered_ids = [
            self._bm25_chunk_ids[i] for i in top_indices if scores[i] > 0
        ]
        return filtered_ids

    def retrieve(
        self,
        query: str,
        dense_top_k: int = settings.RETRIEVAL_DENSE_TOP_K,
        bm25_top_k: int = settings.RETRIEVAL_BM25_TOP_K,
        final_top_k: int = settings.RETRIEVAL_FINAL_TOP_K,
    ) -> List[RetrievedChunk]:
        """Perform hybrid retrieval with Reciprocal Rank Fusion (RRF).

        Calculates RRF score:
            RRF(d) = sum_{method} (1 / (k + rank))
        """
        dense_ids = self.retrieve_dense(query, top_k=dense_top_k)
        bm25_ids = self.retrieve_bm25(query, top_k=bm25_top_k)

        rrf_scores: Dict[str, float] = {}
        dense_ranks: Dict[str, int] = {}
        bm25_ranks: Dict[str, int] = {}

        # Process dense ranks (1-based rank)
        for rank_idx, chunk_id in enumerate(dense_ids, start=1):
            dense_ranks[chunk_id] = rank_idx
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (
                1.0 / (self.k_rrf + rank_idx)
            )

        # Process BM25 ranks (1-based rank)
        for rank_idx, chunk_id in enumerate(bm25_ids, start=1):
            bm25_ranks[chunk_id] = rank_idx
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (
                1.0 / (self.k_rrf + rank_idx)
            )

        # Sort candidate chunks by fused RRF score descending
        sorted_candidates = sorted(
            rrf_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        final_candidates = sorted_candidates[:final_top_k]
        retrieved_chunks: List[RetrievedChunk] = []

        for chunk_id, fused_score in final_candidates:
            meta = self._chunk_metadata_lookup.get(chunk_id)
            if not meta:
                continue
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=meta["doc_id"],
                    source=meta["source"],
                    page=meta["page"],
                    text=meta["text"],
                    score=round(fused_score, 6),
                    dense_rank=dense_ranks.get(chunk_id),
                    bm25_rank=bm25_ranks.get(chunk_id),
                )
            )

        return retrieved_chunks
