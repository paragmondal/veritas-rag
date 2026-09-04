"""Index builder for Veritas RAG.

Extracts raw documents, chunks them, computes dense embeddings,
populates ChromaDB with PRECOMPUTED embeddings (avoiding Chroma's model downloaders),
trains and serializes BM25Okapi sparse index, and saves chunk metadata to JSONL.
"""

import json
import logging
import pickle
import re
from pathlib import Path
from typing import List, Optional
import chromadb
from rank_bm25 import BM25Okapi

from src.config import settings
from src.ingest import load_directory
from src.chunking import chunk_documents, Chunk
from src.embeddings import get_embedding_backend, EmbeddingBackend

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger("veritas.embed_index")

CHROMA_COLLECTION_NAME = "veritas_chunks"


def tokenize_for_bm25(text: str) -> List[str]:
    """Simple, dependency-free tokenizer for BM25 indexing and querying.

    Extracts alphanumeric tokens, preserving dollar amounts and percentage symbols.
    """
    text_lower = text.lower()
    tokens = re.findall(r"\b[\w\$\%\.\-]+\b", text_lower)
    return tokens


def build_indexes(
    raw_dir: Optional[Path] = None,
    backend_name: Optional[str] = None,
) -> int:
    """Build ChromaDB and BM25 indexes from raw documents.

    Returns the number of indexed chunks.
    """
    target_raw_dir = raw_dir or settings.get_raw_dir()
    logger.info(f"Loading raw documents from {target_raw_dir}...")
    docs = load_directory(target_raw_dir)
    if not docs:
        raise ValueError(f"No documents found in {target_raw_dir}")

    logger.info(f"Loaded {len(docs)} document pages/sections. Chunking...")
    chunks = chunk_documents(docs)
    logger.info(f"Generated {len(chunks)} chunks.")

    # 1. Prepare and fit embedding backend
    selected_backend = backend_name or settings.EMBEDDING_BACKEND
    logger.info(f"Using embedding backend: {selected_backend}")
    embedder: EmbeddingBackend = get_embedding_backend(selected_backend)

    chunk_texts = [c.text for c in chunks]
    embedder.fit(chunk_texts)

    # Save TF-IDF model if applicable
    if selected_backend == "tfidf":
        tfidf_path = settings.get_tfidf_path()
        embedder.save(tfidf_path)
        logger.info(f"Saved TF-IDF model to {tfidf_path}")

    # Compute precomputed dense embeddings
    logger.info("Computing dense embeddings...")
    embeddings = embedder.embed(chunk_texts)
    logger.info(f"Computed {len(embeddings)} vectors with dimension {len(embeddings[0]) if embeddings else 0}.")

    # 2. Build ChromaDB collection with precomputed embeddings passed explicitly
    chroma_dir = settings.get_chroma_dir()
    logger.info(f"Initializing ChromaDB PersistentClient at {chroma_dir}...")
    client = chromadb.PersistentClient(path=str(chroma_dir))

    # Reset or get collection
    try:
        client.delete_collection(name=CHROMA_COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [c.chunk_id for c in chunks]
    metadatas = [
        {
            "doc_id": c.doc_id,
            "source": c.source,
            "page": c.page,
            "chunk_index": c.chunk_index,
            "token_count": c.token_count,
        }
        for c in chunks
    ]

    # Explicitly pass precomputed embeddings to prevent Chroma from attempting to download models
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunk_texts,
        metadatas=metadatas,
    )
    logger.info(f"Successfully added {len(chunks)} chunks to ChromaDB collection.")

    # 3. Build and serialize BM25Okapi index
    logger.info("Tokenizing corpus for BM25...")
    tokenized_corpus = [tokenize_for_bm25(c.text) for c in chunks]
    bm25_index = BM25Okapi(tokenized_corpus)

    bm25_path = settings.get_bm25_path()
    bm25_data = {
        "index": bm25_index,
        "chunk_ids": ids,
    }
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_data, f)
    logger.info(f"Saved BM25 index to {bm25_path}")

    # 4. Save chunk metadata and text to JSONL
    metadata_path = settings.get_chunks_metadata_path()
    with open(metadata_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "source": chunk.source,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "text": chunk.text,
            }
            f.write(json.dumps(record) + "\n")
    logger.info(f"Saved chunk metadata to {metadata_path}")

    logger.info("Indexing complete! All artifacts successfully built.")
    return len(chunks)


if __name__ == "__main__":
    count = build_indexes()
    print(f"Index build finished successfully: {count} chunks indexed.")
