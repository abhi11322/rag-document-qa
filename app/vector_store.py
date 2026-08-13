"""Embedding generation and Chroma vector store persistence.

Builds a Chroma collection from chunked Documents (see app.ingestion) using
a local Sentence Transformers embedding model, and loads that collection
back for similarity search. No LLM API is used here — embeddings are
computed entirely locally.
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PERSIST_DIR = "data/chroma"
COLLECTION_NAME = "document_chunks"


def get_embedding_model(model_name: str | None = None) -> HuggingFaceEmbeddings:
    """Return the configured local embedding model.

    The model name is read from the EMBEDDING_MODEL environment variable
    (falling back to DEFAULT_EMBEDDING_MODEL) unless overridden explicitly,
    so the model can be changed without touching code.
    """
    model_name = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(model_name=model_name)


def resolve_persist_dir(persist_dir: str | Path | None = None) -> Path:
    """Resolve the Chroma persistence directory from an override or env var."""
    return Path(persist_dir or os.getenv("CHROMA_PERSIST_DIR", DEFAULT_PERSIST_DIR))


def build_vector_store(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings | None = None,
    persist_dir: str | Path | None = None,
) -> Chroma:
    """Embed chunks and build a Chroma vector store, persisted to disk.

    Each chunk's page_content and metadata (including page_number) are
    stored alongside its embedding, so retrieval results carry full source
    references. Chroma writes to persist_dir as documents are added, so no
    separate save step is required.
    """
    embeddings = embeddings or get_embedding_model()
    persist_dir = resolve_persist_dir(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_dir),
    )


def load_vector_store(
    persist_dir: str | Path | None = None,
    embeddings: HuggingFaceEmbeddings | None = None,
) -> Chroma:
    """Load a previously persisted Chroma vector store from disk.

    The same embedding model used to build the collection must be used to
    load it, so the query vectors land in the same embedding space.
    """
    persist_dir = resolve_persist_dir(persist_dir)
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"No Chroma database found at {persist_dir}. Build it first with scripts/build_index.py."
        )

    embeddings = embeddings or get_embedding_model()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
