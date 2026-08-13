"""Local cross-encoder reranking for retrieved chunks.

Chroma's similarity search uses a bi-encoder (all-MiniLM-L6-v2) that embeds
the query and each chunk independently — this struggles to discriminate
between chunks for many natural-language questions (see
scripts/evaluate_retrieval.py for the measured baseline). A cross-encoder
jointly encodes each (query, chunk) pair, giving sharper relevance scoring
over a small second-pass candidate set. Runs entirely locally — no API.
"""

from __future__ import annotations

import os

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_reranker(model_name: str | None = None) -> CrossEncoder:
    """Return the configured local cross-encoder reranker.

    Model name is read from the RERANKER_MODEL environment variable
    (falling back to DEFAULT_RERANKER_MODEL) unless overridden explicitly.
    """
    model_name = model_name or os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
    return CrossEncoder(model_name)


def rerank(
    query: str,
    chunks: list[Document],
    top_n: int,
    reranker: CrossEncoder | None = None,
) -> list[Document]:
    """Rerank candidate chunks against the query and return the top_n best."""
    if not chunks:
        return chunks

    reranker = reranker or get_reranker()
    pairs = [(query, chunk.page_content) for chunk in chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [chunk for chunk, _score in ranked[:top_n]]
