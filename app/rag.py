"""RAG pipeline: retrieval from Chroma + reranking + grounded LLM generation.

Wires together app.vector_store (retrieval), app.reranker (candidate
reranking), and app.llm_providers (generation) into a single ask(question)
call that returns a grounded answer plus deduplicated source page
references.
"""

from __future__ import annotations

import os
from pathlib import Path

from app.llm_providers import LLMProvider, get_llm_provider
from app.reranker import get_reranker, rerank
from app.vector_store import load_vector_store

DEFAULT_TOP_K = 4
DEFAULT_CANDIDATE_K = int(os.getenv("RETRIEVAL_CANDIDATE_K", "10"))
DEFAULT_RERANK_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", str(DEFAULT_TOP_K)))

PROMPT_TEMPLATE = """You are answering questions about a single document, using ONLY the context excerpts provided below.

Rules:
- Answer using ONLY information present in the context excerpts. Do not use outside knowledge.
- If the context does not contain enough information to answer the question, respond exactly with: "The information could not be found in the provided document."
- Do not fabricate citations, page numbers, or facts not present in the context.
- Be concise and directly answer the question.

Context excerpts:
{context}

Question: {question}

Answer:"""


def retrieve_chunks(question: str, store=None, k: int = DEFAULT_TOP_K):
    """Run similarity search against the persisted Chroma store (baseline, no reranking)."""
    store = store or load_vector_store()
    return store.similarity_search(question, k=k)


def retrieve_chunks_reranked(
    question: str,
    store=None,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    top_n: int = DEFAULT_RERANK_TOP_K,
    reranker=None,
):
    """Retrieve a larger candidate set from Chroma, then rerank down to top_n.

    Reuses retrieve_chunks() for the initial Chroma similarity search, so
    the baseline retrieval logic is never duplicated.
    """
    store = store or load_vector_store()
    reranker = reranker or get_reranker()

    candidates = retrieve_chunks(question, store=store, k=candidate_k)
    return rerank(question, candidates, top_n=top_n, reranker=reranker)


def build_prompt(question: str, chunks) -> str:
    """Construct a grounded prompt from retrieved chunks and the question."""
    context = "\n\n".join(
        f"[Page {chunk.metadata.get('page_number')}]\n{chunk.page_content}" for chunk in chunks
    )
    return PROMPT_TEMPLATE.format(context=context, question=question)


def extract_sources(chunks) -> list[dict]:
    """Deduplicate retrieved chunks into a page-level source list, preserving order."""
    sources: list[dict] = []
    seen_pages = set()
    for chunk in chunks:
        page_number = chunk.metadata.get("page_number")
        if page_number in seen_pages:
            continue
        seen_pages.add(page_number)
        source_path = chunk.metadata.get("source", "")
        sources.append({"page_number": page_number, "source": Path(source_path).name})
    return sources


def ask(
    question: str,
    store=None,
    provider: LLMProvider | None = None,
    reranker=None,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    top_n: int = DEFAULT_RERANK_TOP_K,
) -> dict:
    """Answer a question grounded in reranked, retrieved document context.

    Retrieves candidate_k candidates from Chroma, reranks them down to
    top_n with a local cross-encoder, then generates a grounded answer.
    Returns {"answer": str, "sources": [{"page_number": int, "source": str}, ...]}.
    """
    store = store or load_vector_store()
    provider = provider or get_llm_provider()

    chunks = retrieve_chunks_reranked(
        question, store=store, candidate_k=candidate_k, top_n=top_n, reranker=reranker
    )
    prompt = build_prompt(question, chunks)
    answer = provider.generate(prompt)

    return {"answer": answer, "sources": extract_sources(chunks)}
