"""Retrieval evaluation: BASELINE (no reranking) vs RERANKED, side by side.

Runs a small, manually curated, page-verified question set against two
retrievers and reports Recall@1/@3/@5 for each:

  - BASELINE: app.rag.retrieve_chunks — plain Chroma similarity search,
    no reranking. This pass is unchanged from before reranking existed,
    so it remains a stable reference point.
  - RERANKED: app.rag.retrieve_chunks_reranked — the same size candidate
    pool from Chroma, reordered by a local cross-encoder (app.reranker).

This is a retrieval-only evaluation: it never calls the LLM (Gemini), and
it does not modify embeddings, Chroma, or FastAPI behavior.

Run from the project root (after scripts/build_index.py):
    python scripts/evaluate_retrieval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from app.rag import retrieve_chunks, retrieve_chunks_reranked
from app.reranker import get_reranker
from app.vector_store import load_vector_store

EVAL_DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_questions.json"
MAX_K = 5
RERANK_CANDIDATE_K = 10
RECALL_CUTOFFS = (1, 3, 5)


def load_eval_dataset() -> list[dict]:
    with open(EVAL_DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def _score(question: str, relevant_pages: set, retrieved_pages: list) -> dict:
    first_relevant_rank = None
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in relevant_pages:
            first_relevant_rank = rank
            break

    recall_at = {}
    for cutoff in RECALL_CUTOFFS:
        hit = any(page in relevant_pages for page in retrieved_pages[:cutoff])
        recall_at[cutoff] = 1.0 if hit else 0.0

    return {
        "question": question,
        "relevant_pages": sorted(relevant_pages),
        "retrieved_pages": retrieved_pages,
        "per_page_found": {page: (page in retrieved_pages) for page in sorted(relevant_pages)},
        "first_relevant_rank": first_relevant_rank,
        "recall_at": recall_at,
    }


def evaluate_question_baseline(question_item: dict, store) -> dict:
    question = question_item["question"]
    relevant_pages = set(question_item["relevant_pages"])
    chunks = retrieve_chunks(question, store=store, k=MAX_K)
    retrieved_pages = [chunk.metadata.get("page_number") for chunk in chunks]
    return _score(question, relevant_pages, retrieved_pages)


def evaluate_question_reranked(question_item: dict, store, reranker) -> dict:
    question = question_item["question"]
    relevant_pages = set(question_item["relevant_pages"])
    chunks = retrieve_chunks_reranked(
        question, store=store, candidate_k=RERANK_CANDIDATE_K, top_n=MAX_K, reranker=reranker
    )
    retrieved_pages = [chunk.metadata.get("page_number") for chunk in chunks]
    return _score(question, relevant_pages, retrieved_pages)


def print_question_result(index: int, result: dict) -> None:
    print("=" * 80)
    print(f"Q{index}: {result['question']}")
    print(f"  Expected relevant page(s): {result['relevant_pages']}")
    print(f"  Retrieved pages (rank order, top-{MAX_K}): {result['retrieved_pages']}")
    for page, found in result["per_page_found"].items():
        print(f"    page {page} retrieved: {found}")
    rank = result["first_relevant_rank"]
    print(f"  Rank of first relevant page: {rank if rank is not None else 'not found in top-' + str(MAX_K)}")
    for cutoff in RECALL_CUTOFFS:
        print(f"  Recall@{cutoff}: {result['recall_at'][cutoff]:.0f}")
    print()


def aggregate_recall(results: list[dict]) -> dict:
    n = len(results)
    return {cutoff: sum(r["recall_at"][cutoff] for r in results) / n for cutoff in RECALL_CUTOFFS}


def print_aggregate(label: str, results: list[dict]) -> None:
    print("=" * 80)
    print(f"{label} — Aggregate Retrieval Metrics")
    print("=" * 80)
    n = len(results)
    agg = aggregate_recall(results)
    for cutoff in RECALL_CUTOFFS:
        hits = sum(r["recall_at"][cutoff] for r in results)
        print(f"  Aggregate Recall@{cutoff}: {agg[cutoff]:.2f}  ({hits:.0f}/{n})")
    print()


def run_pass(label: str, dataset: list[dict], evaluate_fn) -> list[dict]:
    print("#" * 80)
    print(f"# {label}")
    print("#" * 80)
    print()
    results = []
    for i, item in enumerate(dataset, start=1):
        result = evaluate_fn(item)
        print_question_result(i, result)
        results.append(result)
    print_aggregate(label, results)
    return results


def main() -> None:
    dataset = load_eval_dataset()
    store = load_vector_store()
    reranker = get_reranker()

    print(f"Dataset: {EVAL_DATASET_PATH.name} ({len(dataset)} questions)")
    print()

    baseline_results = run_pass(
        "BASELINE (no reranking)", dataset, lambda item: evaluate_question_baseline(item, store)
    )
    reranked_results = run_pass(
        "RERANKED (cross-encoder)",
        dataset,
        lambda item: evaluate_question_reranked(item, store, reranker),
    )

    baseline_agg = aggregate_recall(baseline_results)
    reranked_agg = aggregate_recall(reranked_results)

    print("=" * 80)
    print("COMPARISON: BASELINE vs RERANKED")
    print("=" * 80)
    for cutoff in RECALL_CUTOFFS:
        print(
            f"  Recall@{cutoff}: baseline={baseline_agg[cutoff]:.2f}  "
            f"reranked={reranked_agg[cutoff]:.2f}  "
            f"delta={reranked_agg[cutoff] - baseline_agg[cutoff]:+.2f}"
        )


if __name__ == "__main__":
    main()
