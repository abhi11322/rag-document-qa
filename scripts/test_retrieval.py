"""Verification script for the embedding/vector-store stage.

Loads the persisted Chroma vector database and runs similarity search for a
set of questions that each target specific information in the source PDF,
printing the top matches' page numbers and a text preview for manual
inspection. Also runs one keyword-style sanity query. This script only
verifies that Chroma stores/retrieves chunks and preserves metadata — it is
not meant to optimize retrieval quality.

Run from the project root (after scripts/build_index.py):
    python scripts/test_retrieval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from app.vector_store import load_vector_store

TEST_QUERIES = [
    "What is Retrieval-Augmented Generation?",
    "What are the three steps of a typical RAG process?",
    "What are the main limitations of Naive RAG?",
    "How does RAG differ from fine-tuning?",
    "What is query expansion in RAG?",
]
SANITY_QUERY = "Query Expansion multiple queries"
TOP_K = 3
PREVIEW_CHARS = 220


def run_query(store, query: str) -> None:
    print("=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    results = store.similarity_search_with_score(query, k=TOP_K)

    for rank, (doc, score) in enumerate(results, start=1):
        preview = doc.page_content.strip().replace("\n", " ")[:PREVIEW_CHARS]
        print(f"[{rank}] page_number={doc.metadata.get('page_number')}  distance={score:.4f}")
        print(f"    {preview}...")
        print()


def main() -> None:
    store = load_vector_store()

    for query in TEST_QUERIES:
        run_query(store, query)

    print("--- Keyword-style sanity check ---")
    run_query(store, SANITY_QUERY)


if __name__ == "__main__":
    main()
