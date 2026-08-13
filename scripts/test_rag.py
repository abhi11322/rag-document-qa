"""End-to-end verification for the RAG generation stage.

Loads the persisted Chroma vector store, asks a set of test questions
(including one out-of-document question), and prints each generated
answer plus its source page numbers.

Requires LLM_API_KEY (and optionally LLM_PROVIDER / LLM_MODEL) to be set,
e.g. via a .env file. See .env.example.

Run from the project root (after scripts/build_index.py):
    python scripts/test_rag.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from app.llm_providers import get_llm_provider
from app.rag import ask
from app.vector_store import load_vector_store

IN_DOCUMENT_QUESTIONS = [
    "What are the three stages of a typical RAG process?",
    "What are the main limitations of Naive RAG?",
    "What is query expansion in RAG?",
    "How does RAG differ from fine-tuning?",
]
OUT_OF_DOMAIN_QUESTIONS = [
    "What is the capital of France?",
]


def print_result(question: str, result: dict) -> None:
    print("=" * 80)
    print(f"QUESTION: {question}")
    print("-" * 80)
    print(f"ANSWER: {result['answer']}")
    print()
    print("SOURCES:")
    if result["sources"]:
        for src in result["sources"]:
            print(f"  - page {src['page_number']} ({src['source']})")
    else:
        print("  (none)")
    print()


def main() -> None:
    try:
        provider = get_llm_provider()
    except RuntimeError as exc:
        print("Cannot run RAG generation test:\n")
        print(str(exc))
        sys.exit(1)

    store = load_vector_store()

    for question in IN_DOCUMENT_QUESTIONS + OUT_OF_DOMAIN_QUESTIONS:
        result = ask(question, store=store, provider=provider)
        print_result(question, result)


if __name__ == "__main__":
    main()
