"""Build and persist the Chroma vector database from data/Document.pdf.

Run from the project root:
    python scripts/build_index.py
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from app.ingestion import ingest_pdf
from app.vector_store import build_vector_store, get_embedding_model, resolve_persist_dir

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = Path(os.getenv("PDF_PATH", PROJECT_ROOT / "data" / "Document.pdf"))
if not PDF_PATH.is_absolute():
    PDF_PATH = PROJECT_ROOT / PDF_PATH


def main() -> None:
    print(f"Loading and chunking: {PDF_PATH}")
    pages, chunks = ingest_pdf(PDF_PATH)
    print(f"Pages loaded: {len(pages)}")
    print(f"Chunks to embed: {len(chunks)}")

    embeddings = get_embedding_model()
    print(f"Embedding model: {embeddings.model_name}")

    persist_dir = resolve_persist_dir()
    start = time.time()
    build_vector_store(chunks, embeddings=embeddings, persist_dir=persist_dir)
    elapsed = time.time() - start
    print(f"Embedded {len(chunks)} chunks in {elapsed:.1f}s")

    print(f"Chroma database persisted to: {persist_dir.resolve()}")


if __name__ == "__main__":
    main()
