"""Manual smoke test for the ingestion/chunking stage.

Run from the project root:
    python scripts/test_ingestion.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion import ingest_pdf

PDF_PATH = Path(__file__).resolve().parent.parent / "data" / "Document.pdf"


def main() -> None:
    pages, chunks = ingest_pdf(PDF_PATH)

    print(f"Pages loaded: {len(pages)}")
    print(f"Chunks generated: {len(chunks)}")

    representative = chunks[len(chunks) // 2]
    print("\n--- Representative chunk ---")
    print(representative.page_content)
    print("\n--- Representative chunk metadata ---")
    print(
        {
            "page_number": representative.metadata.get("page_number"),
            "page": representative.metadata.get("page"),
            "source": representative.metadata.get("source"),
        }
    )


if __name__ == "__main__":
    main()
