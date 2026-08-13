from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


def load_pdf_pages(pdf_path: str | Path) -> list[Document]:
    """Load a PDF into one Document per page.

    Each Document's metadata includes the loader's 0-indexed `page` plus a
    1-indexed `page_number`, which is what should be surfaced to users as
    the source page reference.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    for page in pages:
        page.metadata["page_number"] = page.metadata["page"] + 1

    return pages


def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split page-level Documents into overlapping chunks.

    Metadata (including page_number) is inherited from the source page by
    the splitter, so every chunk stays traceable to its originating page.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def ingest_pdf(
    pdf_path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[list[Document], list[Document]]:
    """Load and chunk a PDF. Returns (pages, chunks)."""
    pages = load_pdf_pages(pdf_path)
    chunks = chunk_documents(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return pages, chunks
