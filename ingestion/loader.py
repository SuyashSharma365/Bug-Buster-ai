    """Load supported documents from the Bug Buster AI data directory."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}
logger = logging.getLogger(__name__)


def load_pdf(path: Path) -> list[Document]:
    """Load each PDF page as a separate document."""
    reader = PdfReader(str(path))
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": str(path), "page": page_number},
        )
        for page_number, page in enumerate(reader.pages, start=1)
    ]


def load_text(path: Path) -> Document:
    """Load a Markdown or plain-text file as one document."""
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            logger.warning("File %s may have encoding issues; falling back to latin-1", path)
            content = path.read_text(encoding="latin-1")

    return Document(
        page_content=content,
        metadata={"source": str(path), "file_type": path.suffix.lower()},
    )


def load_documents(data_dir: str | Path = "data") -> list[Document]:
    """Recursively load PDFs, Markdown files, and text files from ``data_dir``."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")

    documents: list[Document] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES):
        if path.suffix.lower() == ".pdf":
            documents.extend(load_pdf(path))
        else:
            documents.append(load_text(path))
    return [document for document in documents if document.page_content.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Load Bug Buster AI source documents and report what was found.")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    documents = load_documents(args.data_dir)
    print(f"Loaded {len(documents)} source documents from {args.data_dir}")


if __name__ == "__main__":
    main()
