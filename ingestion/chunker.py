"""Split loaded documents into retrieval-sized chunks."""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.loader import load_documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Document]:
    """Split documents while retaining their source metadata."""
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview DocMind document chunking.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    args = parser.parse_args()
    chunks = chunk_documents(load_documents(args.data_dir), args.chunk_size, args.chunk_overlap)
    print(f"Created {len(chunks)} chunks")


if __name__ == "__main__":
    main()
