"""Embed a Git repository's source code into the separate ``code_chunks`` collection."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer

from ingestion.chunker import chunk_code_documents
from ingestion.repo_loader import RepoLoadReport, load_repo_report

CODE_COLLECTION_NAME = "code_chunks"


def build_code_collection(
    chroma_dir: str | Path | None = None,
    embedding_model: str | None = None,
    reset: bool = False,
) -> tuple[chromadb.Collection, SentenceTransformer]:
    """Open the dedicated code collection and configured embedding model."""
    load_dotenv()
    directory = Path(chroma_dir or os.getenv("DOCMIND_CHROMA_DIR", "storage/chroma"))
    directory.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(directory))
    if reset:
        try:
            client.delete_collection(CODE_COLLECTION_NAME)
        except NotFoundError:
            pass
    collection = client.get_or_create_collection(name=CODE_COLLECTION_NAME)
    model_name = embedding_model or os.getenv("DOCMIND_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    return collection, SentenceTransformer(model_name)


def ingest_repository(
    repo_url: str,
    chroma_dir: str | Path | None = None,
    embedding_model: str | None = None,
    reset: bool = False,
) -> RepoLoadReport:
    """Clone, chunk, embed, and persist a repository's source files."""
    report = load_repo_report(repo_url)
    if not report.documents:
        raise ValueError("No supported source files were found in the repository")

    chunks = chunk_code_documents(report.documents)
    collection, model = build_code_collection(chroma_dir, embedding_model, reset)
    embeddings = model.encode(
        [chunk.page_content for chunk in chunks],
        normalize_embeddings=True,
    ).tolist()
    ids = [
        hashlib.sha1(
            f"{repo_url}:{chunk.metadata.get('file_path')}:{chunk.metadata.get('chunk_index')}".encode(),
            usedforsecurity=False,
        ).hexdigest()
        for chunk in chunks
    ]
    collection.upsert(
        ids=ids,
        documents=[chunk.page_content for chunk in chunks],
        embeddings=embeddings,
        metadatas=[{key: str(value) for key, value in chunk.metadata.items()} for chunk in chunks],
    )
    report.chunks_stored = len(chunks)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed a GitHub repository into Bug Buster AI's code_chunks collection.")
    parser.add_argument("--repo-url", required=True, help="GitHub repository URL to clone")
    parser.add_argument("--chroma-dir", default=os.getenv("DOCMIND_CHROMA_DIR", "storage/chroma"))
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--reset", action="store_true", help="Replace the existing code_chunks collection")
    args = parser.parse_args()

    report = ingest_repository(args.repo_url, args.chroma_dir, args.embedding_model, args.reset)
    print(f"Files processed: {len(report.documents)}")
    print(f"Chunks stored: {report.chunks_stored}")
    print(f"Files skipped due to size: {len(report.skipped_size)}")
    for path in report.skipped_size:
        print(f"  - {path}")
    print(f"Files skipped due to encoding/read errors: {len(report.skipped_encoding)}")
    for path in report.skipped_encoding:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
