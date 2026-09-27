"""Embed document chunks with Sentence Transformers and persist them in Chroma."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from ingestion.chunker import chunk_documents
from ingestion.loader import load_documents


def build_collection(
    chroma_dir: str | Path | None = None,
    collection_name: str | None = None,
    embedding_model: str | None = None,
) -> tuple[chromadb.PersistentClient, chromadb.Collection, SentenceTransformer]:
    """Open the persistent Chroma collection and its matching embedding model."""
    load_dotenv()
    directory = Path(chroma_dir or os.getenv("DOCMIND_CHROMA_DIR", "storage/chroma"))
    directory.mkdir(parents=True, exist_ok=True)
    model_name = embedding_model or os.getenv("DOCMIND_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    client = chromadb.PersistentClient(path=str(directory))
    collection = client.get_or_create_collection(name=collection_name or os.getenv("DOCMIND_COLLECTION", "docmind_documents"))
    return client, collection, SentenceTransformer(model_name)


def ingest(
    data_dir: str | Path = "data",
    chroma_dir: str | Path | None = None,
    collection_name: str | None = None,
    embedding_model: str | None = None,
    reset: bool = False,
) -> int:
    """Load, chunk, embed, and persist source documents. Return chunk count."""
    documents = load_documents(data_dir)
    if not documents:
        raise ValueError(f"No .pdf, .md, or .txt documents found in {data_dir}")
    chunks = chunk_documents(documents)
    client, collection, model = build_collection(chroma_dir, collection_name, embedding_model)
    if reset:
        client.delete_collection(collection.name)
        collection = client.get_or_create_collection(name=collection.name)
    embeddings = model.encode([chunk.page_content for chunk in chunks], normalize_embeddings=True).tolist()
    ids = [f"chunk-{index}" for index in range(len(chunks))]
    collection.upsert(
        ids=ids,
        documents=[chunk.page_content for chunk in chunks],
        embeddings=embeddings,
        metadatas=[{key: str(value) for key, value in chunk.metadata.items()} for chunk in chunks],
    )
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed Bug Buster AI documents into persistent ChromaDB storage.")
    parser.add_argument("--data-dir", default=os.getenv("DOCMIND_DATA_DIR", "data"))
    parser.add_argument("--reset", action="store_true", help="Replace existing chunks with this ingestion run")
    args = parser.parse_args()
    count = ingest(args.data_dir, reset=args.reset)
    print(f"Stored {count} chunks in ChromaDB")


if __name__ == "__main__":
    main()
