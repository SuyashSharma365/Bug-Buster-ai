"""DocMind MCP server exposing semantic document search over stdio."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

load_dotenv()

# FastMCP handles the MCP protocol framing and exposes decorated Python functions
# as tools that MCP clients can discover and call.
mcp = FastMCP("docmind-docs")
_collection: chromadb.Collection | None = None
_encoder: SentenceTransformer | None = None


def _resources() -> tuple[chromadb.Collection, SentenceTransformer]:
    """Initialize the persistent vector resources once per server process."""
    global _collection, _encoder
    if _collection is None or _encoder is None:
        chroma_dir = Path(os.getenv("DOCMIND_CHROMA_DIR", "storage/chroma"))
        if not chroma_dir.exists():
            raise RuntimeError("ChromaDB is empty because storage does not exist. Run ingestion first.")
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection_name = os.getenv("DOCMIND_COLLECTION", "docmind_documents")
        try:
            _collection = client.get_collection(name=collection_name)
        except Exception as error:
            raise RuntimeError(f"Chroma collection '{collection_name}' is missing. Run ingestion first.") from error
        if _collection.count() == 0:
            raise RuntimeError("ChromaDB contains no document chunks. Add files to data/ and run ingestion first.")
        _encoder = SentenceTransformer(os.getenv("DOCMIND_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    return _collection, _encoder


@mcp.tool()
async def search_docs(query: str, top_k: int = 5) -> str:
    """Search ingested document chunks and return matches with metadata."""
    if not query.strip():
        return json.dumps({"error": "query must not be empty"})
    if top_k < 1 or top_k > 20:
        return json.dumps({"error": "top_k must be between 1 and 20"})
    try:
        collection, encoder = _resources()
        query_embedding = encoder.encode([query], normalize_embeddings=True).tolist()
        result: dict[str, Any] = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        matches = [
            {
                "text": text,
                "metadata": metadata or {},
                "distance": distance,
            }
            for text, metadata, distance in zip(
                result.get("documents", [[]])[0],
                result.get("metadatas", [[]])[0],
                result.get("distances", [[]])[0],
            )
        ]
        return json.dumps({"query": query, "matches": matches}, ensure_ascii=False)
    except Exception as error:
        return json.dumps({"error": str(error)})


if __name__ == "__main__":
    # stdio is intentional: langchain-mcp-adapters starts this process and
    # exchanges JSON-RPC messages through its stdin/stdout streams.
    mcp.run(transport="stdio")
