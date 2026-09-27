"""Split loaded documents into retrieval-sized chunks."""

from __future__ import annotations

import argparse

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

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


LANGUAGE_BY_NAME: dict[str, Language] = {
    "python": Language.PYTHON,
    "javascript": Language.JS,
    "typescript": Language.TS,
    "jsx": Language.JS,
    "tsx": Language.TS,
    "java": Language.JAVA,
    "go": Language.GO,
    "ruby": Language.RUBY,
    "php": Language.PHP,
    "c": Language.C,
    "cpp": Language.CPP,
    "csharp": Language.CSHARP,
}


def chunk_code_documents(
    documents: list[Document],
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Split source files using language-aware function/class separators."""
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    chunks: list[Document] = []
    for document in documents:
        language = str(document.metadata.get("language", "")).lower()
        splitter_language = LANGUAGE_BY_NAME.get(language)
        if splitter_language is None:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True,
            )
        else:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=splitter_language,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True,
            )

        file_chunks = splitter.split_documents([document])
        for chunk_index, chunk in enumerate(file_chunks):
            start_index = int(chunk.metadata.get("start_index", 0))
            start_line = document.page_content.count("\n", 0, start_index) + 1
            end_line = start_line + chunk.page_content.count("\n")
            chunk.metadata.update(
                {
                    "chunk_index": chunk_index,
                    "start_line": start_line,
                    "end_line": end_line,
                    "file_path": document.metadata.get("file_path", ""),
                    "language": language,
                }
            )
            chunks.append(chunk)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview Bug Buster AI document chunking.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    args = parser.parse_args()
    chunks = chunk_documents(load_documents(args.data_dir), args.chunk_size, args.chunk_overlap)
    print(f"Created {len(chunks)} chunks")


if __name__ == "__main__":
    main()
