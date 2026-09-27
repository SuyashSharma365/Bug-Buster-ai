"""Clone a GitHub repository and load supported source files as Documents."""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from git import Repo
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SOURCE_EXTENSIONS: Final[dict[str, str]] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "jsx",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".cpp": "cpp",
    ".cs": "csharp",
}
SKIPPED_DIRECTORIES: Final[set[str]] = {
    "node_modules",
    "venv",
    ".venv",
    ".git",
    "dist",
    "build",
    "__pycache__",
    ".next",
    "vendor",
}
MAX_FILE_SIZE_BYTES: Final[int] = 500 * 1024


@dataclass
class RepoLoadReport:
    """Loaded code documents and files that could not be included."""

    documents: list[Document] = field(default_factory=list)
    skipped_size: list[str] = field(default_factory=list)
    skipped_encoding: list[str] = field(default_factory=list)
    chunks_stored: int = 0

    @property
    def skipped_files(self) -> list[str]:
        """Return all skipped files in stable order."""
        return sorted(set(self.skipped_size + self.skipped_encoding))


def _read_source(path: Path) -> str:
    """Read source using UTF-8, BOM-aware UTF-8, then Latin-1."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            logger.warning("File %s may have encoding issues; falling back to latin-1", path)
            return path.read_text(encoding="latin-1")


def load_repo_report(repo_url: str) -> RepoLoadReport:
    """Shallow-clone ``repo_url`` and load supported source files from it."""
    if not repo_url.strip():
        raise ValueError("repo_url must not be empty")

    report = RepoLoadReport()
    with tempfile.TemporaryDirectory(prefix="docmind-repo-") as temporary_directory:
        repo_root = Path(temporary_directory) / "repository"
        Repo.clone_from(repo_url, repo_root, depth=1)
        for current_root, directory_names, file_names in os.walk(repo_root):
            directory_names[:] = sorted(
                name for name in directory_names if name not in SKIPPED_DIRECTORIES
            )
            for file_name in sorted(file_names):
                path = Path(current_root) / file_name
                language = SOURCE_EXTENSIONS.get(path.suffix.lower())
                if language is None:
                    continue
                relative_path = path.relative_to(repo_root).as_posix()
                if path.stat().st_size > MAX_FILE_SIZE_BYTES:
                    report.skipped_size.append(relative_path)
                    logger.info("Skipping oversized source file: %s", relative_path)
                    continue
                try:
                    content = _read_source(path)
                except (OSError, UnicodeError) as error:
                    report.skipped_encoding.append(relative_path)
                    logger.warning("Skipping unreadable source file %s: %s", relative_path, error)
                    continue
                if not content.strip():
                    continue
                report.documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "file_path": relative_path,
                            "language": language,
                            "repo_url": repo_url,
                        },
                    )
                )
    return report


def load_repo_documents(repo_url: str) -> list[Document]:
    """Clone a repository and return its supported source files as Documents."""
    return load_repo_report(repo_url).documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone and load source files from a GitHub repository.")
    parser.add_argument("repo_url")
    args = parser.parse_args()
    report = load_repo_report(args.repo_url)
    print(f"Loaded {len(report.documents)} files")
    print(f"Skipped oversized files: {len(report.skipped_size)}")
    print(f"Skipped unreadable files: {len(report.skipped_encoding)}")


if __name__ == "__main__":
    main()
