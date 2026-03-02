"""
Filesystem tools — reading files, listing directories, parsing .gitignore.

All tools are decorated with @tool so LangGraph agents can call them.
File reads are capped to stay token-efficient.
"""

from __future__ import annotations

import os
from pathlib import Path

import pathspec
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BINARY_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".exe", ".dll", ".so", ".dylib", ".bin",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".pyc", ".pyo", ".class", ".o",
    ".db", ".sqlite", ".sqlite3",
})


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
def read_file(file_path: str) -> str:
    """Read the full contents of a text file.

    Args:
        file_path: Absolute path to the file.

    Returns:
        Full file contents, or an error/skip message for binary/missing files.
    """
    path = Path(file_path)

    if not path.exists():
        return f"[error] File not found: {file_path}"

    if not path.is_file():
        return f"[error] Not a file: {file_path}"

    if path.suffix.lower() in BINARY_EXTENSIONS:
        return f"[skip] Binary file: {file_path}"

    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"[error] Could not read {file_path}: {exc}"


@tool
def list_directory(
    directory_path: str,
    gitignore_patterns: list[str] | None = None,
) -> str:
    """List all files in a directory tree, respecting .gitignore patterns.

    Args:
        directory_path: Absolute path to the root directory.
        gitignore_patterns: Optional list of gitignore-style patterns to skip.

    Returns:
        Newline-separated list of relative file paths, or an error message.
    """
    root = Path(directory_path)

    if not root.is_dir():
        return f"[error] Not a directory: {directory_path}"

    spec = _build_pathspec(gitignore_patterns)
    files: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden directories (e.g. .git)
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".")
        ]

        rel_dir = Path(dirpath).relative_to(root)

        for fname in sorted(filenames):
            if fname.startswith("."):
                continue

            rel_path = str(rel_dir / fname) if str(rel_dir) != "." else fname

            if spec and spec.match_file(rel_path):
                continue

            files.append(rel_path)

    return "\n".join(files) if files else "[empty] No files found."


@tool
def parse_gitignore(codebase_path: str) -> list[str]:
    """Parse the .gitignore file at the root of a codebase.

    Args:
        codebase_path: Absolute path to the codebase root.

    Returns:
        List of gitignore pattern strings, or an empty list if no .gitignore.
    """
    gitignore_path = Path(codebase_path) / ".gitignore"

    if not gitignore_path.exists():
        return []

    lines = gitignore_path.read_text(
        encoding="utf-8", errors="replace").splitlines()

    # Filter out blank lines and comments
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_pathspec(patterns: list[str] | None) -> pathspec.PathSpec | None:
    """Turn a list of gitignore patterns into a pathspec matcher."""
    if not patterns:
        return None
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)
