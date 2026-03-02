"""
Enricher — builds enriched code chunks for vector indexing.

The key idea: embed BOTH the AI-generated context header AND the raw code
together, so chunks are retrievable by meaning even in comment-free codebases.

Each chunk is structured as:

    # File: failsafe/llm.py
    # Subsystem: Token Optimization Layer
    # Purpose: Unified LLM invocation helper with caching and provider optimizations
    # Patterns: Session-scoped response cache, Anthropic cache_control hints
    # Key elements: invoke_cached, _detect_provider, _add_anthropic_cache_control
    # ---- code below ----
    def invoke_cached(llm, messages):
        ...

The embed_text (header + code) goes into LanceDB for similarity search.
Only the raw code is returned to coding agents on retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass


CHUNK_SIZE = 8_000  # characters per chunk — matches explorer.py


@dataclass
class EnrichedChunk:
    """One indexable unit: a slice of a source file with its AI context header."""

    file: str
    chunk_index: int
    subsystem: str
    embed_text: str    # header + code — used for searching
    code: str          # raw code only — returned on retrieval


def build_enriched_chunks(
    file_path: str,
    code: str,
    file_summary: str,
    subsystem: str = "",
    patterns: str = "",
    chunk_size: int = CHUNK_SIZE,
) -> list[EnrichedChunk]:
    """Split a source file into chunks and enrich each with an AI context header.

    Args:
        file_path:    Relative path to the file.
        code:         Full source code of the file.
        file_summary: AI-generated summary string from the explorer (multi-line,
                      contains purpose, key elements, dependencies, patterns).
        subsystem:    Subsystem name from the KB (e.g. "Token Optimization").
        patterns:     Comma-separated pattern names applicable to this file.
        chunk_size:   Maximum characters per chunk.

    Returns:
        List of EnrichedChunk objects, one per chunk.
    """
    chunks = _split(code, chunk_size)
    header = _make_header(file_path, file_summary, subsystem, patterns)
    result = []

    for i, chunk in enumerate(chunks):
        embed_text = f"{header}\n# ---- code below ----\n{chunk}"
        result.append(EnrichedChunk(
            file=file_path,
            chunk_index=i,
            subsystem=subsystem,
            embed_text=embed_text,
            code=chunk,
        ))

    return result


def make_header_from_description(
    file_path: str,
    description: str,
    subsystem: str = "",
) -> str:
    """Build a context header from an agent-provided description (fast path).

    Used during hot-updates when the coding agent already knows what it changed,
    so no LLM re-summarization is needed.

    Args:
        file_path:   Relative path to the file.
        description: Short description of what the agent changed/added.
        subsystem:   Subsystem name if known.

    Returns:
        Header string to prepend to the code chunk.
    """
    lines = [f"# File: {file_path}"]
    if subsystem:
        lines.append(f"# Subsystem: {subsystem}")
    lines.append(f"# Description: {description}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_header(
    file_path: str,
    file_summary: str,
    subsystem: str,
    patterns: str,
) -> str:
    """Build the AI context header from the explorer's structured summary."""
    lines = [f"# File: {file_path}"]

    if subsystem:
        lines.append(f"# Subsystem: {subsystem}")

    # Parse the multi-line summary from explorer
    if file_summary:
        for line in file_summary.splitlines():
            low = line.lower()
            if not low.startswith("key elements:") \
               and not low.startswith("dependencies:") \
               and not low.startswith("patterns:") \
               and not low.startswith("notes:"):
                # First line is the purpose
                lines.append(f"# Purpose: {line.strip()}")
                break

        for line in file_summary.splitlines():
            low = line.lower()
            if low.startswith("key elements:"):
                elems = line.split(":", 1)[1].strip()
                lines.append(f"# Key elements: {elems}")
            elif low.startswith("patterns:") and line.split(":", 1)[1].strip().lower() != "none":
                p = line.split(":", 1)[1].strip()
                lines.append(f"# Patterns: {p}")

    if patterns and patterns.lower() != "none":
        lines.append(f"# Additional patterns: {patterns}")

    return "\n".join(lines)


def _split(text: str, chunk_size: int) -> list[str]:
    """Split text into chunks of at most chunk_size characters.

    Tries to break at newlines to avoid cutting mid-line.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # Walk back to the last newline within a 200-char window
            nl = text.rfind("\n", end - 200, end)
            if nl > start:
                end = nl + 1
        chunks.append(text[start:end])
        start = end

    return chunks
