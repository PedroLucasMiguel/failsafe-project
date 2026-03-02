"""
Code-editing tools for the coding agent.

Five LangChain @tool-decorated functions:
  read_file        — read a file from the codebase
  write_file       — write/create a file in the codebase
  list_directory   — list files in a directory
  search_kb        — semantic search over the vector store
  get_file_context — get KB metadata for a specific file

All tools are relative to the active codebase path set in ToolContext.
Call set_tool_context() before running the coding agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Tool context — injected before the agent runs
# ---------------------------------------------------------------------------

@dataclass
class ToolContext:
    """Shared context injected into all tool calls."""
    codebase_path: Path
    kb_store: object        # KnowledgeStore — avoids circular import
    vector_store: object    # VectorStore | None
    files_written: dict[str, str] = field(
        default_factory=dict)  # path → content


_ctx: ToolContext | None = None


def set_tool_context(ctx: ToolContext) -> None:
    """Set the active tool context before running a coding agent."""
    global _ctx  # noqa: PLW0603
    _ctx = ctx


def get_tool_context() -> ToolContext:
    if _ctx is None:
        raise RuntimeError(
            "Tool context not set. Call set_tool_context() first.")
    return _ctx


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def read_file(path: str) -> str:
    """Read a source file from the codebase.

    Args:
        path: Relative path from the codebase root (e.g. 'failsafe/llm.py').

    Returns:
        File contents as a string, or an error message if not found.
    """
    ctx = get_tool_context()
    abs_path = ctx.codebase_path / path
    if not abs_path.exists():
        return f"ERROR: File not found: {path}"
    if not abs_path.is_file():
        return f"ERROR: Not a file: {path}"
    try:
        content = abs_path.read_text(encoding="utf-8", errors="replace")
        return content
    except OSError as e:
        return f"ERROR: Could not read {path}: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write or create a file in the codebase.

    Creates parent directories as needed. Records the write so it can be
    included in the KB update after the agent finishes.

    Args:
        path:    Relative path from the codebase root.
        content: Full content to write to the file.

    Returns:
        Confirmation message with the number of lines written.
    """
    ctx = get_tool_context()
    abs_path = ctx.codebase_path / path
    abs_path.parent.mkdir(parents=True, exist_ok=True)

    existing = ""
    if abs_path.exists():
        try:
            existing = abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass

    abs_path.write_text(content, encoding="utf-8")
    ctx.files_written[path] = content

    lines = content.count("\n") + 1
    is_new = not existing
    action = "Created" if is_new else "Updated"
    diff_lines = abs(content.count("\n") - existing.count("\n"))
    diff_msg = f" ({diff_lines} line{'s' if diff_lines != 1 else ''} changed)" if not is_new else ""

    return f"{action} {path} — {lines} lines{diff_msg}"


@tool
def list_directory(path: str = ".") -> str:
    """List files and directories at the given path within the codebase.

    Args:
        path: Relative path (default: codebase root '.').

    Returns:
        Formatted directory listing, or an error message.
    """
    ctx = get_tool_context()
    abs_path = ctx.codebase_path / path

    if not abs_path.exists():
        return f"ERROR: Path not found: {path}"
    if not abs_path.is_dir():
        return f"ERROR: Not a directory: {path}"

    entries = []
    for item in sorted(abs_path.iterdir()):
        rel = item.relative_to(ctx.codebase_path)
        if item.name.startswith(".") or item.name == "__pycache__":
            continue
        if item.is_dir():
            entries.append(f"  📁 {rel}/")
        else:
            size_kb = item.stat().st_size / 1024
            entries.append(f"  📄 {rel}  ({size_kb:.1f} KB)")

    if not entries:
        return f"(empty directory: {path})"

    return f"Contents of {path}:\n" + "\n".join(entries)


@tool
def search_kb(query: str, n: int = 5) -> str:
    """Search the knowledge base for code relevant to a query.

    Uses semantic vector search to find the most relevant code chunks
    from the indexed codebase. Returns actual source code.

    Args:
        query: Natural language description of what you're looking for.
               Examples: "how are LLM providers registered",
                         "caching implementation", "file reading tools"
        n:     Number of results (default 5, max 10).

    Returns:
        Code chunks with file paths and subsystem labels.
    """
    ctx = get_tool_context()
    n = min(n, 10)

    if ctx.vector_store is not None:
        return ctx.vector_store.search_as_context(query, n=n)

    # Fallback to keyword search if no vector store
    if ctx.kb_store is not None:
        results = ctx.kb_store.search_patterns(query)
        if results:
            return "\n\n".join(
                f"### Pattern: {r['name']}\n{r.get('description', '')}"
                for r in results[:n]
            )

    return "No vector store available. Run full discovery first."


@tool
def get_file_context(path: str) -> str:
    """Get knowledge base metadata for a specific file.

    Returns what the KB knows about this file: which subsystem it belongs to,
    its purpose, key elements, and any coding patterns that apply to it.

    Args:
        path: Relative path to the file (e.g. 'failsafe/llm.py').

    Returns:
        Formatted KB metadata for the file.
    """
    ctx = get_tool_context()

    if ctx.kb_store is None:
        return "No KB available."

    lines = [f"## KB context for `{path}`"]

    # Subsystem
    subsystem = ctx.kb_store.get_subsystem_for_file(path)
    if subsystem:
        lines.append(f"**Subsystem:** {subsystem}")

    # File analysis
    analysis = ctx.kb_store.get_file_analysis(path)
    if analysis:
        lines.append(f"**Purpose:** {analysis.get('purpose', 'N/A')}")
        elems = analysis.get("key_elements", [])
        if elems:
            lines.append(f"**Key elements:** {', '.join(elems)}")
        deps = analysis.get("dependencies", [])
        if deps:
            lines.append(f"**Dependencies:** {', '.join(deps)}")
    else:
        lines.append("*(no KB entry for this file yet)*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

CODING_TOOLS = [read_file, write_file,
                list_directory, search_kb, get_file_context]
