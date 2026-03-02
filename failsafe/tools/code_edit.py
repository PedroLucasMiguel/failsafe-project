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


@tool
def create_directory(path: str, with_init: bool = False) -> str:
    """Create a new directory (and any missing parents) in the codebase.

    Use this to scaffold new packages or module directories before writing files.
    Note: write_file already creates parent directories automatically, so you
    only need this when you want an empty directory or need to add __init__.py.

    Args:
        path:       Relative path of the directory to create.
        with_init:  If True, also creates an empty __init__.py inside it,
                    making it a proper Python package.

    Returns:
        Confirmation message, or error if creation fails.
    """
    ctx = get_tool_context()
    abs_path = ctx.codebase_path / path
    try:
        abs_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return f"ERROR: Could not create directory {path}: {e}"

    msg = f"Created directory: {path}/"
    if with_init:
        init_file = abs_path / "__init__.py"
        if not init_file.exists():
            init_file.write_text("", encoding="utf-8")
            ctx.files_written[f"{path}/__init__.py"] = ""
            msg += " (with __init__.py)"

    return msg


@tool
def read_file_section(path: str, start_line: int, end_line: int) -> str:
    """Read a specific range of lines from a file without loading the whole thing.

    Use this when you only need to see a specific function, class, or
    block in a large file. Much more token-efficient than read_file for
    files longer than ~150 lines.

    Args:
        path:       Relative path from the codebase root.
        start_line: First line to read (1-indexed, inclusive).
        end_line:   Last line to read (1-indexed, inclusive).

    Returns:
        The requested lines with line numbers prepended, or an error message.
    """
    ctx = get_tool_context()
    abs_path = ctx.codebase_path / path
    if not abs_path.exists():
        return f"ERROR: File not found: {path}"
    if not abs_path.is_file():
        return f"ERROR: Not a file: {path}"

    try:
        lines = abs_path.read_text(
            encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        return f"ERROR: Could not read {path}: {e}"

    total = len(lines)
    start = max(1, start_line) - 1
    end = min(total, end_line)

    if start >= total:
        return f"ERROR: start_line {start_line} exceeds file length ({total} lines)"

    selected = lines[start:end]
    numbered = "\n".join(
        f"{start_line + i:4d} | {l}" for i, l in enumerate(selected))
    return f"{path} (lines {start_line}–{end}, total {total}):\n{numbered}"


@tool
def patch_file(path: str, old_text: str, new_text: str) -> str:
    """Replace an exact snippet of code in a file without rewriting the whole file.

    This is the preferred way to edit large files. Instead of reading the
    entire file and writing it back, supply only the exact block you want
    to change and its replacement.

    Rules:
    - old_text must match EXACTLY (including whitespace/indentation).
    - If old_text appears more than once, only the FIRST occurrence is replaced.
    - If old_text is not found, returns an error — do NOT guess.

    Args:
        path:     Relative path from the codebase root.
        old_text: The exact existing code to replace (must be unique in the file).
        new_text: The replacement code.

    Returns:
        Success message with the line range that was patched, or an error.
    """
    ctx = get_tool_context()
    abs_path = ctx.codebase_path / path
    if not abs_path.exists():
        return f"ERROR: File not found: {path}"

    try:
        content = abs_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"ERROR: Could not read {path}: {e}"

    if old_text not in content:
        # Give a useful hint: show a few lines around the closest-matching region
        return (
            f"ERROR: old_text not found in {path}. "
            "Use read_file_section to inspect the exact content first, then retry."
        )

    # replace first occurrence only
    new_content = content.replace(old_text, new_text, 1)

    try:
        abs_path.write_text(new_content, encoding="utf-8")
    except OSError as e:
        return f"ERROR: Could not write {path}: {e}"

    # Track the write
    ctx.files_written[path] = new_content

    # Report which line was patched (helps the agent confirm the right spot)
    patch_start = content[:content.index(old_text)].count("\n") + 1
    patch_end = patch_start + old_text.count("\n")
    return (
        f"Patched {path} (lines {patch_start}-{patch_end}): "
        f"{old_text.count(chr(10)) + 1} lines → {new_text.count(chr(10)) + 1} lines"
    )


@tool
def grep_file(path: str, pattern: str, context_lines: int = 20) -> str:
    """Search for a pattern in a file and return matching lines with surrounding context.

    Use this as your PRIMARY research tool before touching any file. It returns
    exact line numbers and up to context_lines of surrounding code. The default
    of 20 lines covers most function bodies — if you can see the full block you
    want to change in the output, call patch_file DIRECTLY without reading more.

    Args:
        path:          Relative path from the codebase root.
        pattern:       Text to search for (case-sensitive substring, not regex).
        context_lines: Lines to show before/after each match (default 20).
                       Increase to 40-50 for very large functions.

    Returns:
        Matching lines with line numbers and context, or a message if not found.
    """
    ctx = get_tool_context()
    abs_path = ctx.codebase_path / path
    if not abs_path.exists():
        return f"ERROR: File not found: {path}"
    if not abs_path.is_file():
        return f"ERROR: Not a file: {path}"

    try:
        lines = abs_path.read_text(
            encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        return f"ERROR: Could not read {path}: {e}"

    # Find all line indices where pattern appears
    matches = [i for i, line in enumerate(lines) if pattern in line]
    if not matches:
        return f"Pattern {pattern!r} not found in {path} ({len(lines)} lines total)."

    total = len(lines)
    blocks: list[str] = []
    covered: set[int] = set()

    for mi in matches:
        start = max(0, mi - context_lines)
        end = min(total, mi + context_lines + 1)
        # Skip if already covered by a previous match
        if mi in covered:
            continue
        covered.update(range(start, end))

        block_lines = []
        for i in range(start, end):
            marker = ">>>" if i == mi else "   "
            block_lines.append(f"{i + 1:4d} {marker} {lines[i]}")
        blocks.append("\n".join(block_lines))

    header = f"{path}: {len(matches)} match(es) for {pattern!r}\n"
    return header + "\n---\n".join(blocks)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

CODING_TOOLS = [
    grep_file,          # find exact line numbers first — avoids read_file_section loops
    read_file_section,
    read_file,
    write_file,
    patch_file,
    create_directory,
    list_directory,
    search_kb,
    get_file_context,
]
