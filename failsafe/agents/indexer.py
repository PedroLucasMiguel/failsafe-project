"""
Indexer agent — builds the vector store after the analyzer runs.

This is the final node in the discovery graph:

  interview → explorer → analyzer → indexer → END

What it does:
  1. Reads every file from disk (using file_tree from state)
  2. Gets the AI-generated summary for each file (from file_analyses in state)
  3. Gets subsystem and pattern metadata from the knowledge_base
  4. Builds enriched chunks: AI context header + raw code
  5. Embeds all chunks locally (all-MiniLM-L6-v2, runs on CPU)
  6. Stores them in LanceDB at <codebase_path>/.failsafe/vectors/

Output: state["vector_store_path"] is set so downstream can find the DB.

The indexer also prints a Rich completion panel showing how many files and
chunks were indexed.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from failsafe.knowledge.enricher import build_enriched_chunks
from failsafe.knowledge.vector_store import VectorStore
from failsafe.state.discovery import DiscoveryState

_console = Console()


def indexer_node(state: DiscoveryState) -> dict:
    """Embed and index all discovered files into LanceDB."""
    codebase_path = Path(state["codebase_path"])
    file_tree: list[str] = state.get("file_tree", [])
    file_analyses: dict[str, str] = state.get("file_analyses", {})
    kb_dict: dict = state.get("knowledge_base", {})

    # Build a subsystem lookup: file_path → subsystem_name
    subsystem_map = _build_subsystem_map(kb_dict)
    pattern_map = _build_pattern_map(kb_dict)

    vs = VectorStore.for_codebase(codebase_path)

    _console.rule("[bold dim]Vector Indexing[/bold dim]", style="dim")
    _console.print(
        f"  [dim]Embedding {len(file_tree)} files into local vector store...[/dim]")
    _console.print()

    all_chunks = []
    skipped = 0

    for rel_path in file_tree:
        abs_path = codebase_path / rel_path
        if not abs_path.exists() or not abs_path.is_file():
            skipped += 1
            continue

        try:
            code = abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            skipped += 1
            continue

        if not code.strip():
            continue

        file_summary = file_analyses.get(rel_path, "")
        subsystem = subsystem_map.get(rel_path, "")
        patterns = pattern_map.get(rel_path, "")

        chunks = build_enriched_chunks(
            rel_path, code, file_summary, subsystem, patterns)
        all_chunks.extend(chunks)

    if all_chunks:
        _console.print(f"  [dim]Indexing {len(all_chunks)} chunks...[/dim]")
        n_indexed = vs.index(all_chunks)
    else:
        n_indexed = 0

    store_path = str(vs.path)

    _console.print()
    _console.print(Panel(
        f"[green]✓[/green] Indexed [bold]{n_indexed}[/bold] chunks "
        f"from [bold]{len(file_tree) - skipped}[/bold] files\n"
        f"[dim]Store: {store_path}[/dim]",
        title="[bold]Vector Store Ready[/bold]",
        border_style="green",
        padding=(0, 2),
    ))
    _console.print()

    return {"vector_store_path": store_path}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_subsystem_map(kb_dict: dict) -> dict[str, str]:
    """Map each file path to its subsystem name."""
    result: dict[str, str] = {}
    for s in kb_dict.get("subsystems", []):
        for f in s.get("files", []):
            result[f] = s.get("name", "")
    return result


def _build_pattern_map(kb_dict: dict) -> dict[str, str]:
    """Map each file path to the pattern names that reference it."""
    result: dict[str, list[str]] = {}
    for p in kb_dict.get("coding_patterns", []):
        name = p.get("name", "")
        for ex in p.get("examples", []):
            result.setdefault(ex, []).append(name)
        for snippet in p.get("code_snippets", []):
            f = snippet.get("file", "")
            if f:
                result.setdefault(f, []).append(name)
    return {k: ", ".join(v) for k, v in result.items()}
