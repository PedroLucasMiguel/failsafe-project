"""
Sync engine — git-aware incremental KB and vector store update.

Designed for collaborative teams and post-git-pull workflows:

  git pull
  failsafe sync          # detects what changed, re-indexes only those files
  failsafe sync --full   # rebuilds entire vector store from committed KB (0 LLM calls)

How it works
------------
1. Reads `failsafe_kb.json` (committed to git — the shared source of truth).
2. Compares the current git HEAD to the last-synced commit (stored in
   `.failsafe/sync_state.json`).
3. For each changed/added file:
   - Has a summary in KB?  → re-enrich + re-embed → 0 LLM calls
   - New file, no summary? → 1 LLM call to summarize, then embed
4. Deleted files → removed from vector store.
5. Updates `.failsafe/sync_state.json` with the new HEAD commit.

Collaborative workflow
----------------------
  Developer A  →  full `failsafe` run  →  commits `failsafe_kb.json`
  Developer B  →  `git pull`           →  `failsafe sync` (0 LLM calls)
  New teammate →  `git clone`          →  `failsafe sync --full` (0 LLM calls)
  Coding agent →  edits files          →  `kb_update()` automatically (fast path)
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

_console = Console()

SYNC_STATE_FILE = ".failsafe/sync_state.json"
KB_FILE = "failsafe_kb.json"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_sync(
    codebase_path: str | Path,
    full: bool = False,
    llm=None,
) -> dict:
    """Sync the local vector store with the committed KB and codebase state.

    Args:
        codebase_path: Root of the codebase (where .git/ lives).
        full:          If True, rebuild vectors for ALL files (ignores git diff).
                       Useful for new teammates or after major restructuring.
        llm:           Optional LLM instance for summarizing new files.
                       If None, new files (not in KB) are embedded raw.

    Returns:
        Summary dict: {re_indexed, new_files_llm, deleted, skipped, errors}
    """
    codebase_path = Path(codebase_path)
    kb_path = codebase_path / KB_FILE

    if not kb_path.exists():
        _console.print(
            f"[red]No KB found at {kb_path}.[/red] "
            "Run [bold]failsafe[/bold] (full discovery) first."
        )
        return {}

    # Load committed KB
    kb_data = json.loads(kb_path.read_text(encoding="utf-8"))
    file_summaries: dict[str, str] = {
        fa["path"]: fa.get("purpose", "")
        for fa in kb_data.get("file_analyses", [])
    }
    subsystem_map = _build_subsystem_map(kb_data)
    pattern_map = _build_pattern_map(kb_data)

    # Open or create vector store
    from failsafe.knowledge.vector_store import VectorStore
    vs = VectorStore.for_codebase(codebase_path)

    sync_state = _load_sync_state(codebase_path)
    last_commit = sync_state.get("last_commit", "")
    current_commit = _get_head_commit(codebase_path)

    # Determine which files to re-index
    if full:
        _console.print(
            "\n[bold cyan]Full rebuild[/bold cyan] — re-indexing all files from KB...\n")
        changed, deleted = _all_files(codebase_path, kb_data), []
    else:
        _console.print(
            f"\n[bold cyan]Sync[/bold cyan] — detecting changes since [dim]{last_commit[:8] or 'initial'}[/dim]...\n")
        changed, deleted = _detect_changes(codebase_path, last_commit)
        if not changed and not deleted:
            _console.print(
                "[green]✓ Nothing to sync — vectors are up to date.[/green]\n")
            _save_sync_state(codebase_path, current_commit, vs.chunk_count)
            return {"re_indexed": 0, "new_files_llm": 0, "deleted": 0, "skipped": 0, "errors": 0}

    _console.print(
        f"  [dim]{len(changed)} file(s) to re-index, "
        f"{len(deleted)} deleted[/dim]\n"
    )

    # Remove stale chunks (changed + deleted files)
    files_to_delete = list({f for f in changed + deleted})
    if files_to_delete:
        vs.delete_files(files_to_delete)

    # Re-index changed files
    stats = {"re_indexed": 0, "new_files_llm": 0,
             "deleted": len(deleted), "skipped": 0, "errors": 0}

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        console=_console,
        transient=False,
    )
    task_id = progress.add_task("Indexing", total=len(changed))

    with progress:
        for rel_path in changed:
            abs_path = codebase_path / rel_path
            progress.update(task_id, advance=1,
                            description=f"[bold cyan]{_short(rel_path)}")

            if not abs_path.exists() or not abs_path.is_file():
                stats["skipped"] += 1
                continue

            try:
                code = abs_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                stats["errors"] += 1
                continue

            if not code.strip():
                stats["skipped"] += 1
                continue

            subsystem = subsystem_map.get(rel_path, "")
            patterns = pattern_map.get(rel_path, "")

            if rel_path in file_summaries:
                # Fast path: use committed KB summary — 0 LLM calls
                from failsafe.knowledge.enricher import build_enriched_chunks
                chunks = build_enriched_chunks(
                    rel_path, code, file_summaries[rel_path], subsystem, patterns
                )
            elif llm is not None:
                # New file: 1 LLM call to summarize
                chunks = _summarize_and_build(
                    rel_path, code, subsystem, patterns, llm)
                stats["new_files_llm"] += 1
            else:
                # No LLM available: embed raw with minimal header
                from failsafe.knowledge.enricher import build_enriched_chunks
                chunks = build_enriched_chunks(
                    rel_path, code, "", subsystem, patterns)

            n = vs.index(chunks)
            stats["re_indexed"] += n

    _save_sync_state(codebase_path, current_commit, vs.chunk_count)

    _console.print()
    _console.print(Panel(
        f"[green]✓[/green] Re-indexed [bold]{stats['re_indexed']}[/bold] chunks\n"
        f"[dim]New files analyzed: {stats['new_files_llm']} | "
        f"Deleted: {stats['deleted']} | "
        f"Skipped: {stats['skipped']}[/dim]\n"
        f"[dim]Total in store: {vs.chunk_count} chunks | "
        f"Commit: {current_commit[:8]}[/dim]",
        title="[bold]Sync Complete[/bold]",
        border_style="green",
        padding=(0, 2),
    ))
    _console.print()

    return stats


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _get_head_commit(codebase_path: Path) -> str:
    """Return current HEAD commit hash, or empty string if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=codebase_path,
            capture_output=True, text=True, timeout=5, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _detect_changes(
    codebase_path: Path,
    since_commit: str,
) -> tuple[list[str], list[str]]:
    """Return (changed_files, deleted_files) since the given commit.

    Falls back to all tracked files if since_commit is empty (first sync).
    """
    if not since_commit:
        # No previous sync — return all tracked files as "changed"
        return _all_tracked_files(codebase_path), []

    try:
        result = subprocess.run(
            ["git", "diff", "--name-status", since_commit, "HEAD"],
            cwd=codebase_path,
            capture_output=True, text=True, timeout=10, check=False,
        )
        if result.returncode != 0:
            return _all_tracked_files(codebase_path), []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return _all_tracked_files(codebase_path), []

    changed = []
    deleted = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        status, path = parts[0], parts[1]
        if status.startswith("D"):
            deleted.append(path)
        else:
            changed.append(path)

    return changed, deleted


def _all_tracked_files(codebase_path: Path) -> list[str]:
    """Return all git-tracked files."""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=codebase_path,
            capture_output=True, text=True, timeout=10, check=False,
        )
        return result.stdout.splitlines() if result.returncode == 0 else []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def _all_files(codebase_path: Path, kb_data: dict) -> list[str]:
    """Return all files that are either tracked by git or in the KB."""
    tracked = set(_all_tracked_files(codebase_path))
    in_kb = {fa["path"] for fa in kb_data.get("file_analyses", [])}
    return sorted(tracked | in_kb)


# ---------------------------------------------------------------------------
# Sync state
# ---------------------------------------------------------------------------

def _load_sync_state(codebase_path: Path) -> dict:
    state_path = codebase_path / SYNC_STATE_FILE
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_sync_state(codebase_path: Path, commit: str, chunk_count: int) -> None:
    state_path = codebase_path / SYNC_STATE_FILE
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "last_commit": commit,
        "last_sync": datetime.now(timezone.utc).isoformat(),
        "chunks_indexed": chunk_count,
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Summarization helpers
# ---------------------------------------------------------------------------

def _summarize_and_build(
    rel_path: str,
    code: str,
    subsystem: str,
    patterns: str,
    llm,
) -> list:
    """Summarize a new file with LLM, then build enriched chunks."""
    from langchain_core.messages import SystemMessage, HumanMessage
    from failsafe.llm import invoke_cached
    from failsafe.agents.explorer import BATCH_SUMMARY_PROMPT, _parse_batch_blocks
    from failsafe.knowledge.enricher import build_enriched_chunks

    content = f"--- {rel_path} ---\n{code[:8000]}\n"
    response = invoke_cached(llm, [
        SystemMessage(content=BATCH_SUMMARY_PROMPT),
        HumanMessage(content=content),
    ])
    parsed = _parse_batch_blocks(response)
    summary = parsed.get(rel_path, "")
    return build_enriched_chunks(rel_path, code, summary, subsystem, patterns)


# ---------------------------------------------------------------------------
# KB helpers
# ---------------------------------------------------------------------------

def _build_subsystem_map(kb_data: dict) -> dict[str, str]:
    result: dict[str, str] = {}
    for s in kb_data.get("subsystems", []):
        for f in s.get("files", []):
            result[f] = s.get("name", "")
    return result


def _build_pattern_map(kb_data: dict) -> dict[str, str]:
    result: dict[str, list[str]] = {}
    for p in kb_data.get("coding_patterns", []):
        name = p.get("name", "")
        for ex in p.get("examples", []):
            result.setdefault(ex, []).append(name)
    return {k: ", ".join(v) for k, v in result.items()}


def _short(path: str, max_len: int = 40) -> str:
    return path if len(path) <= max_len else f"...{path[-(max_len-3):]}"
