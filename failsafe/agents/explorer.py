"""
Explorer agent — traverses the codebase and generates per-file summaries.

Strategy for full file reading:
  1. List all files, sort by likely importance.
  2. Read each file in full.
  3. Small files (≤ CHUNK_SIZE): batch together and summarize in one LLM call.
  4. Large files (> CHUNK_SIZE): split into chunks, summarize each chunk
     individually, then combine chunk summaries into a final file summary.
  5. Skip binary files entirely.

Progress display:
  A rich Live layout shows a progress bar, files done/total, ETA, and a
  live token counter while exploration is running.
"""

from __future__ import annotations

import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from failsafe.llm import extract_text, invoke_cached
from failsafe.state.discovery import DiscoveryState
from failsafe.tools.filesystem import list_directory, parse_gitignore, read_file
from failsafe.tracking import tracker


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_SIZE = 8_000     # Characters per chunk (~2k tokens) sent to LLM
# Target characters per batch (fills context efficiently)
BATCH_BUDGET = 30_000

PRIORITY_FILENAMES = frozenset({
    "readme.md", "readme", "readme.txt",
    "package.json", "pyproject.toml", "cargo.toml", "go.mod", "pom.xml",
    "makefile", "dockerfile", "docker-compose.yml", "docker-compose.yaml",
    ".env.example", "requirements.txt", "setup.py", "setup.cfg",
    "main.py", "main.go", "main.ts", "main.js", "app.py", "index.ts",
    "index.js", "manage.py", "mod.rs", "lib.rs",
})

_console = Console()


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
BATCH_SUMMARY_PROMPT = """\
You are a codebase explorer. Analyze the contents of the given source files \
and produce a structured summary for EACH file.

Output format — repeat this block for every file, separated by "---":

FILE: <relative_path>
PURPOSE: <one sentence describing what this file does>
ELEMENTS: <comma-separated list of key classes, functions, constants, or types>
DEPENDENCIES: <comma-separated list of key imports or modules this file relies on>
PATTERNS: <any notable coding patterns, conventions, or design decisions visible in this file — be specific, e.g. "@tool decorator for LangChain registration", "TypedDict state schema", "singleton module-level instance">
---

Rules:
- Output one block per file, always in this exact format.
- ELEMENTS: list the most important symbols (not all of them), comma-separated.
- PATTERNS: if nothing notable, write "none".
- Do NOT add explanations, preambles, or any extra text outside the blocks.
- Trivial/empty files (blank __init__.py, etc.): write ELEMENTS: none, PATTERNS: none.
"""

CHUNK_SUMMARY_PROMPT = """\
You are a codebase explorer. Summarize the following code chunk. \
Extract structured information about its purpose, key symbols, and patterns.

Output this exact format:
PURPOSE: <one sentence>
ELEMENTS: <comma-separated key classes/functions/types>
DEPENDENCIES: <comma-separated imports/modules>
PATTERNS: <notable patterns or conventions visible in this chunk, or "none">
NOTES: <anything else a developer needs to know about this chunk>
"""

CHUNK_COMBINE_PROMPT = """\
You are a codebase explorer. Given structured summaries of chunks from the \
SAME file, produce a SINGLE consolidated structured summary for the whole file.

Output this exact format (no extra text):
FILE: <relative_path>
PURPOSE: <one sentence describing the whole file>
ELEMENTS: <combined comma-separated list of key classes, functions, types>
DEPENDENCIES: <combined comma-separated imports/modules>
PATTERNS: <notable patterns or conventions — merge insights from all chunks>
---
"""


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------
def explorer_node(state: DiscoveryState) -> dict:
    """Explore the codebase and produce per-file summaries."""
    from failsafe.llm import get_llm

    llm = get_llm()
    codebase_path = state["codebase_path"]

    # 1. Parse gitignore
    patterns = state.get("gitignore_patterns") or []
    if not patterns:
        patterns = parse_gitignore.invoke({"codebase_path": codebase_path})

    # 2. List all files
    file_list_raw = list_directory.invoke({
        "directory_path": codebase_path,
        "gitignore_patterns": patterns,
    })
    file_tree = [
        line.strip()
        for line in file_list_raw.splitlines()
        if line.strip() and not line.startswith("[")
    ]

    # 3. Sort by priority (important files first)
    file_tree = _prioritize(file_tree)

    # 4. Read all files and split into small vs large (with progress)
    small_files: list[tuple[str, str]] = []
    large_files: list[tuple[str, str]] = []

    _console.print(f"\n  [dim]Reading {len(file_tree)} files...[/dim]")

    for rel_path in file_tree:
        abs_path = str(Path(codebase_path) / rel_path)
        content = read_file.invoke({"file_path": abs_path})

        if content.startswith("[skip]") or content.startswith("[error]"):
            continue

        if len(content) <= CHUNK_SIZE:
            small_files.append((rel_path, content))
        else:
            large_files.append((rel_path, content))

    # Build budget-aware batches (fill by chars, not file count)
    batches = _build_batches(small_files)

    # Total LLM calls = one per batch + (chunks + 1) per large file
    total_large_calls = sum(
        len(_split_into_chunks(content)) + 1
        for _, content in large_files
    )
    total_llm_calls = len(batches) + total_large_calls
    total_files = len(small_files) + len(large_files)

    file_analyses: dict[str, str] = {}
    calls_done = 0
    start_time = time.monotonic()

    # Build a rich Progress display
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=_console,
        transient=False,
    )
    task_id = progress.add_task(
        "Summarizing files",
        total=total_llm_calls,
    )

    def _tick(label: str = "") -> None:
        """Advance progress and refresh the token stat line."""
        nonlocal calls_done
        calls_done += 1
        progress.update(
            task_id,
            advance=1,
            description=f"[bold cyan]{label}" if label else "Summarizing files",
        )

    with progress:
        # 5. Batch-summarize small files (budget-aware batches)
        for batch_num, batch in enumerate(batches, 1):
            _tick(f"Batch {batch_num}/{len(batches)}  ({_token_str()})")
            results = _summarize_batch(llm, batch)
            file_analyses.update(results)

        # 6. Chunk-summarize large files
        for rel_path, content in large_files:
            chunks = _split_into_chunks(content)
            chunk_summaries: list[str] = []

            for j, chunk in enumerate(chunks, 1):
                _tick(
                    f"[large] {_short(rel_path)} chunk {j}/{len(chunks)}  ({_token_str()})")
                header = f"File: {rel_path} — Chunk {j}/{len(chunks)}"
                text = invoke_cached(llm, [
                    SystemMessage(content=CHUNK_SUMMARY_PROMPT),
                    HumanMessage(content=f"{header}\n\n{chunk}"),
                ])
                chunk_summaries.append(f"Chunk {j}: {text}")

            # Combine chunk summaries
            _tick(f"[large] {_short(rel_path)} combining  ({_token_str()})")
            combined_input = (
                f"File: {rel_path}\n"
                f"Total chunks: {len(chunks)}\n\n"
                + "\n\n".join(chunk_summaries)
            )
            combine_text = invoke_cached(llm, [
                SystemMessage(content=CHUNK_COMBINE_PROMPT),
                HumanMessage(content=combined_input),
            ])
            # Parse the structured block; fallback to raw text if parsing fails
            parsed = _parse_batch_blocks(combine_text)
            file_analyses[rel_path] = parsed.get(
                rel_path) or combine_text.strip()

    elapsed = time.monotonic() - start_time
    _print_exploration_summary(total_files, elapsed)

    return {
        "file_tree": file_tree,
        "file_analyses": file_analyses,
        "gitignore_patterns": patterns,
        "current_phase": "exploration_done",
    }


# ---------------------------------------------------------------------------
# Small file batching (budget-aware)
# ---------------------------------------------------------------------------
def _build_batches(
    files: list[tuple[str, str]],
    budget: int = BATCH_BUDGET,
) -> list[list[tuple[str, str]]]:
    """Pack small files into batches by character budget.

    Instead of a fixed file count per batch(BATCH_SIZE), we fill each batch
    up to ``budget`` characters. This makes batches as dense as possible:
    - Many tiny files pack into fewer batches.
    - Larger files that still fit under CHUNK_SIZE each get their own batch.
    """
    batches: list[list[tuple[str, str]]] = []
    current_batch: list[tuple[str, str]] = []
    current_size = 0

    for rel_path, content in files:
        file_size = len(content) + len(rel_path) + 20  # separator overhead
        if current_batch and current_size + file_size > budget:
            batches.append(current_batch)
            current_batch = []
            current_size = 0
        current_batch.append((rel_path, content))
        current_size += file_size

    if current_batch:
        batches.append(current_batch)

    return batches or [[]]  # always at least one batch (may be empty)


def _summarize_batch(llm, batch: list[tuple[str, str]]) -> dict[str, str]:
    """Summarize a batch of small files in one LLM call, returning structured summaries."""
    results: dict[str, str] = {}

    batch_content = "\n".join(
        f"--- {rel_path} ---\n{content}\n"
        for rel_path, content in batch
    )

    if not batch_content.strip():
        return results

    text = invoke_cached(llm, [
        SystemMessage(content=BATCH_SUMMARY_PROMPT),
        HumanMessage(content=batch_content),
    ])

    results.update(_parse_batch_blocks(text))
    return results


def _parse_batch_blocks(text: str) -> dict[str, str]:
    """Parse structured FILE/PURPOSE/ELEMENTS/DEPENDENCIES/PATTERNS blocks.

    Each block is separated by '---'. Returns {rel_path: structured_summary}.
    The structured_summary is a multi-line string that the analyzer can read
    directly — much richer than one-liner pipe format.
    """
    results: dict[str, str] = {}
    # Split on '---' separator (blocks start with FILE:)
    raw_blocks = text.split("---")

    for block in raw_blocks:
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        if not lines:
            continue

        parsed: dict[str, str] = {}
        for line in lines:
            for key in ("FILE", "PURPOSE", "ELEMENTS", "DEPENDENCIES", "PATTERNS", "NOTES"):
                if line.upper().startswith(f"{key}:"):
                    parsed[key] = line[len(key) + 1:].strip()
                    break

        if "FILE" not in parsed or "PURPOSE" not in parsed:
            continue

        # Build a rich multi-line summary string
        summary = f"{parsed.get('PURPOSE', '')}"
        if parsed.get("ELEMENTS") and parsed["ELEMENTS"].lower() != "none":
            summary += f"\nKey elements: {parsed['ELEMENTS']}"
        if parsed.get("DEPENDENCIES") and parsed["DEPENDENCIES"].lower() != "none":
            summary += f"\nDependencies: {parsed['DEPENDENCIES']}"
        if parsed.get("PATTERNS") and parsed["PATTERNS"].lower() != "none":
            summary += f"\nPatterns: {parsed['PATTERNS']}"
        if parsed.get("NOTES") and parsed["NOTES"].lower() not in ("none", ""):
            summary += f"\nNotes: {parsed['NOTES']}"

        results[parsed["FILE"]] = summary

    return results


# ---------------------------------------------------------------------------
# Large file chunked summarization
# ---------------------------------------------------------------------------
def _summarize_large_file(llm, rel_path: str, content: str) -> str:
    """Summarize a large file by chunking, summarizing each, then combining."""
    chunks = _split_into_chunks(content)
    chunk_summaries: list[str] = []

    for i, chunk in enumerate(chunks, 1):
        header = f"File: {rel_path} — Chunk {i}/{len(chunks)}"
        response = llm.invoke([
            SystemMessage(content=CHUNK_SUMMARY_PROMPT),
            HumanMessage(content=f"{header}\n\n{chunk}"),
        ])
        tracker.record(response)
        chunk_summaries.append(f"Chunk {i}: {extract_text(response)}")

    combined_input = (
        f"File: {rel_path}\n"
        f"Total chunks: {len(chunks)}\n\n"
        + "\n\n".join(chunk_summaries)
    )
    response = llm.invoke([
        SystemMessage(content=CHUNK_COMBINE_PROMPT),
        HumanMessage(content=combined_input),
    ])
    tracker.record(response)
    return extract_text(response)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _token_str() -> str:
    """Compact token count string for the progress description."""
    return f"🪙 {tracker.total_tokens:,} tok"


def _short(path: str, max_len: int = 30) -> str:
    """Shorten a path for display."""
    return path if len(path) <= max_len else f"…{path[-(max_len - 1):]}"


def _print_exploration_summary(total_files: int, elapsed: float) -> None:
    """Print a summary panel after exploration completes."""
    mins, secs = divmod(int(elapsed), 60)
    elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"
    _console.print(Panel(
        f"[bold green]Exploration complete[/bold green]\n"
        f"Files:   [cyan]{total_files}[/cyan]\n"
        f"Tokens:  [cyan]{tracker.summary_str()}[/cyan]\n"
        f"Elapsed: [cyan]{elapsed_str}[/cyan]",
        title="📂 Explorer",
        border_style="green",
    ))


def _prioritize(files: list[str]) -> list[str]:
    """Sort files so high-priority ones come first."""
    priority: list[str] = []
    rest: list[str] = []

    for f in files:
        if Path(f).name.lower() in PRIORITY_FILENAMES:
            priority.append(f)
        else:
            rest.append(f)

    return priority + rest


def _split_into_chunks(content: str) -> list[str]:
    """Split file content into chunks, trying to break at line boundaries."""
    if len(content) <= CHUNK_SIZE:
        return [content]

    chunks: list[str] = []
    start = 0

    while start < len(content):
        end = start + CHUNK_SIZE

        if end >= len(content):
            chunks.append(content[start:])
            break

        # Try to find a newline near the end to break cleanly
        newline_pos = content.rfind("\n", start + CHUNK_SIZE // 2, end)
        if newline_pos != -1:
            end = newline_pos + 1

        chunks.append(content[start:end])
        start = end

    return chunks
