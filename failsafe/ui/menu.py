"""
Main menu — shown after setup completes.

Lists available actions and dispatches to the chosen one.
Loops until the user quits.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from failsafe.ui.session import SessionConfig

_console = Console()


# ---------------------------------------------------------------------------
# Menu items registry
# ---------------------------------------------------------------------------
_MENU_ITEMS: list[tuple[str, str]] = [
    ("1", "Run Discovery  [dim](full analysis + index)[/dim]"),
    ("2", "Sync KB        [dim](fast, after git pull)[/dim]"),
    ("3", "Sync KB (full) [dim](rebuild all vectors from committed KB)[/dim]"),
    ("4", "Code           [dim](run a coding agent task)[/dim]"),
    ("5", "Settings       [dim](reconfigure provider, model, etc.)[/dim]"),
    ("Q", "Quit"),
]


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------
def run_menu(config: SessionConfig) -> None:
    """Display the main menu and dispatch actions until the user quits."""
    while True:
        _print_menu(config)
        choice = Prompt.ask(
            "  [bold]Choose an action[/bold]",
            console=_console,
        ).strip().upper()

        _console.print()

        if choice in ("Q", "QUIT", "EXIT"):
            _console.print("[dim]Goodbye.[/dim]\n")
            break
        elif choice == "1":
            _run_discover(config)
        elif choice == "2":
            _run_sync(config, full=False)
        elif choice == "3":
            _run_sync(config, full=True)
        elif choice == "4":
            _run_code(config)
        elif choice == "5":
            from failsafe.ui.setup import run_setup
            config = run_setup(current_config=config)
        else:
            _console.print("[red]Unknown option.[/red] Please try again.\n")


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
def _run_discover(config: SessionConfig) -> None:
    """Run the full discovery pipeline with the current session config."""
    from failsafe.graph.discovery import build_discovery_graph
    from failsafe.knowledge.store import KnowledgeStore
    from failsafe.knowledge.vector_store import VectorStore
    from failsafe.llm import create_llm, set_llm
    from failsafe.tracking import tracker

    _console.print(Panel(
        f"[bold]Failsafe Discovery[/bold]\n"
        f"Codebase: [cyan]{config.codebase_path}[/cyan]\n"
        f"Provider: [cyan]{config.provider}[/cyan]\n"
        f"Model:    [cyan]{config.model_label}[/cyan]",
        title="🔍 Starting",
        border_style="bright_blue",
    ))

    try:
        llm = create_llm(
            provider=config.provider,
            api_key=config.api_key,
            model=config.model,
        )
    except (ValueError, RuntimeError) as exc:
        _console.print(f"[red]Error:[/red] {exc}")
        return

    set_llm(llm)
    tracker.reset()
    from failsafe.cache import response_cache
    response_cache.reset()

    graph = build_discovery_graph()

    _console.print("\n[bold cyan]Phase 1/4:[/bold cyan] Interview")
    _console.print(
        "[dim]Answer a few questions (or press Enter to skip).[/dim]\n")

    initial_state = {
        "codebase_path": config.codebase_path,
        "user_context": "",
        "gitignore_patterns": [],
        "file_tree": [],
        "file_analyses": {},
        "knowledge_base": {},
        "messages": [],
        "current_phase": "starting",
        "vector_store_path": "",
    }

    final_state = None
    for event in graph.stream(initial_state, stream_mode="updates"):
        for _node, update in event.items():
            phase = update.get("current_phase", "")
            if phase == "interview_done":
                _console.print(
                    f"  [dim]🪙 Tokens so far: {tracker.summary_str()}[/dim]"
                )
                _console.print(
                    "\n[bold cyan]Phase 2/4:[/bold cyan] Exploring codebase..."
                )
            elif phase == "exploration_done":
                n = len(update.get("file_tree", []))
                _console.print(f"[dim]  Found {n} files.[/dim]")
                _console.print(
                    "\n[bold cyan]Phase 3/4:[/bold cyan] Analyzing and building knowledge base..."
                )
            elif phase == "analysis_done":
                final_state = update
                _console.print(
                    f"  [dim]🪙 Total tokens: {tracker.summary_str()}[/dim]"
                )
                _console.print(
                    "\n[bold cyan]Phase 4/4:[/bold cyan] Indexing into vector store..."
                )

    if final_state is None:
        _console.print(
            "[red]Discovery failed — no final state produced.[/red]")
        return

    kb_data = final_state.get("knowledge_base", {})
    store = KnowledgeStore()
    store.load_from_dict(kb_data)

    # Save KB JSON (committed to git — shared with team)
    output_path = str(Path(config.codebase_path) / "failsafe_kb.json")
    store.save(output_path)

    # Attach vector store if indexer ran
    vs_path = final_state.get("vector_store_path", "")
    if vs_path:
        vs = VectorStore(vs_path)
        store.attach_vector_store(vs)
        # Save sync state so future `failsafe sync` knows the last indexed commit
        from failsafe.sync import _save_sync_state, _get_head_commit
        current_commit = _get_head_commit(Path(config.codebase_path))
        _save_sync_state(Path(config.codebase_path),
                         current_commit, vs.chunk_count)

    _console.print()
    _console.print(Panel(
        store.summary(),
        title="✅ Knowledge Base Generated",
        border_style="green",
    ))
    _console.print(f"\n[dim]Saved to:[/dim] [bold]{output_path}[/bold]")
    _console.print(
        "[dim]Tip: commit failsafe_kb.json so teammates can sync instantly.[/dim]\n")


def _run_code(config: SessionConfig) -> None:
    """Run a coding agent task against the current codebase."""
    import json

    from failsafe.agents import CodingAgent, ReviewerAgent
    from failsafe.knowledge.store import KnowledgeStore
    from failsafe.knowledge.vector_store import VectorStore
    from failsafe.llm import create_llm, set_llm
    from failsafe.tracking import tracker

    codebase_path = Path(config.codebase_path)
    kb_json = codebase_path / "failsafe_kb.json"

    if not kb_json.exists():
        _console.print(
            "[red]No KB found.[/red] Run [bold][1] Run Discovery[/bold] first.\n"
        )
        return

    _console.print(Panel(
        f"[bold]Coding Agent[/bold]\n"
        f"Codebase: [cyan]{config.codebase_path}[/cyan]\n"
        f"Provider: [cyan]{config.provider}[/cyan] · [cyan]{config.model_label}[/cyan]\n"
        f"Reviews:  [cyan]{config.num_reviews}[/cyan]",
        title="🤖 Code",
        border_style="bright_blue",
    ))

    task = Prompt.ask(
        "  [bold]Describe the coding task[/bold]",
        console=_console,
    ).strip()
    if not task:
        _console.print("[dim]No task entered. Returning to menu.[/dim]\n")
        return

    _console.print()

    try:
        llm = create_llm(
            provider=config.provider,
            api_key=config.api_key,
            model=config.model,
        )
    except (ValueError, RuntimeError) as exc:
        _console.print(f"[red]Error:[/red] {exc}")
        return

    set_llm(llm)
    tracker.reset()

    # Load KB and vector store
    kb_store = KnowledgeStore()
    kb_store.load_from_dict(json.loads(kb_json.read_text(encoding="utf-8")))

    vs = None
    vs_path = codebase_path / ".failsafe" / "vectors"
    if vs_path.exists():
        vs = VectorStore(vs_path)
        kb_store.attach_vector_store(vs)

    agent = CodingAgent(
        llm=llm,
        kb_store=kb_store,
        vector_store=vs,
        codebase_path=codebase_path,
    )
    reviewer = ReviewerAgent(llm=llm)

    _console.print("[dim]Agent is working...[/dim]\n")

    current_task = task
    for i in range(config.num_reviews + 1):
        if i > 0:
            _console.print(f"\n[bold yellow]Review Cycle {i}/{config.num_reviews}[/bold yellow]")
        
        result = agent.run(current_task)
        
        if i == config.num_reviews:
            # Last iteration, no more reviews
            break
            
        _console.print("\n[dim]Reviewing changes...[/dim]")
        review_result = reviewer.review(task, result.files_modified, codebase_path)
        
        if review_result.is_approved:
            _console.print("[green]✓ Code approved by reviewer.[/green]")
            break
        else:
            _console.print(f"[yellow]✗ Reviewer feedback:[/yellow]\n{review_result.feedback}")
            if review_result.suggestions:
                _console.print("[dim]Suggestions:[/dim]")
                for s in review_result.suggestions:
                    _console.print(f"  - {s}")
            
            # Update task for next iteration
            current_task = (
                f"Original task: {task}\n\n"
                f"Previous attempt resulted in these changes: {result.summary}\n\n"
                f"The reviewer rejected the changes with the following feedback:\n"
                f"{review_result.feedback}\n\n"
                f"Please address the feedback and fix the issues."
            )

    # Display final result
    impact_color = "yellow" if result.impact == "significant" else "green"
    files_lines = [
        f"  [dim]{path}[/dim]: {desc}"
        for path, desc in result.files_modified.items()
    ] or ["  [dim](no files modified)[/dim]"]

    _console.print(Panel(
        f"{result.summary}\n\n"
        f"[bold]Files changed ({len(result.files_modified)}):[/bold]\n"
        + "\n".join(files_lines)
        + f"\n\n[bold]Impact:[/bold] [{impact_color}]{result.impact}[/{impact_color}] "
          f"→ KB {'updated (full sync)' if result.impact == 'significant' else 'patched (fast)'}\n"
          f"[dim]🪙 Tokens: {tracker.summary_str()}[/dim]"
        + (f"\n[yellow]{result.error}[/yellow]" if result.error else ""),
        title="✅ Task Complete",
        border_style="green",
    ))
    _console.print()


def _run_sync(config: SessionConfig, full: bool = False) -> None:
    """Sync local vector store from committed KB (fast, no full re-analysis)."""
    from failsafe.sync import run_sync
    from failsafe.llm import create_llm, set_llm

    mode_label = "Full rebuild" if full else "Incremental sync"
    _console.print(Panel(
        f"[bold]{mode_label}[/bold]\n"
        f"Codebase: [cyan]{config.codebase_path}[/cyan]",
        title="🔄 Sync",
        border_style="bright_blue",
    ))

    # Provide LLM for new file summarization (accurate path)
    try:
        llm = create_llm(
            provider=config.provider,
            api_key=config.api_key,
            model=config.model,
        )
        set_llm(llm)
    except (ValueError, RuntimeError):
        llm = None  # Will embed new files raw without summarization

    run_sync(config.codebase_path, full=full, llm=llm)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def _print_menu(config: SessionConfig) -> None:
    """Render the main menu panel."""
    table = Table(box=None, show_header=False, padding=(0, 2))
    for key, label in _MENU_ITEMS:
        style = "bold red" if key == "Q" else "bold cyan"
        table.add_row(
            Text(f"[{key}]", style=style),
            Text.from_markup(label),
        )

    _console.print(Panel(
        table,
        title="[bold]Main Menu[/bold]",
        subtitle=f"[dim]{config.provider} · {config.model_label} · {config.codebase_path}[/dim]",
        border_style="bright_blue",
        padding=(0, 2),
    ))
