"""
Setup wizard — collects provider, API key, model, and codebase path.

Runs interactively using rich.prompt. Nothing is written to disk.
Returns a SessionConfig ready for the main menu.

Auto-detection:
    .env file is loaded on startup. When a provider is selected, the
    corresponding env var is checked first:
        openai    → OPENAI_API_KEY
        anthropic → ANTHROPIC_API_KEY  (also ANTROPHIC_API_KEY typo variant)
        google    → GOOGLE_API_KEY
    If found, the API key prompt is skipped entirely.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from failsafe.llm import PROVIDER_REGISTRY, SUPPORTED_PROVIDERS
from failsafe.ui.session import SessionConfig

_console = Console()

# Env var names per provider (first match wins)
_ENV_VARS: dict[str, list[str]] = {
    "openai":    ["OPENAI_API_KEY"],
    # cover the typo too
    "anthropic": ["ANTHROPIC_API_KEY", "ANTROPHIC_API_KEY"],
    "google":    ["GOOGLE_API_KEY"],
}


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------
def run_setup(current_config: SessionConfig | None = None) -> SessionConfig:
    """Run the interactive setup wizard and return a SessionConfig."""
    # Load .env from cwd (or any parent) — silently no-ops if not found
    load_dotenv(override=False)

    _print_header()
    _console.print()

    provider = _ask_provider(default=current_config.provider if current_config else "1")
    api_key = _resolve_api_key(provider)

    # If provider is the same as current, use current model as default
    model_default = None
    if current_config and current_config.provider == provider:
        model_default = current_config.model
    model = _ask_model(provider, default=model_default)

    codebase_path = _ask_codebase_path(default=current_config.codebase_path if current_config else ".")
    num_reviews = _ask_num_reviews(default=current_config.num_reviews if current_config else 1)

    _print_summary(provider, model, codebase_path, num_reviews)
    return SessionConfig(
        provider=provider,
        api_key=api_key,
        model=model,
        codebase_path=codebase_path,
        num_reviews=num_reviews,
    )


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def _ask_num_reviews(default: int = 1) -> int:
    """Ask for the number of code reviews to perform."""
    _console.print()
    _console.rule("[bold dim]Step 5 — Code Reviews[/bold dim]", style="dim")

    while True:
        answer = Prompt.ask(
            "  [bold]Number of reviews[/bold]",
            console=_console,
            default=str(default),
        ).strip()

        if answer.isdigit():
            return int(answer)

        _console.print("  [red]Invalid choice.[/red] Please enter a number.")


def _ask_provider(default: str = "1") -> str:
    """Ask which LLM provider to use, highlighting any that have a key in .env."""
    _console.rule("[bold dim]Step 1 — LLM Provider[/bold dim]", style="dim")

    # Check which providers already have a key loaded
    env_detected = {
        p for p in SUPPORTED_PROVIDERS if _get_env_key(p) is not None
    }

    table = Table(box=None, show_header=False, padding=(0, 2))
    defaults = {p: PROVIDER_REGISTRY[p][2] for p in SUPPORTED_PROVIDERS}
    for i, p in enumerate(SUPPORTED_PROVIDERS, 1):
        env_badge = " [green bold](.env)[/green bold]" if p in env_detected else ""
        table.add_row(
            f"[bold cyan][{i}][/bold cyan]",
            f"[white]{p}[/white]{env_badge}",
            f"[dim]default: {defaults[p]}[/dim]",
        )
    _console.print(table)
    _console.print()

    choices = {str(i): p for i, p in enumerate(SUPPORTED_PROVIDERS, 1)}
    choices.update({p: p for p in SUPPORTED_PROVIDERS})

    while True:
        answer = Prompt.ask(
            "  [bold]Provider[/bold]",
            console=_console,
            default=default,
        ).strip().lower()

        if answer in choices:
            chosen = choices[answer]
            _console.print(f"  [green]✓[/green] {chosen}")
            return chosen

        _console.print(
            f"  [red]Invalid choice.[/red] Enter 1–{len(SUPPORTED_PROVIDERS)} or a provider name.")


def _resolve_api_key(provider: str) -> str:
    """Return the API key — from .env if available, otherwise ask the user."""
    env_key = _get_env_key(provider)

    _console.print()
    _console.rule(
        f"[bold dim]Step 2 — {provider.capitalize()} API Key[/bold dim]", style="dim")

    if env_key:
        var_name = _which_env_var(provider)
        masked = env_key[:6] + "•" * min(8, len(env_key) - 6)
        _console.print(
            f"  [green]✓[/green] Loaded from [bold]{var_name}[/bold]: [dim]{masked}[/dim]"
        )
        return env_key

    key_hints = {
        "openai":    "https://platform.openai.com/api-keys",
        "anthropic": "https://console.anthropic.com/settings/keys",
        "google":    "https://aistudio.google.com/apikey",
    }
    _console.print(f"  [dim]Get a key at: {key_hints[provider]}[/dim]")
    _console.print(
        f"  [dim]Tip: add [bold]{_ENV_VARS[provider][0]}[/bold] to your .env to skip this step.[/dim]"
    )
    _console.print()

    while True:
        import getpass
        key = getpass.getpass("  API key: ").strip()
        if key:
            _console.print("  [green]✓[/green] API key received")
            return key
        _console.print("  [red]API key cannot be empty.[/red]")


def _ask_model(provider: str, default: str | None = None) -> str:
    """Ask for model name, defaulting to the provider's default."""
    _console.print()
    _console.rule("[bold dim]Step 3 — Model[/bold dim]", style="dim")

    _, _, provider_default = PROVIDER_REGISTRY[provider]
    default_model = default or provider_default

    _console.print("  [dim]Press Enter to use the default.[/dim]")
    _console.print()

    answer = Prompt.ask(
        "  [bold]Model[/bold]",
        console=_console,
        default=default_model,
    ).strip()

    model = answer if answer else default_model
    _console.print(f"  [green]✓[/green] {model}")
    return model


def _ask_codebase_path(default: str = ".") -> str:
    """Ask for the codebase directory path."""
    _console.print()
    _console.rule("[bold dim]Step 4 — Codebase Path[/bold dim]", style="dim")
    _console.print()

    while True:
        raw = Prompt.ask(
            "  [bold]Codebase directory[/bold]",
            console=_console,
            default=default,
        ).strip()

        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            _console.print(f"  [green]✓[/green] {path}")
            return str(path)

        _console.print(f"  [red]Not a directory:[/red] {path}")


# ---------------------------------------------------------------------------
# .env helpers
# ---------------------------------------------------------------------------
def _get_env_key(provider: str) -> str | None:
    """Return the first non-empty env var value for the given provider."""
    for var in _ENV_VARS.get(provider, []):
        value = os.environ.get(var, "").strip()
        if value:
            return value
    return None


def _which_env_var(provider: str) -> str:
    """Return the name of the env var that has a value for this provider."""
    for var in _ENV_VARS.get(provider, []):
        if os.environ.get(var, "").strip():
            return var
    return _ENV_VARS.get(provider, [""])[0]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def _print_header() -> None:
    _console.print()
    _console.print(Panel(
        Text.from_markup(
            "[bold bright_white]🛡  failsafe[/bold bright_white]\n"
            "[dim]Multi-agent codebase discovery[/dim]"
        ),
        border_style="bright_blue",
        padding=(1, 4),
        expand=False,
    ))


def _print_summary(provider: str, model: str, codebase_path: str, num_reviews: int) -> None:
    """Print a recap of what was configured."""
    _console.print()
    _console.print(Panel(
        f"[bold]Provider:[/bold]  [cyan]{provider}[/cyan]\n"
        f"[bold]Model:   [/bold]  [cyan]{model}[/cyan]\n"
        f"[bold]Path:    [/bold]  [cyan]{codebase_path}[/cyan]\n"
        f"[bold]Reviews: [/bold]  [cyan]{num_reviews}[/cyan]",
        title="Session Config",
        border_style="bright_blue",
        padding=(0, 2),
    ))
    _console.print()
