"""
Interaction tools — CLI-based communication with the user during agent runs.

Uses `rich` for pretty prompts so the terminal experience is polished.
"""

from __future__ import annotations

from langchain_core.tools import tool
from rich.console import Console
from rich.prompt import Prompt

_console = Console()


@tool
def ask_user(question: str) -> str:
    """Ask the user a question and return their answer.

    Args:
        question: The question to display to the user.

    Returns:
        The user's text response, or '[skipped]' if they press Enter.
    """
    _console.print()
    answer = Prompt.ask(f"[bold cyan]🤖 Agent[/bold cyan]  {question}")
    return answer.strip() or "[skipped]"


@tool
def ask_user_permission(path: str, reason: str) -> bool:
    """Ask the user for permission to access a gitignored path.

    Args:
        path: The path requiring permission.
        reason: Why the agent wants to access it.

    Returns:
        True if the user grants permission, False otherwise.
    """
    _console.print()
    _console.print(f"[bold yellow]⚠  Permission needed[/bold yellow]")
    _console.print(f"   Path:   [dim]{path}[/dim]")
    _console.print(f"   Reason: {reason}")

    answer = Prompt.ask(
        "   Allow access? [y/n]",
        choices=["y", "n"],
        default="n",
    )
    return answer.lower() == "y"
