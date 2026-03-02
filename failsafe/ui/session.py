"""
SessionConfig — in-memory configuration for a single failsafe session.

Collected during the setup wizard and passed to every action in the main menu.
Nothing here is persisted to disk (by design for now).
"""

from __future__ import annotations

from dataclasses import dataclass

from failsafe.llm import PROVIDER_REGISTRY


@dataclass
class SessionConfig:
    """Holds all user-supplied settings for one run."""

    provider: str
    api_key: str
    model: str          # Always resolved (never None after setup)
    codebase_path: str  # Absolute path to the target directory

    @property
    def model_label(self) -> str:
        """Display string: 'gemini-2.0-flash (default)' or 'gpt-4o'."""
        _, _, default = PROVIDER_REGISTRY[self.provider]
        suffix = " [dim](default)[/dim]" if self.model == default else ""
        return f"{self.model}{suffix}"
