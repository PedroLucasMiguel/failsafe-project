"""
Token usage tracker — a lightweight session-level singleton.

All LLM calls flow through this tracker so the CLI can display a running
total at any time. Thread-safe via a simple lock.

Usage:
    from failsafe.tracking import tracker

    # After an LLM call:
    tracker.record(response)

    # Read totals:
    tracker.total_tokens   # int
    tracker.input_tokens   # int
    tracker.output_tokens  # int
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class TokenTracker:
    """Session-scoped token usage accumulator."""

    input_tokens: int = 0
    output_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def record(self, response) -> None:
        """Extract and accumulate token counts from an LLM response.

        Compatible with LangChain's AIMessage which exposes usage_metadata.
        Silently no-ops if the response has no token info (e.g. streaming stubs).
        """
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            # Older LangChain versions may use response_metadata
            meta = getattr(response, "response_metadata", {})
            usage = meta.get("usage", meta.get("token_usage", {}))
            if not usage:
                return
            inp = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
            out = usage.get("completion_tokens", 0) or usage.get(
                "output_tokens", 0)
        else:
            inp = usage.get("input_tokens", 0)
            out = usage.get("output_tokens", 0)

        with self._lock:
            self.input_tokens += inp
            self.output_tokens += out

    def reset(self) -> None:
        """Reset all counters (call at session start)."""
        with self._lock:
            self.input_tokens = 0
            self.output_tokens = 0

    def summary_str(self) -> str:
        """Short human-readable summary, e.g. '1,234 tokens (↑512 ↓722)'."""
        return (
            f"{self.total_tokens:,} tokens "
            f"([dim]↑{self.input_tokens:,} in  ↓{self.output_tokens:,} out[/dim])"
        )


# Module-level singleton — import and use anywhere
tracker = TokenTracker()
