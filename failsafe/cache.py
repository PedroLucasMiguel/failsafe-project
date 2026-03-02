"""
Session-scoped response cache for LLM calls.

Uses a content-hash of the input messages as the cache key.
Hits are served from memory — no disk I/O, no network round-trip.

Works with ALL providers (OpenAI, Anthropic, Google, etc.) since it
operates at the LangChain message level, not the provider API level.

Usage:
    from failsafe.cache import response_cache

    text = response_cache.get(messages)
    if text is None:
        response = llm.invoke(messages)
        text = extract_text(response)
        response_cache.set(messages, text)
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from typing import Sequence

from rich.console import Console

_console = Console()


@dataclass
class ResponseCache:
    """In-memory, content-addressed LLM response cache."""

    _store: dict[str, str] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    hits: int = 0
    misses: int = 0

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------
    def get(self, messages) -> str | None:
        """Return cached text for the given messages, or None on a miss."""
        key = self._key(messages)
        with self._lock:
            value = self._store.get(key)
            if value is not None:
                self.hits += 1
            else:
                self.misses += 1
            return value

    def set(self, messages, text: str) -> None:
        """Store the response text under the hash of the messages."""
        key = self._key(messages)
        with self._lock:
            self._store[key] = text

    def reset(self) -> None:
        """Clear the cache and reset counters (call at session start)."""
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0

    @property
    def size(self) -> int:
        return len(self._store)

    def summary_str(self) -> str:
        total = self.hits + self.misses
        rate = f"{100 * self.hits // total}%" if total else "n/a"
        return (
            f"cache: {self.hits} hits / {self.misses} misses "
            f"({rate} hit-rate, {self.size} entries)"
        )

    # ---------------------------------------------------------------------------
    # Internal
    # ---------------------------------------------------------------------------
    @staticmethod
    def _key(messages) -> str:
        """Produce a stable SHA-256 hash of message content."""
        parts = []
        for msg in messages:
            # Handle both LangChain message objects and dicts
            if hasattr(msg, "content"):
                role = type(msg).__name__
                content = msg.content
            else:
                role = msg.get("role", "")
                content = msg.get("content", "")

            # Normalise list content (Gemini blocks) to a string for hashing
            if isinstance(content, list):
                content = json.dumps(content, sort_keys=True)

            parts.append(f"{role}:{content}")

        raw = "\n---\n".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# Module-level singleton — import and use anywhere
response_cache = ResponseCache()
