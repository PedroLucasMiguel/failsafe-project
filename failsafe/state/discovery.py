"""
Discovery state — shared TypedDict flowing through the discovery graph.

Every agent reads from and writes to this state. Fields use Annotated
reducers so parallel updates merge cleanly.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


def _replace(_, new):
    """Reducer that always takes the newest value."""
    return new


def _merge_dicts(old: dict | None, new: dict | None) -> dict:
    """Reducer that shallow-merges two dicts."""
    merged = dict(old or {})
    merged.update(new or {})
    return merged


class DiscoveryState(TypedDict):
    """Shared state for the discovery workflow.

    Attributes:
        codebase_path: Absolute path to the target codebase.
        user_context: Free-form info the user provided about their codebase.
        gitignore_patterns: Parsed gitignore patterns for the codebase.
        file_tree: Ordered list of relative file paths discovered.
        file_analyses: Mapping of relative_path → short summary string.
        knowledge_base: Final structured knowledge (set by Analyzer).
        messages: LangGraph message history (for agent ↔ LLM communication).
        current_phase: Which agent is currently running.
    """

    codebase_path: Annotated[str, _replace]
    user_context: Annotated[str, _replace]
    gitignore_patterns: Annotated[list[str], _replace]
    file_tree: Annotated[list[str], _replace]
    file_analyses: Annotated[dict[str, str], _merge_dicts]
    knowledge_base: Annotated[dict, _replace]
    messages: Annotated[list[BaseMessage], add_messages]
    current_phase: Annotated[str, _replace]
    vector_store_path: Annotated[str, _replace]  # set by indexer node
