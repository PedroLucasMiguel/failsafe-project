"""
Coding state — shared TypedDict for the coding agent workflow.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def _replace(_, new):
    return new


def _merge_dicts(old: dict | None, new: dict | None) -> dict:
    merged = dict(old or {})
    merged.update(new or {})
    return merged


class CodingState(TypedDict):
    """Shared state for the coding agent workflow.

    Attributes:
        task:              Natural language coding task from the user.
        codebase_path:     Absolute path to the target codebase.
        kb_context:        Pre-fetched KB context for the task.
        messages:          The tool-calling conversation loop.
        files_modified:    {rel_path: description} of all files changed.
        impact:            "minor" | "significant" — drives KB update strategy.
        summary:           Human-readable summary of what the agent did.
        vector_store_path: Path to the LanceDB store (for KB update).
        review_approved:   Whether the reviewer approved the changes.
        review_feedback:   Feedback from the reviewer if changes were rejected.
        review_suggestions: Suggested modifications from the reviewer.
    """

    task:              Annotated[str, _replace]
    codebase_path:     Annotated[str, _replace]
    kb_context:        Annotated[str, _replace]
    messages:          Annotated[list[BaseMessage], add_messages]
    files_modified:    Annotated[dict[str, str],
                                 _merge_dicts]  # path → description
    impact:            Annotated[str, _replace]
    summary:           Annotated[str, _replace]
    vector_store_path: Annotated[str, _replace]
    review_approved:   Annotated[bool, _replace]
    review_feedback:   Annotated[str, _replace]
    review_suggestions: Annotated[list[str], _replace]
