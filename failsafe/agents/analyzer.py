"""
Analyzer agent — synthesizes exploration results into a structured knowledge base.

Pipeline split:
  1. FILE ANALYSES are built directly in Python from the explorer's structured
     summaries — no LLM needed for this. The explorer already extracted purpose,
     key elements, dependencies, and patterns per file.

  2. The LLM is called ONCE to perform the harder synthesis tasks only:
     - group files into logical subsystems
     - identify cross-cutting coding patterns WITH real code snippets
     - identify proprietary/custom technologies
     - write architecture notes
     - derive project name, summary, tech stack

This division means:
  - file_analyses is always fully populated (one entry per file), not truncated
    by output token limits
  - The LLM's output budget is spent on synthesis, not on re-deriving what the
    explorer already computed
  - Code snippets in patterns are extracted from the rich summaries, not invented
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from failsafe.state.discovery import DiscoveryState
from failsafe.llm import invoke_cached


# ---------------------------------------------------------------------------
# System prompt — synthesis only (file_analyses already built in Python)
# ---------------------------------------------------------------------------
SYNTHESIS_SYSTEM_PROMPT = """\
You are a senior software architect. You are given per-file structured \
summaries of a codebase (purpose, key elements, dependencies, patterns \
observed). Your task is to synthesize this into high-level architectural \
knowledge.

You MUST output valid JSON matching exactly this schema:

{
  "project_name": "<string>",
  "summary": "<2-3 sentences: what the project does and its main value>",
  "tech_stack": ["<language/framework/library>", ...],
  "subsystems": [
    {
      "name": "<short name>",
      "description": "<what this logical group does>",
      "files": ["<relative_path>", ...]
    }
  ],
  "coding_patterns": [
    {
      "name": "<specific pattern name, e.g. 'LangChain @tool decorator'>",
      "description": "<how and why this pattern is used in this codebase>",
      "when_to_use": "<actionable guidance: when a coding agent should apply this>",
      "code_snippets": [
        {
          "file": "<relative_path>",
          "label": "<what this snippet demonstrates>",
          "language": "<python|typescript|go|etc>",
          "code": "<short verbatim code excerpt from the patterns/elements you observed>"
        }
      ],
      "examples": ["<relative_path>", ...]
    }
  ],
  "proprietary_tech": [
    {
      "name": "<name>",
      "description": "<what it does and how it is used>",
      "related_files": ["<relative_path>", ...]
    }
  ],
  "architecture_notes": "<3-5 sentences: overall structure, data flow, key design decisions>"
}

IMPORTANT rules:
- Identify REAL, SPECIFIC patterns — not generic ones (not "uses classes").
  Good: "LangGraph StateGraph node pattern", "Module-level singleton with reset()",
  "Budget-aware batch packing", "Provider registry with lazy imports".
- code_snippets: reconstruct concise representative snippets from the key
  elements and patterns you were given. Keep them short (5-20 lines).
- Ensure ALL files appear in at least one subsystem.
- tech_stack: list concrete items (Python 3.12, LangGraph, Pydantic v2, Rich, etc.)
- Output ONLY the JSON — no markdown, no explanation.
"""


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------
def analyzer_node(state: DiscoveryState) -> dict:
    """Build the knowledge base:
    1. file_analyses — constructed in Python from explorer summaries.
    2. LLM synthesis — subsystems, patterns, proprietary tech, architecture.
    """
    from failsafe.llm import get_llm
    llm = get_llm()

    user_context = state.get("user_context", "No user context provided.")
    raw_file_analyses: dict[str, str] = state.get("file_analyses", {})
    file_tree: list[str] = state.get("file_tree", [])

    # 1. Build file_analyses KB entries directly from explorer output (no LLM)
    file_analyses_kb = _build_file_analyses(file_tree, raw_file_analyses)

    # 2. Format structured summaries for the synthesis prompt
    summary_block = _format_summaries_for_synthesis(
        file_tree, raw_file_analyses)

    synthesis_prompt = (
        f"## User-provided context\n{user_context}\n\n"
        f"## Per-file structured summaries ({len(file_tree)} files)\n\n"
        f"{summary_block}\n\n"
        f"Now produce the synthesis JSON."
    )

    raw = invoke_cached(llm, [
        SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
        HumanMessage(content=synthesis_prompt),
    ])

    if raw.startswith("```"):
        raw = _strip_code_fences(raw)

    try:
        kb_dict = json.loads(raw)
    except json.JSONDecodeError:
        kb_dict = {
            "project_name": "Unknown",
            "summary": "Synthesis failed to produce valid JSON.",
            "tech_stack": [],
            "subsystems": [],
            "coding_patterns": [],
            "proprietary_tech": [],
            "architecture_notes": raw[:2000],
        }

    # 3. Inject the fully populated file_analyses (overrides whatever LLM returned)
    kb_dict["file_analyses"] = file_analyses_kb

    return {
        "knowledge_base": kb_dict,
        "current_phase": "analysis_done",
    }


# ---------------------------------------------------------------------------
# File analyses builder — pure Python, no LLM
# ---------------------------------------------------------------------------
def _build_file_analyses(
    file_tree: list[str],
    raw_summaries: dict[str, str],
) -> list[dict]:
    """Build file_analyses KB entries from the explorer's rich summaries.

    Parses the multi-line structured summary strings into structured dicts.
    Falls back gracefully for any file with missing or plain-text summaries.
    """
    entries = []

    for path in file_tree:
        summary = raw_summaries.get(path, "")
        if not summary:
            continue

        lines = summary.splitlines()
        purpose = lines[0].strip() if lines else "No summary available."
        key_elements: list[str] = []
        dependencies: list[str] = []

        for line in lines[1:]:
            low = line.lower()
            if low.startswith("key elements:"):
                key_elements = [e.strip() for e in line.split(":", 1)[
                    1].split(",") if e.strip()]
            elif low.startswith("dependencies:"):
                dependencies = [d.strip() for d in line.split(":", 1)[
                    1].split(",") if d.strip()]

        entries.append({
            "path": path,
            "purpose": purpose,
            "key_elements": key_elements[:10],    # cap at 10 for readability
            "dependencies": dependencies[:10],
        })

    return entries


def _format_summaries_for_synthesis(
    file_tree: list[str],
    raw_summaries: dict[str, str],
) -> str:
    """Format per-file summaries as a compact block for the synthesis prompt."""
    blocks = []
    for path in file_tree:
        summary = raw_summaries.get(path, "no summary")
        blocks.append(f"### {path}\n{summary}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` fences from model output."""
    lines = text.splitlines()
    # Drop the first and last fence lines
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()
