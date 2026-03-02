"""
CodingAgent — LLM-powered agent that reads the KB, edits the codebase,
and keeps the knowledge base current after every change.

Architecture
------------
The agent runs a ReAct tool-calling loop:

  1. Fetch task context from KB (semantic search + get_context_for_task)
  2. Build initial messages: [system_prompt, kb_context, user_task]
  3. LLM decides which tools to call (bind_tools)
  4. Execute tool calls → append results → loop
  5. When LLM emits no tool calls: extract structured final answer
  6. Parse CodingResult (files_modified, impact, summary)
  7. Apply two-tier KB update:
       minor      → patch KB JSON file_analyses + re-embed vectors
       significant → same + run_sync(full=True) to rebuild all vectors

The agent self-assesses "minor vs significant":
  - minor: bug fix, refactor, adding a function/docstring, editing existing file
  - significant: new file created, multiple subsystems touched, new API/pattern added
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from failsafe.tools.code_edit import CODING_TOOLS, ToolContext, set_tool_context


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CodingResult:
    """Structured output of a completed coding agent run."""
    summary: str
    files_modified: dict[str, str]   # {rel_path: description}
    impact: str                       # "minor" | "significant"
    kb_updated: bool = False
    error: str = ""


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

CODER_SYSTEM_PROMPT = """\
You are an expert coding agent. You have a KB and tools to read/write files.

## TOOL USAGE — STRICT RULES

### Finding code (research, ≤2 iterations total)
- **grep_file is your primary research tool.** Call it with the function/class/variable
  name to instantly get exact line numbers AND surrounding context. Default context_lines=20
  shows most full function bodies in one call.
- After grep_file, if the full block you need to change is already visible in the output:
  **call patch_file DIRECTLY** — do NOT call read_file_section.
- Use read_file_section ONLY if grep_file shows you the location but not enough body
  (e.g., a 100-line function). Then read exactly that range once.
- Use read_file only for small files (<80 lines).
- Use search_kb at most ONCE for orientation. Use get_file_context for file metadata.
- **NEVER call list_directory.** The KB context already tells you which files exist.
  Only use it if the task explicitly asks you to audit the directory structure.

### Making changes (action, immediately after research)
- Large files (>80 lines): use **patch_file(path, old_text, new_text)**.
  - old_text must match EXACTLY — copy it verbatim from grep_file or read_file_section output.
  - Include enough surrounding lines (3-5) so old_text is unique in the file.
- Small files or new files: use **write_file(path, full_content)**.
- Do NOT re-read a file after patching. Do NOT search again.

### Ideal flow (3 iterations for most tasks)
  iter 1: grep_file("path/to/file.py", "def target_function")
  iter 2: patch_file("path/to/file.py", old_text, new_text)   ← patch directly from grep output
  iter 3: emit final JSON answer

### Final answer — emit immediately after all edits, no preamble:
{
  "summary": "<1-3 sentences of what you did>",
  "files_modified": {"<rel_path>": "<one-sentence description>"},
  "impact": "minor" | "significant"
}

Impact: minor = edited existing files only. significant = new file or public API changed.
Match the coding style you observe.
"""


# ---------------------------------------------------------------------------
# CodingAgent
# ---------------------------------------------------------------------------

class CodingAgent:
    """ReAct coding agent with KB-aware context and two-tier KB update."""

    MAX_ITERATIONS = 20  # prevent infinite loops

    def __init__(
        self,
        llm,
        kb_store,               # KnowledgeStore
        vector_store,           # VectorStore | None
        codebase_path: str | Path,
        kb_json_path: str | Path | None = None,
    ) -> None:
        self._llm = llm
        self._kb_store = kb_store
        self._vs = vector_store
        self._codebase_path = Path(codebase_path)
        self._kb_json_path = Path(kb_json_path) if kb_json_path else (
            self._codebase_path / "failsafe_kb.json"
        )

    def run(
        self,
        task: str,
        feedback: str | None = None,
        suggestions: list[str] | None = None,
    ) -> CodingResult:
        """Run a coding task end-to-end.

        Args:
            task: Natural language description of the task.
            feedback: Optional feedback from a reviewer.
            suggestions: Optional suggestions from a reviewer.

        Returns:
            CodingResult with summary, files_modified, impact, and kb_updated.
        """
        # 1. Set up tool context (shared across all tools)
        ctx = ToolContext(
            codebase_path=self._codebase_path,
            kb_store=self._kb_store,
            vector_store=self._vs,
        )
        set_tool_context(ctx)

        # 2. Fetch KB context for this task
        # compact=True: metadata hints only, no raw code → agent uses search_kb on demand
        kb_context = self._kb_store.get_context_for_task(task, compact=True)

        # 3. Build initial messages
        human_content = (
            f"## Knowledge Base Context\n{kb_context}\n\n"
            f"## Task\n{task}"
        )
        if feedback:
            human_content += f"\n\n## Reviewer Feedback\n{feedback}"
        if suggestions:
            human_content += f"\n\n## Reviewer Suggestions\n" + "\n".join(
                f"- {s}" for s in suggestions)

        messages = [
            SystemMessage(content=CODER_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        # 4. Bind tools to LLM for structured tool calling
        llm_with_tools = self._llm.bind_tools(CODING_TOOLS)

        # 5. ReAct loop
        tool_map = {t.name: t for t in CODING_TOOLS}
        iterations = 0

        from rich.console import Console
        from failsafe.tracking import tracker
        _con = Console()

        # Deduplication cache: (tool_name, frozen_args) → result
        # If agent calls same tool with same args again, return cached result
        # without adding new context to the message history.
        _tool_cache: dict[tuple, str] = {}
        # How many of the most-recent AI+tool message pairs to keep in context.
        # Indices 0 (system) and 1 (initial task) are always kept; everything
        # beyond MAX_HISTORY_MSGS older messages is dropped to bound token usage.
        MAX_HISTORY_MSGS = 12      # 6 AI turns + 6 tool-result turns
        MAX_RESULT_CHARS = 2_000   # truncate long tool outputs to keep context small

        while iterations < self.MAX_ITERATIONS:
            iterations += 1

            # LLM decides what to do next
            response = self._invoke_with_tools(
                llm_with_tools, messages, max_history=MAX_HISTORY_MSGS)
            messages.append(response)

            # No tool calls → agent is done, print final token count
            if not response.tool_calls:
                _con.print(f"  [dim]🪙 {tracker.summary_str()}[/dim]")
                break

            # Show which tools are being called + cumulative token count
            tool_names = ", ".join(tc["name"] for tc in response.tool_calls)
            _con.print(
                f"  [dim]iter {iterations} · {tool_names} · "
                f"🪙 {tracker.summary_str()}[/dim]"
            )

            # Execute all tool calls (with deduplication + truncation)
            for tc in response.tool_calls:
                tool_fn = tool_map.get(tc["name"])
                if tool_fn is None:
                    result = f"ERROR: Unknown tool '{tc['name']}'"
                else:
                    # Build a hashable cache key from the tool name + args
                    try:
                        cache_key = (tc["name"], tuple(
                            sorted(tc["args"].items())))
                    except Exception:  # noqa: BLE001
                        cache_key = None

                    if cache_key and cache_key in _tool_cache:
                        # Return cached result — no new context added
                        result = f"[cached] {_tool_cache[cache_key]}"
                    else:
                        try:
                            result = tool_fn.invoke(tc["args"])
                        except Exception as exc:  # noqa: BLE001
                            result = f"ERROR: Tool {tc['name']} failed: {exc}"

                        result = str(result)
                        # Truncate very long results to keep context bounded
                        if len(result) > MAX_RESULT_CHARS:
                            result = result[:MAX_RESULT_CHARS] + \
                                "\n...[truncated]"

                        if cache_key:
                            _tool_cache[cache_key] = result

                messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tc["id"],
                ))

        # 6. Extract final answer from last AI message
        last_ai = next(
            (m for m in reversed(messages) if isinstance(
                m, AIMessage) and not m.tool_calls),
            None,
        )
        raw_answer = last_ai.content if last_ai else ""

        result = self._parse_result(raw_answer)

        # 7. Merge tool-tracked file writes with LLM-reported files_modified
        for path in ctx.files_written:
            if path not in result.files_modified:
                result.files_modified[path] = "Modified by coding agent"

        # 8. Apply KB update
        if result.files_modified:
            self._apply_kb_update(result)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _invoke_with_tools(self, llm_with_tools, messages: list, max_history: int = 12):
        """Invoke LLM with tool-calling support, applying a sliding history window.

        To prevent unbounded context growth across many iterations, we keep:
          - messages[0]: system prompt (always)
          - messages[1]: initial HumanMessage with task + KB context (always)
          - messages[-max_history:]: the most recent AI/tool-result pairs

        Older intermediate tool results are dropped. The _tool_cache ensures
        the agent won't re-invoke tools it has already called with the same args.

        (Bypasses invoke_cached since tool-call responses have structured
        metadata not suitable for caching.)
        """
        from failsafe.tracking import tracker
        from failsafe.llm import _apply_provider_caching

        # Pin the first two messages; slide a window over the rest
        if len(messages) > 2 + max_history:
            windowed = messages[:2] + messages[-max_history:]
        else:
            windowed = messages

        provider_messages = _apply_provider_caching(self._llm, windowed)
        response = llm_with_tools.invoke(provider_messages)
        tracker.record(response)
        return response

    def _parse_result(self, raw) -> CodingResult:
        """Parse the agent's structured JSON final answer."""
        # Normalize content: Gemini returns list[dict], OpenAI/Anthropic return str
        if isinstance(raw, list):
            parts = []
            for block in raw:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    parts.append(block.get("text", ""))
            raw = "".join(parts)
        raw = str(raw).strip()

        # Strip markdown fences if model wrapped the JSON
        if raw.startswith("```"):
            lines = raw.splitlines()
            lines = [l for l in lines if not l.startswith("```")]
            raw = "\n".join(lines).strip()

        try:
            data = json.loads(raw)
            return CodingResult(
                summary=data.get("summary", "Task completed."),
                files_modified=data.get("files_modified", {}),
                impact=data.get("impact", "minor"),
            )
        except (json.JSONDecodeError, ValueError):
            # Fallback: treat entire output as summary, impact unknown
            return CodingResult(
                summary=raw[:500] if raw else "Task completed.",
                files_modified={},
                impact="minor",
                error="Could not parse structured output — falling back.",
            )

    def _apply_kb_update(self, result: CodingResult) -> None:
        """Apply the two-tier KB update based on result.impact.

        Minor:      patch file_analyses in JSON + re-embed changed files
        Significant: same + run_sync(full=True) to rebuild ALL vectors
        """
        from failsafe.knowledge.patcher import patch_kb_json
        from failsafe.knowledge.updater import kb_update

        # Always: patch KB JSON file_analyses entries
        if self._kb_json_path.exists():
            patch_kb_json(self._kb_json_path, result.files_modified)

        # Always: fast vector re-index for changed files
        if self._vs is not None:
            kb = self._kb_store.knowledge_base
            kb_update(
                changed_files=list(result.files_modified.keys()),
                codebase_path=self._codebase_path,
                kb=kb,
                vector_store=self._vs,
                agent_descriptions=result.files_modified,  # fast path: 0 LLM calls
            )

        # Significant: also rebuild all vectors from the now-updated KB
        if result.impact == "significant" and self._vs is not None:
            from failsafe.sync import run_sync
            run_sync(self._codebase_path, full=True)

        result.kb_updated = True
