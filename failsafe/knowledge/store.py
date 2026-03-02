"""
Knowledge store — in-memory access layer over the KnowledgeBase model.

Provides query helpers so other agents can look up info without parsing
the raw dict. Also handles JSON serialization for persistence.

Search strategy for coding agents:
  1. search_patterns(keyword)   — finds patterns by name/description keyword
  2. get_context_for_task(desc) — keyword search across ALL kb fields, returns
                                   a compact context block ready to inject into
                                   an LLM prompt
  3. find_snippets(keyword)     — searches code_snippets across all patterns
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from failsafe.knowledge.vector_store import VectorStore

from failsafe.knowledge.models import KnowledgeBase


class KnowledgeStore:
    """In-memory wrapper around a KnowledgeBase with query helpers."""

    def __init__(self, kb: KnowledgeBase | None = None) -> None:
        self._kb = kb or KnowledgeBase()
        self._vector_store: "VectorStore | None" = None

    # ------------------------------------------------------------------
    # Vector store attachment
    # ------------------------------------------------------------------

    def attach_vector_store(self, vs: "VectorStore") -> None:
        """Attach a VectorStore for semantic search.

        When attached, get_context_for_task() and search_semantic() use
        vector similarity instead of keyword matching.
        """
        self._vector_store = vs

    def search_semantic(self, query: str, n: int = 5) -> list[dict]:
        """Semantic similarity search over indexed code chunks.

        Args:
            query: Natural language task description or code query.
            n:     Number of results to return.

        Returns:
            List of {file, chunk_index, subsystem, code, _distance} dicts.
            Empty list if no vector store is attached.
        """
        if self._vector_store is None:
            return []
        return self._vector_store.search(query, n=n)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def knowledge_base(self) -> KnowledgeBase:
        return self._kb

    def load_from_dict(self, data: dict) -> None:
        """Replace the internal knowledge base from a raw dict."""
        self._kb = KnowledgeBase.model_validate(data)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_subsystem_for_file(self, file_path: str) -> str | None:
        """Return the subsystem name that contains the given file, or None."""
        for subsystem in self._kb.subsystems:
            if file_path in subsystem.files:
                return subsystem.name
        return None

    def get_file_analysis(self, file_path: str) -> dict | None:
        """Return the analysis dict for a specific file, or None."""
        for fa in self._kb.file_analyses:
            if fa.path == file_path:
                return fa.model_dump()
        return None

    def search_patterns(self, keyword: str) -> list[dict]:
        """Return coding patterns whose name or description match a keyword."""
        keyword_lower = keyword.lower()
        return [
            p.model_dump()
            for p in self._kb.coding_patterns
            if keyword_lower in p.name.lower()
            or keyword_lower in p.description.lower()
        ]

    def find_snippets(self, keyword: str) -> list[dict]:
        """Search for code snippets across all coding patterns.

        Returns a list of dicts with the pattern name, snippet label, and code.
        Useful for a coding agent asking "show me how X is done in this codebase".

        Args:
            keyword: Searched in pattern name, description, snippet label, and code.

        Returns:
            List of dicts: {pattern, file, label, language, code}
        """
        keyword_lower = keyword.lower()
        results = []

        for pattern in self._kb.coding_patterns:
            pattern_matches = (
                keyword_lower in pattern.name.lower()
                or keyword_lower in pattern.description.lower()
            )
            for snippet in pattern.code_snippets:
                snippet_matches = (
                    keyword_lower in snippet.label.lower()
                    or keyword_lower in snippet.code.lower()
                    or keyword_lower in snippet.file.lower()
                )
                if pattern_matches or snippet_matches:
                    results.append({
                        "pattern": pattern.name,
                        "when_to_use": pattern.when_to_use,
                        "file": snippet.file,
                        "label": snippet.label,
                        "language": snippet.language,
                        "code": snippet.code,
                    })

        return results

    def get_context_for_task(
        self,
        task_description: str,
        compact: bool = True,
    ) -> str:
        """Build a context block for a coding agent working on a task.

        When a VectorStore is attached (via attach_vector_store), uses semantic
        similarity search to find the most relevant code chunks. Falls back to
        keyword search on KB patterns and subsystems.

        Args:
            task_description: What the coding agent is about to implement.
            compact: When True (default), returns lightweight metadata only —
                no raw source code, just file paths, subsystems, and purposes.
                The agent can fetch full code on demand via the search_kb tool.
                When False, includes full raw code chunks (expensive).

        Returns:
            A formatted string with relevant KB context, ready to inject into
            an LLM prompt.
        """
        sections: list[str] = []

        # Project overview (always included)
        sections.append(
            f"## Project: {self._kb.project_name}\n{self._kb.summary}"
        )

        # --- Semantic search path (preferred when vector store is available) ---
        if self._vector_store is not None:
            # compact=True: 3 results, metadata only; compact=False: 6 results + full code
            n_results = 3 if compact else 6
            semantic_results = self._vector_store.search(
                task_description, n=n_results)
            if semantic_results:
                if compact:
                    # Only show file path + purpose extracted from embed_text header
                    # Raw code is intentionally omitted — agent uses search_kb tool
                    lines = [
                        "## Relevant Files (use search_kb tool for full code)"
                    ]
                    seen_files: set[str] = set()
                    for r in semantic_results:
                        if r["file"] in seen_files:
                            continue
                        seen_files.add(r["file"])
                        purpose = _extract_purpose_from_embed(
                            r.get("embed_text", ""))
                        label = r["file"]
                        if r.get("subsystem"):
                            label += f" [{r['subsystem']}]"
                        if purpose:
                            label += f" — {purpose}"
                        lines.append(f"- `{label}`")
                    sections.append("\n".join(lines))
                else:
                    lines = ["## Relevant Code (semantic search)"]
                    seen_files_full: set[str] = set()
                    for r in semantic_results:
                        from failsafe.knowledge.vector_store import _guess_language
                        lang = _guess_language(r["file"])
                        file_label = r["file"]
                        if r.get("subsystem"):
                            file_label += f" [{r['subsystem']}]"
                        if r["file"] not in seen_files_full:
                            seen_files_full.add(r["file"])
                        lines.append(
                            f"\n### {file_label} (chunk {r['chunk_index']})")
                        lines.append(f"```{lang}\n{r['code']}\n```")
                    sections.append("\n".join(lines))

        # --- Keyword fallback for patterns and subsystems ---
        keywords = _extract_keywords(task_description)

        matched_subsystems = [
            s for s in self._kb.subsystems
            if any(kw in s.name.lower() or kw in s.description.lower() for kw in keywords)
        ]
        if matched_subsystems:
            lines = ["## Relevant Subsystems"]
            for s in matched_subsystems:
                files = ", ".join(s.files[:5])
                lines.append(f"- **{s.name}**: {s.description} ({files})")
            sections.append("\n".join(lines))

        # Pattern snippets from KB — skip code in compact mode
        snippets = []
        for kw in keywords:
            snippets.extend(self.find_snippets(kw))
        seen_snip: set[tuple] = set()
        unique_snippets = []
        for s in snippets:
            key = (s["pattern"], s["label"])
            if key not in seen_snip:
                seen_snip.add(key)
                unique_snippets.append(s)

        if unique_snippets:
            lines = ["## Coding Patterns"]
            for s in unique_snippets[:3 if compact else 4]:
                lines.append(f"\n### {s['pattern']} — {s['label']}")
                if s.get("when_to_use"):
                    lines.append(f"*When to use:* {s['when_to_use']}")
                lines.append(f"*File:* `{s['file']}`")
                if not compact:
                    lang = s.get("language", "")
                    lines.append(f"```{lang}\n{s['code']}\n```")
            sections.append("\n".join(lines))

        # Architecture notes (always appended, truncated if long)
        if self._kb.architecture_notes:
            notes = self._kb.architecture_notes[:600]
            sections.append(f"## Architecture Notes\n{notes}")

        return "\n\n".join(sections)

    def summary(self) -> str:
        """Return a short textual summary of the knowledge base."""
        total_snippets = sum(
            len(p.code_snippets) for p in self._kb.coding_patterns
        )
        lines = [
            f"Project: {self._kb.project_name or 'Unknown'}",
            f"Summary: {self._kb.summary or 'N/A'}",
            f"Tech stack: {', '.join(self._kb.tech_stack) or 'N/A'}",
            f"Subsystems: {len(self._kb.subsystems)}",
            f"Coding patterns: {len(self._kb.coding_patterns)} "
            f"({total_snippets} code snippets)",
            f"Proprietary tech: {len(self._kb.proprietary_tech)}",
            f"Files analyzed: {len(self._kb.file_analyses)}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialize the knowledge base to a JSON string."""
        return self._kb.model_dump_json(indent=2)

    def save(self, path: str | Path) -> None:
        """Save the knowledge base to a JSON file."""
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeStore":
        """Load a knowledge base from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls()
        store.load_from_dict(data)
        return store


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful search keywords from a task description.

    Simple word-split with stopword removal. Good enough for KB search
    without needing an embedding model.
    """
    stopwords = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "how", "what", "when", "where", "i", "me",
        "my", "we", "our", "you", "your", "it", "its", "is", "are", "was",
        "be", "been", "being", "have", "has", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "new", "add", "create",
        "implement", "make", "write", "build",
    }
    words = text.lower().split()
    return [w.strip(".,;:!?\"'()[]{}") for w in words if w not in stopwords and len(w) > 2]


def _extract_purpose_from_embed(embed_text: str) -> str:
    """Extract the '# Purpose: ...' line from an enriched chunk's embed_text header.

    The enricher writes headers in this format::

        # File: failsafe/llm.py
        # Subsystem: Token Layer
        # Purpose: Unified LLM invocation helper with caching
        # Key elements: invoke_cached, ...
        # ---- code below ----

    Returns the purpose string, or empty string if not found.
    """
    for line in embed_text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("# purpose:"):
            return stripped[len("# purpose:"):].strip()
    return ""
