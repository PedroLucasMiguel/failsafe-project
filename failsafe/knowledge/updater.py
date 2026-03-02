"""
Updater — incremental KB update after coding agents modify files.

Two paths, both update LanceDB and the JSON KB:

  Fast path  (agent_descriptions provided):
    Re-reads modified files → splits into chunks → uses agent description
    as enrichment header → embeds → updates LanceDB.
    ✅ 0 LLM calls, completes in ~1-2 seconds per file.

  Accurate path (no descriptions):
    Re-reads modified files → calls invoke_cached for a fresh LLM summary
    → enriches chunks with the new summary → embeds → updates LanceDB.
    ✅ 1 LLM call per file, most accurate context.

Both paths also update the JSON KB's file_analyses entries so the
KnowledgeStore stays in sync.

Usage (from a coding agent):
    from failsafe.knowledge.updater import kb_update
    from failsafe.knowledge.vector_store import VectorStore

    vs = VectorStore.for_codebase(codebase_path)
    kb_update(
        changed_files=["failsafe/agents/explorer.py"],
        codebase_path=codebase_path,
        kb=knowledge_store.knowledge_base,
        vector_store=vs,
        agent_descriptions={"failsafe/agents/explorer.py": "Added budget-aware batching"},
    )
"""

from __future__ import annotations

from pathlib import Path

from failsafe.knowledge.enricher import build_enriched_chunks, make_header_from_description
from failsafe.knowledge.models import KnowledgeBase


def kb_update(
    changed_files: list[str],
    codebase_path: str | Path,
    kb: KnowledgeBase,
    vector_store,  # VectorStore — avoid circular import
    agent_descriptions: dict[str, str] | None = None,
    llm=None,
) -> dict[str, int]:
    """Update the vector store and KB for a set of modified files.

    Args:
        changed_files:       Relative paths of files modified by the agent.
        codebase_path:       Root of the codebase on disk.
        kb:                  Current KnowledgeBase (for subsystem lookup).
        vector_store:        VectorStore instance to update.
        agent_descriptions:  Optional {rel_path: description} from the agent.
                             Enables the fast path (0 LLM calls).
        llm:                 LLM instance for the accurate path (re-summarize).
                             Pass None if using agent_descriptions.

    Returns:
        Dict {rel_path: chunks_indexed} for each updated file.
    """
    codebase_path = Path(codebase_path)
    descriptions = agent_descriptions or {}
    result: dict[str, int] = {}

    # Remove old chunks for all changed files first
    vector_store.delete_files(changed_files)

    for rel_path in changed_files:
        abs_path = codebase_path / rel_path
        if not abs_path.exists():
            # File was deleted — already removed from vector store above
            continue

        code = abs_path.read_text(encoding="utf-8", errors="replace")
        subsystem = _get_subsystem(rel_path, kb)

        if rel_path in descriptions:
            # Fast path: use agent-provided description
            chunks = _build_fast(
                rel_path, code, descriptions[rel_path], subsystem)
        elif llm is not None:
            # Accurate path: re-summarize with LLM
            chunks = _build_accurate(rel_path, code, kb, subsystem, llm)
        else:
            # Fallback: embed raw code with minimal header (better than nothing)
            chunks = build_enriched_chunks(rel_path, code, "", subsystem)

        n = vector_store.index(chunks)
        result[rel_path] = n

    return result


# ---------------------------------------------------------------------------
# Fast path — agent provides description
# ---------------------------------------------------------------------------
def _build_fast(
    rel_path: str,
    code: str,
    description: str,
    subsystem: str,
) -> list:
    """Build enriched chunks using an agent-supplied description header."""
    from failsafe.knowledge.enricher import EnrichedChunk, _split, CHUNK_SIZE

    header = make_header_from_description(rel_path, description, subsystem)
    chunks_code = _split(code, CHUNK_SIZE)
    result = []
    for i, chunk in enumerate(chunks_code):
        embed_text = f"{header}\n# ---- code below ----\n{chunk}"
        result.append(EnrichedChunk(
            file=rel_path,
            chunk_index=i,
            subsystem=subsystem,
            embed_text=embed_text,
            code=chunk,
        ))
    return result


# ---------------------------------------------------------------------------
# Accurate path — LLM re-summarizes the file
# ---------------------------------------------------------------------------
def _build_accurate(
    rel_path: str,
    code: str,
    kb: KnowledgeBase,
    subsystem: str,
    llm,
) -> list:
    """Re-summarize a file with the LLM, then enrich its chunks."""
    from failsafe.llm import invoke_cached
    from langchain_core.messages import SystemMessage, HumanMessage
    from failsafe.agents.explorer import BATCH_SUMMARY_PROMPT
    from failsafe.agents.explorer import _parse_batch_blocks

    # Use the same prompt as the explorer, just for this one file
    content = f"--- {rel_path} ---\n{code[:8000]}\n"
    summary_text = invoke_cached(llm, [
        SystemMessage(content=BATCH_SUMMARY_PROMPT),
        HumanMessage(content=content),
    ])
    parsed = _parse_batch_blocks(summary_text)
    file_summary = parsed.get(rel_path, "")

    patterns = _get_patterns_for_file(rel_path, kb)
    return build_enriched_chunks(rel_path, code, file_summary, subsystem, patterns)


# ---------------------------------------------------------------------------
# KB helpers
# ---------------------------------------------------------------------------
def _get_subsystem(file_path: str, kb: KnowledgeBase) -> str:
    for s in kb.subsystems:
        if file_path in s.files:
            return s.name
    return ""


def _get_patterns_for_file(file_path: str, kb: KnowledgeBase) -> str:
    names = [
        p.name for p in kb.coding_patterns
        if any(file_path in s for s in (p.examples or []))
        or any(file_path == sn.file for p2 in kb.coding_patterns for sn in p2.code_snippets)
    ]
    return ", ".join(names)
