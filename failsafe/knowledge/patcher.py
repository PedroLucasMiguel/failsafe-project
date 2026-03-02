"""
KB JSON patcher — updates only file_analyses[] entries in failsafe_kb.json.

After a coding agent modifies a file, we don't need to re-run the full
analysis pipeline. We just update the relevant file_analyses entry(ies)
in the committed KB JSON and write it back to disk.

This keeps failsafe_kb.json current so the next `failsafe sync` run has
fresh summaries to embed — 0 additional LLM calls required.
"""

from __future__ import annotations

import json
from pathlib import Path


def patch_kb_json(
    kb_path: str | Path,
    changes: dict[str, str],
) -> tuple[int, int]:
    """Patch file_analyses[] entries in failsafe_kb.json.

    Args:
        kb_path: Path to failsafe_kb.json.
        changes: {relative_path: new_purpose_description} for each changed file.
                 The purpose is a one-sentence description of what the file now does.
                 In the fast path this comes directly from the coding agent.

    Returns:
        (n_updated, n_inserted) — how many entries were updated vs newly added.
    """
    kb_path = Path(kb_path)
    if not kb_path.exists():
        return 0, 0

    kb_data = json.loads(kb_path.read_text(encoding="utf-8"))
    fa_list: list[dict] = kb_data.get("file_analyses", [])

    # Build index: path → list position
    index = {fa["path"]: i for i, fa in enumerate(fa_list)}

    n_updated = 0
    n_inserted = 0

    for rel_path, description in changes.items():
        entry = _make_entry(rel_path, description)

        if rel_path in index:
            # Update existing entry — preserve other fields if present
            existing = fa_list[index[rel_path]]
            existing["purpose"] = description
            # Also update key_elements if provided in description (agent may enrich)
            n_updated += 1
        else:
            fa_list.append(entry)
            index[rel_path] = len(fa_list) - 1
            n_inserted += 1

    kb_data["file_analyses"] = fa_list
    kb_path.write_text(
        json.dumps(kb_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return n_updated, n_inserted


def remove_from_kb_json(kb_path: str | Path, deleted_files: list[str]) -> int:
    """Remove file_analyses entries for deleted files.

    Args:
        kb_path:       Path to failsafe_kb.json.
        deleted_files: Relative paths of files that were deleted.

    Returns:
        Number of entries removed.
    """
    kb_path = Path(kb_path)
    if not kb_path.exists() or not deleted_files:
        return 0

    deleted_set = set(deleted_files)
    kb_data = json.loads(kb_path.read_text(encoding="utf-8"))
    fa_list: list[dict] = kb_data.get("file_analyses", [])

    original_len = len(fa_list)
    kb_data["file_analyses"] = [
        fa for fa in fa_list if fa.get("path") not in deleted_set
    ]
    removed = original_len - len(kb_data["file_analyses"])

    if removed:
        kb_path.write_text(
            json.dumps(kb_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return removed


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_entry(rel_path: str, purpose: str) -> dict:
    """Build a minimal file_analyses entry."""
    return {
        "path": rel_path,
        "purpose": purpose,
        "key_elements": [],
        "dependencies": [],
    }
