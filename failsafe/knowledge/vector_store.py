"""
Vector store — LanceDB wrapper for semantic code search.

Data layout on disk:  <codebase_path>/.failsafe/vectors/
                      └── chunks.lance     (the LanceDB table)

Schema per row:
    file         str          relative file path
    chunk_index  int          position within file
    subsystem    str          subsystem name from KB
    embed_text   str          header + code (what was embedded)
    code         str          raw code only (returned on retrieval)
    vector       list[float]  all-MiniLM-L6-v2 embedding (dim=384)

All operations are local — no network calls for storage or retrieval.
"""

from __future__ import annotations

from pathlib import Path

from failsafe.knowledge.embedder import embedder, EMBEDDING_DIM
from failsafe.knowledge.enricher import EnrichedChunk

TABLE_NAME = "chunks"


class VectorStore:
    """Local LanceDB-backed vector store for code chunk search."""

    def __init__(self, db_path: str | Path) -> None:
        import lancedb
        self._path = Path(db_path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self._path))
        self._table = self._open_or_create_table()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, chunks: list[EnrichedChunk]) -> int:
        """Embed and store a list of enriched chunks.

        Returns the number of rows successfully inserted.
        """
        if not chunks:
            return 0

        texts = [c.embed_text for c in chunks]
        vectors = embedder.embed(texts)

        rows = [
            {
                "file":        c.file,
                "chunk_index": c.chunk_index,
                "subsystem":   c.subsystem,
                "embed_text":  c.embed_text,
                "code":        c.code,
                "vector":      v,
            }
            for c, v in zip(chunks, vectors)
        ]

        self._table.add(rows)
        return len(rows)

    def delete_files(self, file_paths: list[str]) -> None:
        """Remove all rows belonging to any of the given file paths.

        Called before re-indexing when a file is modified.
        """
        if not file_paths:
            return
        # LanceDB uses SQL-like filter predicates
        escaped = [f"'{p}'" for p in file_paths]
        predicate = f"file IN ({', '.join(escaped)})"
        self._table.delete(predicate)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        n: int = 5,
        file_filter: list[str] | None = None,
    ) -> list[dict]:
        """Semantic search over all indexed chunks.

        Args:
            query:       Natural language or code query.
            n:           Maximum results to return.
            file_filter: Optional list of file paths to restrict results to.

        Returns:
            List of dicts with keys: file, chunk_index, subsystem, code,
            embed_text, _distance (similarity score, lower = better).
        """
        query_vec = embedder.embed_one(query)

        search = self._table.search(query_vec).limit(n)
        if file_filter:
            escaped = [f"'{p}'" for p in file_filter]
            search = search.where(
                f"file IN ({', '.join(escaped)})", prefilter=True)

        results = search.to_list()

        return [
            {
                "file":        r["file"],
                "chunk_index": r["chunk_index"],
                "subsystem":   r.get("subsystem", ""),
                "code":        r["code"],
                "embed_text":  r["embed_text"],
                "_distance":   r.get("_distance", 0.0),
            }
            for r in results
        ]

    def search_as_context(self, query: str, n: int = 5) -> str:
        """Return search results formatted as a context block for LLM injection.

        Coding agents can call this to get the most relevant code chunks
        for their current task, ready to inject into their prompt.
        """
        results = self.search(query, n=n)
        if not results:
            return "No relevant code found in the vector store."

        blocks = []
        for r in results:
            lang = _guess_language(r["file"])
            blocks.append(
                f"### {r['file']} (chunk {r['chunk_index']})\n"
                f"```{lang}\n{r['code']}\n```"
            )
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def chunk_count(self) -> int:
        return len(self._table)

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open_or_create_table(self):
        """Open existing table or create a fresh schema."""
        import pyarrow as pa

        schema = pa.schema([
            pa.field("file",        pa.utf8()),
            pa.field("chunk_index", pa.int32()),
            pa.field("subsystem",   pa.utf8()),
            pa.field("embed_text",  pa.utf8()),
            pa.field("code",        pa.utf8()),
            pa.field("vector",      pa.list_(pa.float32(), EMBEDDING_DIM)),
        ])

        if TABLE_NAME in self._db.table_names():
            return self._db.open_table(TABLE_NAME)

        return self._db.create_table(TABLE_NAME, schema=schema)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def for_codebase(cls, codebase_path: str | Path) -> "VectorStore":
        """Open (or create) the vector store for a given codebase.

        Stored at: <codebase_path>/.failsafe/vectors/
        """
        store_path = Path(codebase_path) / ".failsafe" / "vectors"
        return cls(store_path)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _guess_language(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    return {
        ".py": "python", ".ts": "typescript", ".js": "javascript",
        ".go": "go", ".rs": "rust", ".java": "java", ".rb": "ruby",
        ".cs": "csharp", ".cpp": "cpp", ".c": "c", ".kt": "kotlin",
        ".swift": "swift", ".sh": "bash",
    }.get(ext, "")
