"""
Embedder — local sentence-transformers wrapper for vector embeddings.

Model: all-MiniLM-L6-v2  (~80 MB, downloaded once to ~/.cache/huggingface)
  - 384-dimensional vectors
  - Fully offline after first download
  - Fast CPU inference (~5ms per chunk)

Usage:
    from failsafe.knowledge.embedder import embedder
    vectors = embedder.embed(["some code", "another chunk"])
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class Embedder:
    """Lazy-loading, thread-safe wrapper around a sentence-transformers model.

    The model is only loaded on the first call to embed() — so importing this
    module has zero overhead at startup.
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: "SentenceTransformer | None" = None
        self._lock = threading.Lock()

    def _load(self) -> "SentenceTransformer":
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings, returning a list of float vectors.

        Args:
            texts: Strings to embed (code chunks, queries, descriptions).

        Returns:
            List of 384-dimensional float vectors, one per input string.
        """
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_one(self, text: str) -> list[float]:
        """Convenience wrapper for embedding a single string."""
        return self.embed([text])[0]

    @property
    def dim(self) -> int:
        return EMBEDDING_DIM


# Module-level singleton
embedder = Embedder()
