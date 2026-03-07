"""Embedding adapter for S-MMU semantic edges.

Provides a unified ``Embedder`` class that auto-selects between a real
embedding backend (sentence-transformers) and a deterministic hash-based
fallback.  The hash fallback allows development and testing without
installing heavy ML dependencies.

Usage::

    from sage.memory.embedder import Embedder

    emb = Embedder()          # auto-detects best backend
    vec = emb.embed("hello")  # list[float], length == EMBEDDING_DIM

The S-MMU (Rust side) expects ``Vec<f32>`` of length 384 for cosine
similarity on semantic edges.
"""

from __future__ import annotations

import hashlib
import logging
import struct
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

EMBEDDING_DIM: int = 384
"""Dimensionality of all embeddings (matches all-MiniLM-L6-v2)."""


# ---------------------------------------------------------------------------
# Protocol for pluggable backends
# ---------------------------------------------------------------------------

@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol that any embedding backend must satisfy."""

    @property
    def is_semantic(self) -> bool:
        """True if this provider produces semantically meaningful embeddings."""
        ...

    def embed(self, text: str) -> list[float]:
        """Embed a single text string into a float vector of length EMBEDDING_DIM."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        ...


# ---------------------------------------------------------------------------
# Hash-based fallback embedder
# ---------------------------------------------------------------------------

class _HashEmbedder:
    """Deterministic SHA-256-based projection to EMBEDDING_DIM floats.

    Produces vectors in the [-1, 1] range by repeatedly hashing the input
    with a counter suffix and converting digest bytes to floats.  The output
    is *not* semantically meaningful but is deterministic, cheap, and never
    produces NaN or Inf values.
    """

    @property
    def is_semantic(self) -> bool:
        return False

    def embed(self, text: str) -> list[float]:
        """Embed *text* into a deterministic float vector."""
        result: list[float] = []
        # Each SHA-256 digest gives 32 bytes = 8 floats (4 bytes each,
        # interpreted as uint32 then mapped to [-1, 1]).
        # We need ceil(384 / 8) = 48 digests.
        floats_per_hash = 8  # 32 bytes / 4 bytes per float
        rounds_needed = (EMBEDDING_DIM + floats_per_hash - 1) // floats_per_hash

        for i in range(rounds_needed):
            h = hashlib.sha256(f"{text}::{i}".encode("utf-8")).digest()
            # Unpack 32 bytes as 8 unsigned 32-bit ints (big-endian)
            values = struct.unpack(">8I", h)
            for v in values:
                if len(result) >= EMBEDDING_DIM:
                    break
                # Map uint32 [0, 2^32-1] to [-1.0, 1.0]
                result.append((v / 2_147_483_647.5) - 1.0)

        return result[:EMBEDDING_DIM]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch by calling embed() for each text."""
        return [self.embed(t) for t in texts]


# ---------------------------------------------------------------------------
# Sentence-Transformers backend
# ---------------------------------------------------------------------------

class _SentenceTransformerEmbedder:
    """Real semantic embeddings via sentence-transformers (all-MiniLM-L6-v2).

    Lazy-loads the model on first call to avoid import-time overhead.
    Outputs are L2-normalized (unit vectors) for cosine similarity.
    """

    _MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        self._model = None  # lazy

    def _load_model(self):
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        self._model = SentenceTransformer(self._MODEL_NAME)

    @property
    def is_semantic(self) -> bool:
        return True

    def embed(self, text: str) -> list[float]:
        """Embed a single text with sentence-transformers."""
        if self._model is None:
            self._load_model()
        vec = self._model.encode(text, normalize_embeddings=True)  # type: ignore[union-attr]
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts with sentence-transformers (vectorised)."""
        if not texts:
            return []
        if self._model is None:
            self._load_model()
        vecs = self._model.encode(texts, normalize_embeddings=True)  # type: ignore[union-attr]
        return [v.tolist() for v in vecs]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Embedder:
    """Unified embedding adapter with automatic backend selection.

    Parameters
    ----------
    force_hash : bool
        If True, always use the hash-based fallback regardless of whether
        sentence-transformers is available.  Useful for testing and CI.

    Examples
    --------
    >>> emb = Embedder()                 # auto-select
    >>> emb = Embedder(force_hash=True)  # force hash fallback
    >>> vec = emb.embed("hello world")
    >>> len(vec)
    384
    """

    def __init__(self, force_hash: bool = False) -> None:
        self._provider: EmbeddingProvider
        if force_hash:
            self._provider = _HashEmbedder()
        else:
            self._provider = self._auto_select()

    @staticmethod
    def _auto_select() -> EmbeddingProvider:
        """Try sentence-transformers first, fall back to hash."""
        try:
            import sentence_transformers  # noqa: F401
            provider = _SentenceTransformerEmbedder()
            logger.info("Embedder: using sentence-transformers backend")
            return provider
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; falling back to "
                "deterministic hash embedder. Semantic similarity will be "
                "degraded. Install with: pip install sentence-transformers"
            )
            return _HashEmbedder()

    @property
    def is_semantic(self) -> bool:
        """True if the active backend produces semantically meaningful embeddings."""
        return self._provider.is_semantic

    def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Returns a list of EMBEDDING_DIM (384) floats.
        """
        return self._provider.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Returns a list of embedding vectors, one per input text.
        """
        return self._provider.embed_batch(texts)
