"""ConsistencyScore — mean pairwise cosine similarity of text embeddings.

Used by TopologyController to detect inconsistent parallel node outputs.
Uses Rust SIMD batch_cosine_similarity when available, sentence-transformers
fallback, or returns 1.0 if only hash embeddings available (meaningless cosine).
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def consistency_score(texts: list[str], embedder: Any = None) -> float:
    """Mean pairwise cosine similarity. 1.0=identical, 0.0=orthogonal.

    Parameters
    ----------
    texts : list of strings to compare
    embedder : RustEmbedder instance (optional, for Rust SIMD path)

    Returns
    -------
    float in [0.0, 1.0]
    """
    if len(texts) <= 1:
        return 1.0

    # Try Rust SIMD path (batch_cosine_similarity on RustEmbedder)
    if embedder and hasattr(embedder, 'batch_cosine_similarity'):
        try:
            sims = embedder.batch_cosine_similarity(texts)
            return sum(sims) / len(sims) if sims else 1.0
        except Exception as exc:
            log.debug("Rust batch_cosine_similarity failed: %s", exc)

    # Try sentence-transformers fallback (real semantic embeddings)
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts, normalize_embeddings=True)
        n = len(embeddings)
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                sims.append(float(np.dot(embeddings[i], embeddings[j])))
        return sum(sims) / len(sims) if sims else 1.0
    except ImportError:
        pass

    # Hash embeddings = meaningless cosine → return 1.0 to avoid spurious reroutes
    log.debug("No semantic embedder available, returning 1.0 (skip consistency check)")
    return 1.0
