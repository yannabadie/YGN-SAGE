"""S-MMU context retrieval helper for agent loop injection.

Queries the multi-view S-MMU graph (temporal, semantic, causal, entity edges)
to retrieve relevant memory chunks and formats them as an injectable context
string for LLM prompts.

This is a best-effort module: all failures are caught and logged, returning
an empty string so the agent loop is never disrupted.

Usage::

    from sage.memory.smmu_context import retrieve_smmu_context

    context = retrieve_smmu_context(working_memory)
    if context:
        messages.insert(2, Message(role=Role.SYSTEM, content=context))
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sage.memory.working import WorkingMemory

logger = logging.getLogger(__name__)

# Default weights: (temporal, semantic, causal, entity)
# Boost semantic + causal to favour meaning-related and cause-effect links.
_DEFAULT_WEIGHTS: tuple[float, float, float, float] = (1.0, 2.0, 1.5, 1.0)


def retrieve_smmu_context(
    working_memory: WorkingMemory,
    max_hops: int = 2,
    top_k: int = 5,
    weights: tuple[float, float, float, float] | None = None,
) -> str:
    """Retrieve relevant S-MMU chunks and format as a context string.

    Args:
        working_memory: The agent's working memory (wraps Rust S-MMU).
        max_hops: Maximum graph traversal depth for relevance scoring.
        top_k: Number of top-scoring chunks to include.
        weights: Optional (temporal, semantic, causal, entity) weight tuple.
                 Defaults to (1.0, 2.0, 1.5, 1.0) — boosting semantic + causal.

    Returns:
        A formatted string ready for injection as a SYSTEM message,
        or "" if the S-MMU has no chunks or retrieval fails.
    """
    try:
        chunk_count = working_memory.smmu_chunk_count()
        if chunk_count == 0:
            return ""

        # Use the most recent chunk as the query anchor
        active_chunk_id = chunk_count - 1
        w = weights or _DEFAULT_WEIGHTS

        hits = working_memory.retrieve_relevant_chunks(
            active_chunk_id, max_hops, w
        )

        if not hits:
            return ""

        # Take top_k results (already sorted descending by score from Rust)
        top_hits = hits[:top_k]

        # Format as injectable context
        lines = ["[S-MMU Graph Memory] Relevant context from compacted memory chunks:"]
        for chunk_id, score in top_hits:
            lines.append(f"- Chunk {chunk_id} (relevance: {score:.2f})")

        return "\n".join(lines)

    except Exception:
        logger.warning(
            "S-MMU context retrieval failed (best-effort, continuing without)",
            exc_info=True,
        )
        return ""
