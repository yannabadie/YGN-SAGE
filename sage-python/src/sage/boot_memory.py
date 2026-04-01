"""Boot sub-module: memory tier initialization."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sage.memory.compressor import MemoryCompressor
from sage.memory.episodic import EpisodicMemory
from sage.memory.memory_agent import MemoryAgent
from sage.memory.remote_rag import ExoCortex

__all__ = ["init_memory"]

_log = logging.getLogger("sage.boot")


def init_memory(
    event_bus: Any,
    provider: Any,
    use_mock_llm: bool,
    agent_loop: Any,
) -> dict[str, Any]:
    """Initialize all memory tiers and wire them into the agent loop.

    Returns:
        dict with keys: memory_agent, memory_compressor, episodic_memory,
        semantic_memory, causal_memory, consolidator, exocortex.
    """
    from sage.constants import (
        MEMORY_COMPRESSION_THRESHOLD,
        MEMORY_KEEP_RECENT,
    )

    memory_agent = MemoryAgent(
        use_llm=not use_mock_llm,
        llm_provider=provider if not use_mock_llm else None,
    )

    # Memory compressor (fires on pressure -- MEM1 pattern)
    memory_compressor = MemoryCompressor(
        llm=provider,
        compression_threshold=MEMORY_COMPRESSION_THRESHOLD,
        keep_recent=MEMORY_KEEP_RECENT,
    )

    # Embedder for S-MMU semantic edges
    from sage.memory.embedder import Embedder
    memory_compressor.embedder = Embedder()

    # --- Degradation warnings (loud, not silent) ---
    from sage.memory.working import _has_rust as _rust_available
    if not _rust_available:
        _log.warning(
            "sage_core Rust extension not compiled -- working memory uses a "
            "pure-Python mock that returns dummy values for Arrow/S-MMU "
            "operations. Build with: cd sage-core && maturin develop"
        )

    # Episodic memory -- defaults to persistent SQLite
    _ep_db = Path.home() / ".sage" / "episodic.db"
    _ep_db.parent.mkdir(parents=True, exist_ok=True)
    episodic_memory = EpisodicMemory(db_path=str(_ep_db))

    # Safety net: warn if someone overrides with db_path=None upstream
    if not episodic_memory._db_path:
        _log.warning(
            "Episodic memory is volatile (in-memory only, data lost on "
            "restart). Pass db_path to EpisodicMemory for persistence."
        )

    # ExoCortex (persistent RAG via Google GenAI File Search)
    exocortex = ExoCortex()

    # Wire episodic into loop
    agent_loop.episodic_memory = episodic_memory
    agent_loop.exocortex = exocortex

    # Semantic memory + MemoryAgent wiring (persistent SQLite in real mode)
    from sage.memory.semantic import SemanticMemory
    if not use_mock_llm:
        _sem_db = Path.home() / ".sage" / "semantic.db"
        _sem_db.parent.mkdir(parents=True, exist_ok=True)
        semantic_memory = SemanticMemory(db_path=str(_sem_db))
        semantic_memory.load()
    else:
        semantic_memory = SemanticMemory()
    agent_loop.memory_agent = memory_agent
    agent_loop.semantic_memory = semantic_memory

    # Causal memory (persistent SQLite in real mode)
    from sage.memory.causal import CausalMemory
    if not use_mock_llm:
        _causal_db = Path.home() / ".sage" / "causal.db"
        _causal_db.parent.mkdir(parents=True, exist_ok=True)
        causal_memory = CausalMemory(db_path=str(_causal_db))
        causal_memory.load()
    else:
        causal_memory = CausalMemory()
    agent_loop.causal_memory = causal_memory

    # Inter-tier consolidation: episodic -> semantic -> causal (MAGMA 2601.03236)
    from sage.memory.consolidator import MemoryConsolidator
    consolidator = MemoryConsolidator(
        episodic=episodic_memory,
        semantic=semantic_memory,
        causal=causal_memory,
        memory_agent=memory_agent,
    )
    agent_loop.consolidator = consolidator

    return {
        "memory_agent": memory_agent,
        "memory_compressor": memory_compressor,
        "episodic_memory": episodic_memory,
        "semantic_memory": semantic_memory,
        "causal_memory": causal_memory,
        "consolidator": consolidator,
        "exocortex": exocortex,
    }
