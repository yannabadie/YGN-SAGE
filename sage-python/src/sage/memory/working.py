"""Working memory - short-term, in-memory, per-agent execution.

Delegates to the Rust implementation in sage_core when available,
falls back to a pure-Python mock otherwise.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any

# Windows: ensure ONNX Runtime DLL is found before System32 fallback.
# An old onnxruntime.dll in C:\Windows\System32 can shadow the correct one.
if sys.platform == "win32":
    _ort_dll_dirs = [
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "sage-core", "target", "release"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "sage-core", "target", "debug"),
    ]
    for _d in _ort_dll_dirs:
        if os.path.isdir(_d):
            try:
                os.add_dll_directory(_d)
            except OSError:
                pass

try:
    import sage_core
    _has_rust = hasattr(sage_core, "WorkingMemory")
except ImportError:
    _has_rust = False

if not _has_rust:
    # Pure-Python fallback when sage_core Rust extension isn't compiled.
    import types as _types

    class _MockEvent:
        def __init__(self, eid, etype, content):
            self.id = eid
            self.event_type = etype
            self.content = content
            self.timestamp_str = ""
            self.is_summary = False

    class _PyWorkingMemory:
        def __init__(self, agent_id="", parent_id=None):
            self.agent_id = agent_id
            self.parent_id = parent_id
            self._events: list = []
            self._counter = 0

        def add_event(self, t, c):
            self._counter += 1
            eid = f"evt-{self._counter}"
            self._events.append(_MockEvent(eid, t, c))
            return eid

        def get_event(self, eid):
            return next((e for e in self._events if e.id == eid), None)

        def recent_events(self, n=10):
            return self._events[-n:]

        def event_count(self):
            return len(self._events)

        def add_child_agent(self, cid): pass
        def child_agents(self): return []
        def compress_old_events(self, k, s): self._events = self._events[-k:]
        def compact_to_arrow(self): return 0
        def compact_to_arrow_with_meta(self, kw, emb=None, parent=None, summary=None): return 0
        def retrieve_relevant_chunks(self, cid, hops, w=None): return []
        def get_page_out_candidates(self, cid, hops, budget): return []
        def smmu_chunk_count(self): return 0
        def get_latest_arrow_chunk(self): return None

    sage_core = _types.ModuleType("sage_core")
    sage_core.WorkingMemory = _PyWorkingMemory

class WorkingMemory:
    """In-memory working memory for a single agent execution."""

    def __init__(self, agent_id: str, parent_id: str | None = None):
        self._inner = sage_core.WorkingMemory(agent_id, parent_id)
        self.logger = logging.getLogger(__name__)

    @property
    def agent_id(self) -> str:
        return self._inner.agent_id

    @property
    def parent_id(self) -> str | None:
        return self._inner.parent_id

    def add_event(self, event_type: str, content: str) -> str:
        """Add an event and return its ID."""
        return self._inner.add_event(event_type, content)

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        """Get an event by ID."""
        # The Rust method returns a MemoryEvent which has these properties
        event = self._inner.get_event(event_id)
        if event is None:
            return None
        return {
            "id": event.id,
            "type": event.event_type,
            "content": event.content,
            "timestamp": event.timestamp_str,
            "is_summary": event.is_summary,
        }

    def recent_events(self, n: int) -> list[dict[str, Any]]:
        """Get the N most recent events (oldest first within window)."""
        events = self._inner.recent_events(n)
        return [{
            "id": e.id,
            "type": e.event_type,
            "content": e.content,
            "timestamp": e.timestamp_str,
            "is_summary": e.is_summary,
        } for e in events]

    def event_count(self) -> int:
        return self._inner.event_count()

    def add_child_agent(self, child_id: str) -> None:
        self._inner.add_child_agent(child_id)

    def child_agents(self) -> list[str]:
        return self._inner.child_agents()

    def compress(self, keep_recent: int, summary: str) -> None:
        """Compress old events into a summary."""
        self._inner.compress_old_events(keep_recent, summary)

    def compact_to_arrow(self) -> int:
        """Compact active buffer into an immutable Arrow RecordBatch and register it in the S-MMU graph."""
        return self._inner.compact_to_arrow()

    def compact_to_arrow_with_meta(
        self,
        keywords: list[str],
        embedding: list[float] | None = None,
        parent_chunk_id: int | None = None,
        summary: str | None = None,
    ) -> int:
        """Compact active buffer with full metadata (keywords, embedding, parent chunk, summary).

        Args:
            keywords: Entity/keyword tags for entity-graph linking.
            embedding: Optional embedding vector for semantic similarity linking.
            parent_chunk_id: Optional parent chunk ID for causal linking.
            summary: Optional human-readable summary for S-MMU chunk registration.

        Returns:
            The assigned chunk ID in the S-MMU.
        """
        return self._inner.compact_to_arrow_with_meta(
            keywords, embedding, parent_chunk_id, summary
        )

    def retrieve_relevant_chunks(
        self,
        active_chunk_id: int,
        max_hops: int,
        weights: tuple[float, float, float, float] | None = None,
    ) -> list[tuple[int, float]]:
        """Retrieve relevant chunks by walking the multi-view S-MMU graph.

        Args:
            active_chunk_id: The chunk to search from.
            max_hops: Maximum graph traversal depth.
            weights: Optional (temporal, semantic, causal, entity) weighting factors.

        Returns:
            List of (chunk_id, score) tuples, sorted descending by score.
        """
        return self._inner.retrieve_relevant_chunks(
            active_chunk_id, max_hops, weights
        )

    def get_page_out_candidates(
        self,
        active_chunk_id: int,
        max_hops: int,
        budget: int,
    ) -> list[int]:
        """Get chunk IDs that are candidates for eviction (page-out).

        Returns the least relevant chunks relative to the active chunk.

        Args:
            active_chunk_id: The currently active chunk.
            max_hops: Maximum graph traversal depth for relevance scoring.
            budget: Maximum number of candidates to return.

        Returns:
            List of chunk IDs suitable for eviction.
        """
        return self._inner.get_page_out_candidates(
            active_chunk_id, max_hops, budget
        )

    def smmu_chunk_count(self) -> int:
        """Number of chunks registered in the S-MMU."""
        return self._inner.smmu_chunk_count()

    def get_latest_arrow_chunk(self) -> Any:
        """Export the latest compacted Arrow chunk to Python (Zero-Copy)."""
        return self._inner.get_latest_arrow_chunk()

    @property
    def _events(self) -> list[dict[str, Any]]:
        """Backwards compatibility for direct access in some places."""
        return self.recent_events(self.event_count())

    def to_context_string(self) -> str:
        """Render working memory as a context string for LLM."""
        parts = []
        for event in self.recent_events(self.event_count()):
            parts.append(f"[{event['type']}] {event['content']}")
        return "\n".join(parts)
