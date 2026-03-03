"""Working memory - short-term, in-memory, per-agent execution.

Backed by the hyper-performant Rust implementation in sage_core.
"""
from __future__ import annotations

import logging
from typing import Any

import sage_core

class WorkingMemory:
    """In-memory working memory for a single agent execution."""

    def __init__(self, agent_id: str):
        self._inner = sage_core.WorkingMemory(agent_id)
        self.logger = logging.getLogger(__name__)

    @property
    def agent_id(self) -> str:
        return self._inner.agent_id

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
