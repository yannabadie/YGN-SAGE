"""Working memory - short-term, in-memory, per-agent execution."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any


class WorkingMemory:
    """In-memory working memory for a single agent execution."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._events: list[dict[str, Any]] = []
        self._children: list[str] = []

    def add_event(self, event_type: str, content: str) -> str:
        """Add an event and return its ID."""
        event_id = str(uuid.uuid4())
        self._events.append({
            "id": event_id,
            "type": event_type,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_summary": False,
        })
        return event_id

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        """Get an event by ID."""
        for event in self._events:
            if event["id"] == event_id:
                return event
        return None

    def recent_events(self, n: int) -> list[dict[str, Any]]:
        """Get the N most recent events (oldest first within window)."""
        return self._events[-n:] if n < len(self._events) else list(self._events)

    def event_count(self) -> int:
        return len(self._events)

    def add_child_agent(self, child_id: str) -> None:
        self._children.append(child_id)

    def child_agents(self) -> list[str]:
        return list(self._children)

    def compress(self, keep_recent: int, summary: str) -> None:
        """Compress old events into a summary."""
        if len(self._events) <= keep_recent:
            return
        recent = self._events[-keep_recent:]
        self._events = [{
            "id": str(uuid.uuid4()),
            "type": "summary",
            "content": summary,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_summary": True,
        }] + recent

    def to_context_string(self) -> str:
        """Render working memory as a context string for LLM."""
        parts = []
        for event in self._events:
            parts.append(f"[{event['type']}] {event['content']}")
        return "\n".join(parts)
