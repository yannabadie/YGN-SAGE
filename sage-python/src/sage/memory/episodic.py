"""Episodic memory - medium-term, cross-session experience storage."""
from __future__ import annotations

from typing import Any


class EpisodicMemory:
    """In-memory episodic store with keyword search, CRUD, and key listing."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    async def store(
        self,
        key: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store an episodic memory entry."""
        self._entries.append({
            "key": key,
            "content": content,
            "metadata": metadata or {},
        })

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search episodic memory by keyword matching (substring scoring)."""
        query_lower = query.lower()
        scored = []
        for entry in self._entries:
            # Simple relevance: count matching words
            content_lower = entry["content"].lower()
            key_lower = entry["key"].lower()
            score = sum(
                1 for word in query_lower.split()
                if word in content_lower or word in key_lower
            )
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    async def update(
        self,
        key: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update an existing memory entry by key. Returns False if not found."""
        for entry in self._entries:
            if entry["key"] == key:
                if content is not None:
                    entry["content"] = content
                if metadata is not None:
                    entry["metadata"] = metadata
                return True
        return False

    async def delete(self, key: str) -> bool:
        """Delete a memory entry by key. Returns False if not found."""
        for i, entry in enumerate(self._entries):
            if entry["key"] == key:
                self._entries.pop(i)
                return True
        return False

    def list_keys(self) -> list[str]:
        """List all memory entry keys."""
        return [e["key"] for e in self._entries]
