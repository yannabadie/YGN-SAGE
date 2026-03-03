"""Episodic memory - medium-term, cross-session experience storage."""
from __future__ import annotations

from typing import Any


class EpisodicMemory:
    """Simple in-memory episodic store. Replace with Qdrant/Neo4j for production."""

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
        """Search episodic memory by keyword matching.

        In production, this uses vector similarity via Qdrant.
        This in-memory version uses simple substring matching.
        """
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
