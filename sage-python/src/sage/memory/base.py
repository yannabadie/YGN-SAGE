"""Base memory types."""
from __future__ import annotations

from typing import Any, Protocol


class MemoryStore(Protocol):
    """Protocol for memory stores."""

    async def store(self, key: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        ...

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        ...
