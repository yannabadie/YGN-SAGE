"""RAG backend protocol for vendor-independent knowledge stores.

Defines the KnowledgeStore protocol that ExoCortex and future backends
must implement. Allows swapping between Google File Search, Qdrant,
FAISS, or other vector stores without changing consumer code.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class KnowledgeStore(Protocol):
    """Protocol for RAG backend implementations.

    Any class that implements ``search``, ``ingest``, and ``store_name``
    with compatible signatures is considered a valid KnowledgeStore,
    regardless of inheritance.

    Minimal example::

        class MyStore:
            @property
            def store_name(self) -> str:
                return "my-store"

            async def search(self, query: str, top_k: int = 5) -> list[dict]:
                return [{"content": "...", "score": 0.9}]

            async def ingest(self, content: str, metadata: dict | None = None) -> str:
                return "doc-id-123"
    """

    async def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search the knowledge store.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results to return.

        Returns:
            List of dicts, each containing at least ``content`` (str)
            and ``score`` (float, 0-1) keys.
        """
        ...

    async def ingest(self, content: str, metadata: dict | None = None) -> str:
        """Ingest content into the store.

        Args:
            content: Text content to index.
            metadata: Optional metadata dict (tags, source, etc.).

        Returns:
            A document/chunk ID string.
        """
        ...

    @property
    def store_name(self) -> str:
        """Human-readable name of the store backend."""
        ...
