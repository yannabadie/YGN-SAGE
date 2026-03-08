"""Tests for KnowledgeStore protocol and ExoCortex conformance."""
from __future__ import annotations

import asyncio
import pytest
from sage.memory.rag_backend import KnowledgeStore


# ---------------------------------------------------------------------------
# Protocol existence and shape
# ---------------------------------------------------------------------------

def test_knowledge_store_protocol_exists():
    """KnowledgeStore protocol exposes search, ingest, store_name."""
    assert hasattr(KnowledgeStore, "search")
    assert hasattr(KnowledgeStore, "ingest")
    assert hasattr(KnowledgeStore, "store_name")


def test_knowledge_store_is_runtime_checkable():
    """KnowledgeStore must be decorated with @runtime_checkable."""
    from typing import runtime_checkable

    # runtime_checkable protocols support isinstance/issubclass checks
    assert isinstance(KnowledgeStore, type)

    # Verify that isinstance works (will raise TypeError if not runtime_checkable)
    class _Dummy:
        async def search(self, query: str, top_k: int = 5) -> list[dict]:
            return []

        async def ingest(self, content: str, metadata: dict | None = None) -> str:
            return ""

        @property
        def store_name(self) -> str:
            return "dummy"

    assert isinstance(_Dummy(), KnowledgeStore)


# ---------------------------------------------------------------------------
# ExoCortex conformance
# ---------------------------------------------------------------------------

def test_exocortex_conforms_to_protocol():
    """ExoCortex should satisfy the KnowledgeStore protocol at instance level."""
    from sage.memory.remote_rag import ExoCortex

    exo = ExoCortex(store_name="test-store")
    assert isinstance(exo, KnowledgeStore), (
        "ExoCortex instance must satisfy KnowledgeStore protocol"
    )


def test_exocortex_has_search_method():
    """ExoCortex must expose an async search() method."""
    from sage.memory.remote_rag import ExoCortex

    exo = ExoCortex(store_name="test-store")
    assert hasattr(exo, "search")
    assert asyncio.iscoroutinefunction(exo.search)


def test_exocortex_has_ingest_method():
    """ExoCortex must expose an async ingest() method."""
    from sage.memory.remote_rag import ExoCortex

    exo = ExoCortex(store_name="test-store")
    assert hasattr(exo, "ingest")
    assert asyncio.iscoroutinefunction(exo.ingest)


def test_exocortex_has_store_name_property():
    """ExoCortex must expose a store_name property."""
    from sage.memory.remote_rag import ExoCortex

    exo = ExoCortex(store_name="my-store")
    assert exo.store_name == "my-store"


def test_exocortex_store_name_settable():
    """ExoCortex.store_name must remain settable for backward compatibility."""
    from sage.memory.remote_rag import ExoCortex

    exo = ExoCortex(store_name="original")
    exo.store_name = "updated"
    assert exo.store_name == "updated"


# ---------------------------------------------------------------------------
# ExoCortex.search() without API key (graceful degradation)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_exocortex_search_returns_empty_without_api_key(monkeypatch):
    """search() returns empty list when no API key is configured."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    from sage.memory.remote_rag import ExoCortex

    exo = ExoCortex(store_name="test-store")
    exo._api_key = ""  # Ensure no key
    results = await exo.search("test query")
    assert results == []


@pytest.mark.asyncio
async def test_exocortex_search_returns_empty_without_store(monkeypatch):
    """search() returns empty list when no store is configured."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    from sage.memory.remote_rag import ExoCortex

    exo = ExoCortex(store_name="")
    results = await exo.search("test query")
    assert results == []


# ---------------------------------------------------------------------------
# Custom backend conformance (demonstrates protocol is truly vendor-neutral)
# ---------------------------------------------------------------------------

def test_custom_backend_conforms_to_protocol():
    """A plain class with matching methods satisfies KnowledgeStore."""

    class InMemoryStore:
        def __init__(self):
            self._docs: list[dict] = []

        @property
        def store_name(self) -> str:
            return "in-memory"

        async def search(self, query: str, top_k: int = 5) -> list[dict]:
            return [d for d in self._docs if query.lower() in d["content"].lower()][:top_k]

        async def ingest(self, content: str, metadata: dict | None = None) -> str:
            doc_id = f"doc-{len(self._docs)}"
            self._docs.append({"content": content, "score": 1.0, "id": doc_id})
            return doc_id

    store = InMemoryStore()
    assert isinstance(store, KnowledgeStore)


@pytest.mark.asyncio
async def test_custom_backend_search_and_ingest():
    """End-to-end test with an in-memory KnowledgeStore implementation."""

    class InMemoryStore:
        def __init__(self):
            self._docs: list[dict] = []

        @property
        def store_name(self) -> str:
            return "in-memory"

        async def search(self, query: str, top_k: int = 5) -> list[dict]:
            results = []
            for d in self._docs:
                if query.lower() in d["content"].lower():
                    results.append({"content": d["content"], "score": 1.0})
            return results[:top_k]

        async def ingest(self, content: str, metadata: dict | None = None) -> str:
            doc_id = f"doc-{len(self._docs)}"
            self._docs.append({"content": content, "id": doc_id})
            return doc_id

    store = InMemoryStore()
    doc_id = await store.ingest("Alpha-Evolve is a PSRO-based evolution framework")
    assert doc_id == "doc-0"

    results = await store.search("PSRO")
    assert len(results) == 1
    assert "PSRO" in results[0]["content"]

    # top_k=0 returns nothing
    results = await store.search("PSRO", top_k=0)
    assert len(results) == 0


# ---------------------------------------------------------------------------
# Existing ExoCortex API backward compatibility
# ---------------------------------------------------------------------------

def test_exocortex_query_method_still_exists():
    """Existing synchronous query() must still work."""
    from sage.memory.remote_rag import ExoCortex

    exo = ExoCortex(store_name="test")
    assert hasattr(exo, "query")
    assert callable(exo.query)


def test_exocortex_upload_method_still_exists():
    """Existing async upload() must still work."""
    from sage.memory.remote_rag import ExoCortex

    exo = ExoCortex(store_name="test")
    assert hasattr(exo, "upload")
    assert asyncio.iscoroutinefunction(exo.upload)


def test_exocortex_get_file_search_tool_still_exists():
    """Existing get_file_search_tool() must still work."""
    from sage.memory.remote_rag import ExoCortex

    exo = ExoCortex(store_name="test")
    assert hasattr(exo, "get_file_search_tool")
    assert callable(exo.get_file_search_tool)


def test_exocortex_is_available_still_works():
    """Existing is_available property must still work."""
    from sage.memory.remote_rag import ExoCortex

    exo = ExoCortex(store_name="test")
    exo._api_key = ""
    assert exo.is_available is False

    exo._api_key = "fake-key"
    assert exo.is_available is True
