"""Tests for Memory v2: SQLite episodic persistence + SemanticMemory entity graph."""
from __future__ import annotations

import os
import pytest

from sage.memory.episodic import EpisodicMemory


# ---------------------------------------------------------------------------
# EpisodicMemory — SQLite backend
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sqlite_store_and_search(tmp_path):
    """Store entries in SQLite and verify keyword search returns them."""
    db = str(tmp_path / "ep.db")
    mem = EpisodicMemory(db_path=db)
    await mem.initialize()

    await mem.store("fix-auth", "Fixed the auth bug by checking token expiry.", {"task": "bug"})
    await mem.store("add-cache", "Added Redis caching layer for API responses.", {"task": "feature"})

    results = await mem.search("auth bug")
    assert len(results) >= 1
    assert any("auth" in r["content"].lower() for r in results)


@pytest.mark.asyncio
async def test_sqlite_persistence_across_instances(tmp_path):
    """Data persists when a new EpisodicMemory opens the same db file."""
    db = str(tmp_path / "ep.db")

    mem1 = EpisodicMemory(db_path=db)
    await mem1.initialize()
    await mem1.store("persist-key", "Persistent content", {"v": 1})

    # Open a fresh instance on the same file
    mem2 = EpisodicMemory(db_path=db)
    await mem2.initialize()
    results = await mem2.search("persistent")
    assert len(results) == 1
    assert results[0]["key"] == "persist-key"


@pytest.mark.asyncio
async def test_sqlite_count(tmp_path):
    """count() returns the number of stored entries."""
    db = str(tmp_path / "ep.db")
    mem = EpisodicMemory(db_path=db)
    await mem.initialize()

    assert await mem.count() == 0
    await mem.store("a", "alpha")
    await mem.store("b", "bravo")
    assert await mem.count() == 2


@pytest.mark.asyncio
async def test_sqlite_delete(tmp_path):
    """delete() removes the entry; count decreases."""
    db = str(tmp_path / "ep.db")
    mem = EpisodicMemory(db_path=db)
    await mem.initialize()

    await mem.store("temp", "Temporary note")
    assert await mem.count() == 1

    deleted = await mem.delete("temp")
    assert deleted is True
    assert await mem.count() == 0

    deleted_again = await mem.delete("temp")
    assert deleted_again is False


@pytest.mark.asyncio
async def test_sqlite_update(tmp_path):
    """update() modifies content/metadata of an existing entry (upsert)."""
    db = str(tmp_path / "ep.db")
    mem = EpisodicMemory(db_path=db)
    await mem.initialize()

    await mem.store("note", "Original content", {"v": 1})
    updated = await mem.update("note", content="Updated content", metadata={"v": 2})
    assert updated is True

    results = await mem.search("updated")
    assert len(results) == 1
    assert results[0]["content"] == "Updated content"
    assert results[0]["metadata"]["v"] == 2


@pytest.mark.asyncio
async def test_sqlite_update_nonexistent(tmp_path):
    """update() returns False for a key that doesn't exist."""
    db = str(tmp_path / "ep.db")
    mem = EpisodicMemory(db_path=db)
    await mem.initialize()

    updated = await mem.update("ghost", content="nothing")
    assert updated is False


@pytest.mark.asyncio
async def test_sqlite_list_all(tmp_path):
    """list_all() returns entries ordered by creation time descending."""
    db = str(tmp_path / "ep.db")
    mem = EpisodicMemory(db_path=db)
    await mem.initialize()

    await mem.store("first", "Content first")
    await mem.store("second", "Content second")
    await mem.store("third", "Content third")

    all_entries = await mem.list_all(limit=100)
    assert len(all_entries) == 3
    # Most recent first
    assert all_entries[0]["key"] == "third"
    assert all_entries[-1]["key"] == "first"


@pytest.mark.asyncio
async def test_sqlite_list_all_respects_limit(tmp_path):
    db = str(tmp_path / "ep.db")
    mem = EpisodicMemory(db_path=db)
    await mem.initialize()

    for i in range(10):
        await mem.store(f"k{i}", f"content {i}")

    limited = await mem.list_all(limit=3)
    assert len(limited) == 3


@pytest.mark.asyncio
async def test_sqlite_store_upsert(tmp_path):
    """Storing with an existing key replaces the entry (INSERT OR REPLACE)."""
    db = str(tmp_path / "ep.db")
    mem = EpisodicMemory(db_path=db)
    await mem.initialize()

    await mem.store("dup", "version 1")
    await mem.store("dup", "version 2")

    assert await mem.count() == 1
    results = await mem.search("version")
    assert results[0]["content"] == "version 2"


# ---------------------------------------------------------------------------
# EpisodicMemory — In-memory fallback (backward compatibility)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inmemory_fallback_store_and_search():
    """In-memory mode (no db_path) works like the old EpisodicMemory."""
    mem = EpisodicMemory()  # No db_path -> in-memory
    await mem.store("key1", "Alpha content", {"tag": "a"})
    await mem.store("key2", "Bravo content", {"tag": "b"})

    results = await mem.search("alpha")
    assert len(results) == 1
    assert results[0]["key"] == "key1"


@pytest.mark.asyncio
async def test_inmemory_count():
    mem = EpisodicMemory()
    assert await mem.count() == 0
    await mem.store("x", "data")
    assert await mem.count() == 1


@pytest.mark.asyncio
async def test_inmemory_list_all():
    mem = EpisodicMemory()
    await mem.store("a", "first")
    await mem.store("b", "second")
    entries = await mem.list_all()
    assert len(entries) == 2


@pytest.mark.asyncio
async def test_inmemory_delete():
    mem = EpisodicMemory()
    await mem.store("rm", "to remove")
    assert await mem.delete("rm") is True
    assert await mem.count() == 0
    assert await mem.delete("rm") is False


@pytest.mark.asyncio
async def test_inmemory_update():
    mem = EpisodicMemory()
    await mem.store("u", "old")
    assert await mem.update("u", content="new") is True
    results = await mem.search("new")
    assert len(results) == 1


@pytest.mark.asyncio
async def test_inmemory_initialize_is_noop():
    """initialize() on in-memory backend is a harmless no-op."""
    mem = EpisodicMemory()
    await mem.initialize()  # Should not raise
    await mem.store("ok", "works")
    assert await mem.count() == 1


@pytest.mark.asyncio
async def test_inmemory_list_keys_backward_compat():
    """list_keys() still works for backward compatibility."""
    mem = EpisodicMemory()
    await mem.store("k1", "c1")
    await mem.store("k2", "c2")
    keys = mem.list_keys()
    assert set(keys) == {"k1", "k2"}


# ---------------------------------------------------------------------------
# SemanticMemory — Entity graph
# ---------------------------------------------------------------------------


def test_semantic_add_extraction():
    from sage.memory.semantic import SemanticMemory
    from sage.memory.memory_agent import ExtractionResult

    sem = SemanticMemory()
    result = ExtractionResult(
        entities=["AgentLoop", "ToolRegistry", "Sandbox"],
        relationships=[
            ("AgentLoop", "uses", "ToolRegistry"),
            ("AgentLoop", "calls", "Sandbox"),
        ],
        summary="AgentLoop orchestrates tools and sandbox.",
    )
    sem.add_extraction(result)

    assert sem.entity_count() == 3


def test_semantic_query_entities():
    from sage.memory.semantic import SemanticMemory
    from sage.memory.memory_agent import ExtractionResult

    sem = SemanticMemory()
    sem.add_extraction(ExtractionResult(
        entities=["AgentLoop", "ToolRegistry", "Sandbox"],
        relationships=[
            ("AgentLoop", "uses", "ToolRegistry"),
            ("AgentLoop", "calls", "Sandbox"),
            ("Sandbox", "executes", "Code"),
        ],
    ))

    rels = sem.query_entities("AgentLoop", hops=1)
    assert len(rels) == 2
    subjects_and_objects = {r[0] for r in rels} | {r[2] for r in rels}
    assert "AgentLoop" in subjects_and_objects


def test_semantic_query_entities_multi_hop():
    from sage.memory.semantic import SemanticMemory
    from sage.memory.memory_agent import ExtractionResult

    sem = SemanticMemory()
    sem.add_extraction(ExtractionResult(
        entities=["A", "B", "C", "D"],
        relationships=[
            ("A", "links", "B"),
            ("B", "links", "C"),
            ("C", "links", "D"),
        ],
    ))

    # 1 hop from A: only A-B relation
    rels_1 = sem.query_entities("A", hops=1)
    assert len(rels_1) == 1

    # 2 hops from A: A-B and B-C
    rels_2 = sem.query_entities("A", hops=2)
    assert len(rels_2) == 2


def test_semantic_get_context_for():
    from sage.memory.semantic import SemanticMemory
    from sage.memory.memory_agent import ExtractionResult

    sem = SemanticMemory()
    sem.add_extraction(ExtractionResult(
        entities=["AgentLoop", "ToolRegistry"],
        relationships=[("AgentLoop", "uses", "ToolRegistry")],
    ))

    ctx = sem.get_context_for("The AgentLoop is slow")
    assert "AgentLoop" in ctx
    assert "uses" in ctx
    assert "ToolRegistry" in ctx


def test_semantic_get_context_for_empty():
    from sage.memory.semantic import SemanticMemory

    sem = SemanticMemory()
    ctx = sem.get_context_for("totally unrelated query xyz")
    assert ctx == ""


def test_semantic_get_context_for_no_data():
    from sage.memory.semantic import SemanticMemory

    sem = SemanticMemory()
    ctx = sem.get_context_for("anything")
    assert ctx == ""


def test_semantic_duplicate_entities():
    """Adding the same entities twice does not duplicate them."""
    from sage.memory.semantic import SemanticMemory
    from sage.memory.memory_agent import ExtractionResult

    sem = SemanticMemory()
    result = ExtractionResult(entities=["A", "B"], relationships=[("A", "r", "B")])
    sem.add_extraction(result)
    sem.add_extraction(result)

    assert sem.entity_count() == 2
    # Relations may accumulate (append-only), but entities are a set
