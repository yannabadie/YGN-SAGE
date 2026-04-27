"""Tests for Causal Memory — entity-relation graph with causal edges."""
from __future__ import annotations

import os
import tempfile

import pytest
from sage.memory.causal import CausalMemory, CausalEdge


# ---------------------------------------------------------------------------
# Basic entity/relation storage
# ---------------------------------------------------------------------------

def test_add_entity():
    mem = CausalMemory()
    mem.add_entity("AgentA", metadata={"type": "agent"})
    assert mem.entity_count() == 1
    assert mem.has_entity("AgentA")


def test_add_relation():
    mem = CausalMemory()
    mem.add_entity("AgentA")
    mem.add_entity("TaskB")
    mem.add_relation("AgentA", "executes", "TaskB")
    rels = mem.get_relations("AgentA")
    assert len(rels) == 1
    assert rels[0] == ("AgentA", "executes", "TaskB")


# ---------------------------------------------------------------------------
# Causal edges (directed, typed)
# ---------------------------------------------------------------------------

def test_add_causal_edge():
    mem = CausalMemory()
    mem.add_entity("StepA")
    mem.add_entity("StepB")
    edge = mem.add_causal_edge("StepA", "StepB", cause_type="enables")
    assert isinstance(edge, CausalEdge)
    assert edge.source == "StepA"
    assert edge.target == "StepB"
    assert edge.cause_type == "enables"


def test_get_causal_chain():
    """A -> B -> C should return full chain."""
    mem = CausalMemory()
    for ent in ["A", "B", "C"]:
        mem.add_entity(ent)
    mem.add_causal_edge("A", "B", cause_type="caused")
    mem.add_causal_edge("B", "C", cause_type="caused")

    chain = mem.get_causal_chain("A")
    # Chain should include A, B, C in order
    assert chain == ["A", "B", "C"]


def test_get_causal_chain_no_edges():
    mem = CausalMemory()
    mem.add_entity("X")
    chain = mem.get_causal_chain("X")
    assert chain == ["X"]


def test_causal_chain_with_branching():
    """A -> B, A -> C: chain from A returns A then B, C (BFS)."""
    mem = CausalMemory()
    for ent in ["A", "B", "C"]:
        mem.add_entity(ent)
    mem.add_causal_edge("A", "B", cause_type="caused")
    mem.add_causal_edge("A", "C", cause_type="caused")
    chain = mem.get_causal_chain("A")
    assert chain[0] == "A"
    assert set(chain[1:]) == {"B", "C"}


# ---------------------------------------------------------------------------
# Temporal ordering
# ---------------------------------------------------------------------------

def test_temporal_ordering():
    """Entities added later have higher temporal index."""
    mem = CausalMemory()
    mem.add_entity("First")
    mem.add_entity("Second")
    mem.add_entity("Third")
    order = mem.temporal_order()
    assert order == ["First", "Second", "Third"]


# ---------------------------------------------------------------------------
# Context generation
# ---------------------------------------------------------------------------

def test_get_context_for_task():
    mem = CausalMemory()
    mem.add_entity("Python")
    mem.add_entity("FastAPI")
    mem.add_relation("Python", "powers", "FastAPI")
    ctx = mem.get_context_for("Build a FastAPI app")
    assert "FastAPI" in ctx
    assert "Python" in ctx


def test_get_context_empty():
    mem = CausalMemory()
    ctx = mem.get_context_for("unrelated task")
    assert ctx == ""


# ---------------------------------------------------------------------------
# Causal ancestors/descendants
# ---------------------------------------------------------------------------

def test_get_causal_ancestors():
    """If A caused B caused C, ancestors of C = [B, A]."""
    mem = CausalMemory()
    for ent in ["A", "B", "C"]:
        mem.add_entity(ent)
    mem.add_causal_edge("A", "B", cause_type="caused")
    mem.add_causal_edge("B", "C", cause_type="caused")
    ancestors = mem.get_causal_ancestors("C")
    assert "B" in ancestors
    assert "A" in ancestors
    assert "C" not in ancestors


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------

def test_causal_memory_save_load_entities_and_edges():
    """Round-trip: entities, relations, and causal edges survive save/load."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        cm = CausalMemory(db_path=db_path)
        cm.add_entity("A")
        cm.add_entity("B")
        cm.add_entity("C")
        cm.add_relation("A", "related_to", "B")
        cm.add_causal_edge("A", "B", cause_type="caused")
        cm.add_causal_edge("B", "C", cause_type="enabled")
        cm.save()

        cm2 = CausalMemory(db_path=db_path)
        cm2.load()
        assert cm2.has_entity("A")
        assert cm2.has_entity("B")
        assert cm2.has_entity("C")
        assert cm2.entity_count() == 3
        # Semantic relations restored
        rels = cm2.get_relations("A")
        assert ("A", "related_to", "B") in rels
        # Causal chain restored
        chain = cm2.get_causal_chain("A")
        assert chain == ["A", "B", "C"]
    finally:
        os.unlink(db_path)


def test_causal_memory_save_load_preserves_temporal_order():
    """Insertion order must survive a save/load round-trip."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        cm = CausalMemory(db_path=db_path)
        cm.add_entity("First")
        cm.add_entity("Second")
        cm.add_entity("Third")
        cm.save()

        cm2 = CausalMemory(db_path=db_path)
        cm2.load()
        assert cm2.temporal_order() == ["First", "Second", "Third"]
    finally:
        os.unlink(db_path)


def test_causal_memory_no_db_path_noop():
    """save/load should be no-ops when db_path is None."""
    cm = CausalMemory()
    cm.add_entity("X")
    cm.save()  # Should not raise
    cm.load()  # Should not raise
    assert cm.has_entity("X")


def test_causal_memory_load_nonexistent_file():
    """load() with a path to a non-existent file should be a no-op."""
    cm = CausalMemory(db_path="/tmp/does_not_exist_causal_test.db")
    cm.load()  # Should not raise
    assert cm.entity_count() == 0


def test_causal_memory_load_empty_db():
    """load() on an empty SQLite file (no tables) should be a no-op."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        # Create an empty SQLite database (no tables)
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.close()

        cm = CausalMemory(db_path=db_path)
        cm.load()  # Should not raise
        assert cm.entity_count() == 0
    finally:
        os.unlink(db_path)


def test_causal_memory_save_overwrites_previous():
    """A second save should overwrite, not accumulate."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        cm = CausalMemory(db_path=db_path)
        cm.add_entity("A")
        cm.add_entity("B")
        cm.save()

        # Mutate and save again (A removed, D added)
        cm2 = CausalMemory(db_path=db_path)
        cm2.add_entity("D")
        cm2.save()

        cm3 = CausalMemory(db_path=db_path)
        cm3.load()
        assert cm3.has_entity("D")
        assert not cm3.has_entity("A")  # Overwritten, not accumulated
        assert cm3.entity_count() == 1
    finally:
        os.unlink(db_path)


def test_causal_memory_metadata_survives_save_load():
    """Regression: entity metadata + edge metadata round-trip via JSON columns.

    Before 2026-04-27, ``save`` only persisted name/order/source/target/
    cause_type; ``add_entity(metadata={...})`` and
    ``add_causal_edge(**metadata)`` payloads were dropped on every
    save → silent persistence-contract bug. cgpro recommended additive
    ALTER TABLE migration with JSON columns.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        cm = CausalMemory(db_path=db_path)
        cm.add_entity("AgentA", metadata={"type": "agent", "version": 2})
        cm.add_entity("StepB", metadata={"role": "step", "duration_ms": 42})
        cm.add_entity("Bare")  # no metadata
        cm.add_causal_edge("AgentA", "StepB", cause_type="enabled", confidence=0.9, src_run="run-001")
        cm.add_causal_edge("StepB", "Bare", cause_type="caused")  # bare edge
        cm.save()

        cm2 = CausalMemory(db_path=db_path)
        cm2.load()

        # Entity metadata round-trips
        assert cm2._entities["AgentA"] == {"type": "agent", "version": 2}
        assert cm2._entities["StepB"] == {"role": "step", "duration_ms": 42}
        assert cm2._entities["Bare"] == {}

        # Edge metadata round-trips
        edges = list(cm2._causal_edges)
        labelled = next(e for e in edges if e.source == "AgentA" and e.target == "StepB")
        assert labelled.metadata == {"confidence": 0.9, "src_run": "run-001"}
        bare = next(e for e in edges if e.source == "StepB" and e.target == "Bare")
        assert bare.metadata == {}
    finally:
        os.unlink(db_path)


def test_causal_memory_load_legacy_schema_no_metadata_column():
    """Backward compat: a v1 DB without metadata_json columns must still load.

    Pre-2026-04-27 deployments wrote the 3-column schema. Loading these
    must not crash and must default metadata to ``{}``. ``save`` then
    upgrades the schema in place via ``ALTER TABLE``.
    """
    import sqlite3 as _sqlite3
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        # Manually construct a legacy v1 DB exactly as the pre-fix code wrote it.
        conn = _sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE causal_entities (name TEXT PRIMARY KEY, sort_order INTEGER)"
        )
        conn.execute(
            "CREATE TABLE causal_relations "
            "(subj TEXT, pred TEXT, obj TEXT, PRIMARY KEY (subj, pred, obj))"
        )
        conn.execute(
            "CREATE TABLE causal_edges (source TEXT, target TEXT, cause_type TEXT, "
            "PRIMARY KEY (source, target, cause_type))"
        )
        conn.executemany(
            "INSERT INTO causal_entities VALUES (?, ?)",
            [("A", 0), ("B", 1)],
        )
        conn.executemany(
            "INSERT INTO causal_edges VALUES (?, ?, ?)",
            [("A", "B", "caused")],
        )
        conn.commit()
        conn.close()

        # Load the legacy DB; must not raise.
        cm = CausalMemory(db_path=db_path)
        cm.load()
        assert cm.has_entity("A")
        assert cm.has_entity("B")
        assert cm._entities["A"] == {}  # legacy → empty metadata
        assert cm._causal_edges[0].metadata == {}

        # Save once; ALTER TABLE upgrades the schema in place.
        cm.save()

        # Re-open the DB directly and verify the column now exists.
        conn = _sqlite3.connect(db_path)
        ent_cols = {row[1] for row in conn.execute("PRAGMA table_info(causal_entities)")}
        edge_cols = {row[1] for row in conn.execute("PRAGMA table_info(causal_edges)")}
        conn.close()
        assert "metadata_json" in ent_cols
        assert "metadata_json" in edge_cols
    finally:
        os.unlink(db_path)
