"""Tests for SemanticMemory SQLite persistence (save/load)."""
import sys
import types
import os
import tempfile

if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

from sage.memory.semantic import SemanticMemory
from sage.memory.memory_agent import ExtractionResult


def test_save_and_load_roundtrip():
    """Entities and relations survive a save/load cycle."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        mem1 = SemanticMemory(db_path=db_path)
        mem1.add_extraction(ExtractionResult(
            entities=["Python", "Guido"],
            relationships=[("Python", "created_by", "Guido")],
        ))
        assert mem1.entity_count() == 2
        mem1.save()

        mem2 = SemanticMemory(db_path=db_path)
        mem2.load()
        assert mem2.entity_count() == 2
        assert "Python" in mem2._entities
        assert "Guido" in mem2._entities
        rels = mem2.query_entities("Python")
        assert len(rels) == 1
        assert rels[0] == ("Python", "created_by", "Guido")
    finally:
        os.unlink(db_path)


def test_load_empty_db():
    """Loading from a fresh DB gives empty memory."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        mem = SemanticMemory(db_path=db_path)
        mem.load()  # Should not crash
        assert mem.entity_count() == 0
    finally:
        os.unlink(db_path)


def test_save_without_db_path_is_noop():
    """Save without db_path should not crash."""
    mem = SemanticMemory()
    mem.add_extraction(ExtractionResult(
        entities=["Test"],
        relationships=[],
    ))
    mem.save()  # Should be a no-op


def test_incremental_save():
    """Multiple save/load cycles accumulate data."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        mem1 = SemanticMemory(db_path=db_path)
        mem1.add_extraction(ExtractionResult(
            entities=["A", "B"],
            relationships=[("A", "rel", "B")],
        ))
        mem1.save()

        mem2 = SemanticMemory(db_path=db_path)
        mem2.load()
        mem2.add_extraction(ExtractionResult(
            entities=["C"],
            relationships=[("B", "rel2", "C")],
        ))
        mem2.save()

        mem3 = SemanticMemory(db_path=db_path)
        mem3.load()
        assert mem3.entity_count() >= 3  # A, B, C
        assert len(mem3._relations) == 2
    finally:
        os.unlink(db_path)


def test_context_for_after_load():
    """get_context_for works after load."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        mem1 = SemanticMemory(db_path=db_path)
        mem1.add_extraction(ExtractionResult(
            entities=["Python"],
            relationships=[("Python", "is_a", "Language")],
        ))
        mem1.save()

        mem2 = SemanticMemory(db_path=db_path)
        mem2.load()
        ctx = mem2.get_context_for("Tell me about Python")
        assert "Python" in ctx
        assert "Language" in ctx
    finally:
        os.unlink(db_path)
