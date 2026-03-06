"""Tests for Causal Memory — entity-relation graph with causal edges."""
from __future__ import annotations

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
