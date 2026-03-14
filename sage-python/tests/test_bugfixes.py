"""Tests for bugs discovered during deep code review (March 7, 2026).

Each test verifies a specific bug fix:
BF-2: DAGExecutor non-dict output handling
BF-3: RepairLoop non-dict output handling
BF-4: WriteGate bounded dedup (no unbounded growth)
BF-5: CostTracker floating-point epsilon tolerance
BF-6: CausalMemory bounded growth + context truncation
"""
from __future__ import annotations

import pytest
from sage.contracts.task_node import TaskNode, IOSchema
from sage.contracts.dag import TaskDAG
from sage.contracts.executor import DAGExecutor
from sage.contracts.repair import RepairLoop
from sage.contracts.cost_tracker import CostTracker
from sage.memory.write_gate import WriteGate
from sage.memory.causal import CausalMemory


# ===========================================================================
# BF-2: DAGExecutor non-dict output
# ===========================================================================

@pytest.mark.asyncio
async def test_bf2_executor_handles_none_output():
    """Executor should fail gracefully when runner returns None."""
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="bad", description="Returns None"))

    async def bad_runner(nid, desc, data):
        return None  # type: ignore[return-value]

    executor = DAGExecutor(dag, runner=bad_runner)
    result = await executor.execute({})
    assert result.success is False
    assert "NoneType" in result.node_results["bad"].error


@pytest.mark.asyncio
async def test_bf2_executor_handles_string_output():
    """Executor should fail gracefully when runner returns a string."""
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="bad", description="Returns string"))

    async def string_runner(nid, desc, data):
        return "just a string"  # type: ignore[return-value]

    executor = DAGExecutor(dag, runner=string_runner)
    result = await executor.execute({})
    assert result.success is False
    assert "str" in result.node_results["bad"].error


# ===========================================================================
# BF-3: RepairLoop non-dict output
# ===========================================================================

@pytest.mark.asyncio
async def test_bf3_repair_handles_none_output():
    """RepairLoop should handle non-dict output without crashing."""
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="bad", description="Returns None then dict",
        output_schema=IOSchema(fields={"val": "string"}),
    ))

    attempts = 0

    async def eventually_dict(nid, desc, data):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return None  # type: ignore[return-value]
        return {"val": "ok"}

    loop = RepairLoop(dag, runner=eventually_dict, max_retries=5)
    result = await loop.execute({})
    assert result.success is True
    assert attempts == 3


# ===========================================================================
# BF-4: WriteGate bounded dedup
# ===========================================================================

def test_bf4_write_gate_bounded_dedup():
    """WriteGate should evict old entries when max_dedup_size is exceeded."""
    gate = WriteGate(threshold=0.1, max_dedup_size=5)

    # Write 10 unique entries
    for i in range(10):
        gate.evaluate(f"content_{i}", confidence=0.9)

    # Dedup set should be bounded at 5
    assert len(gate._seen_content) == 5

    # Old entries (0-4) should have been evicted, so they're no longer dupes
    d = gate.evaluate("content_0", confidence=0.9)
    assert d.allowed is True  # Not a duplicate anymore

    # Recent entries (5-9) should still be detected as dupes
    d2 = gate.evaluate("content_9", confidence=0.9)
    assert d2.allowed is False
    assert "duplicate" in d2.reason


# ===========================================================================
# BF-5: CostTracker float epsilon
# ===========================================================================

def test_bf5_cost_tracker_exact_budget_not_over():
    """Spending exactly the budget should NOT be considered over budget."""
    tracker = CostTracker(budget_usd=1.0)
    tracker.record("a", 0.5)
    tracker.record("b", 0.5)
    # total_spent == budget_usd exactly — should NOT trigger over_budget
    assert tracker.is_over_budget is False


def test_bf5_cost_tracker_tiny_overshoot_not_over():
    """Floating-point rounding error should NOT trigger false over-budget."""
    tracker = CostTracker(budget_usd=0.3)
    tracker.record("a", 0.1)
    tracker.record("b", 0.1)
    tracker.record("c", 0.1)
    # 0.1 + 0.1 + 0.1 == 0.30000000000000004 in IEEE 754
    # With epsilon tolerance, this should NOT be over budget
    assert tracker.is_over_budget is False


def test_bf5_cost_tracker_real_overshoot_is_over():
    """Genuine overshoot (beyond epsilon) should still be detected."""
    tracker = CostTracker(budget_usd=0.3)
    tracker.record("a", 0.2)
    tracker.record("b", 0.2)
    # 0.4 > 0.3 + epsilon — genuine overshoot
    assert tracker.is_over_budget is True


# ===========================================================================
# BF-6: CausalMemory bounded growth + context truncation
# ===========================================================================

def test_bf6_causal_memory_evicts_oldest():
    """CausalMemory with max_entities evicts oldest entries."""
    mem = CausalMemory(max_entities=5)
    for i in range(10):
        mem.add_entity(f"ent_{i}", metadata={"i": i})

    assert mem.entity_count() == 5
    # Oldest entities (0-4) should be evicted
    assert not mem.has_entity("ent_0")
    assert not mem.has_entity("ent_4")
    # Recent entities (5-9) should remain
    assert mem.has_entity("ent_5")
    assert mem.has_entity("ent_9")


def test_bf6_causal_memory_unlimited_by_default():
    """CausalMemory with max_entities=0 grows without limit."""
    mem = CausalMemory(max_entities=0)
    for i in range(100):
        mem.add_entity(f"e{i}")
    assert mem.entity_count() == 100


def test_bf6_context_truncation():
    """get_context_for() truncates to max_context_lines."""
    mem = CausalMemory(max_context_lines=3)
    for i in range(20):
        mem.add_entity(f"x{i}")
    # Create relations so context has many lines
    for i in range(19):
        mem.add_relation(f"x{i}", "next", f"x{i + 1}")

    # Query with a task that mentions x0 (which has relations)
    ctx = mem.get_context_for("process x0 data")
    lines = ctx.strip().split("\n")
    assert len(lines) <= 3


# ===========================================================================
# BF-7: SemanticMemory relation dedup + bounded growth + context truncation
# ===========================================================================

def test_bf7_semantic_relation_dedup():
    """SemanticMemory deduplicates identical triples on write."""
    from sage.memory.semantic import SemanticMemory
    from sage.memory.memory_agent import ExtractionResult

    sem = SemanticMemory()
    result = ExtractionResult(
        entities=["A", "B"],
        relationships=[("A", "links", "B")],
    )
    # Add same extraction 5 times
    for _ in range(5):
        sem.add_extraction(result)

    # Should have exactly 1 relation, not 5
    rels = sem.query_entities("A", hops=1)
    assert len(rels) == 1


def test_bf7_semantic_max_relations():
    """SemanticMemory evicts oldest relations when max_relations exceeded."""
    from sage.memory.semantic import SemanticMemory
    from sage.memory.memory_agent import ExtractionResult

    sem = SemanticMemory(max_relations=5)
    for i in range(10):
        sem.add_extraction(ExtractionResult(
            entities=[f"e{i}", f"e{i + 100}"],
            relationships=[(f"e{i}", "rel", f"e{i + 100}")],
        ))

    # Both list and set should be capped at 5
    assert len(sem._relations) == 5
    assert len(sem._relations_set) == 5


def test_bf7_semantic_context_truncation():
    """SemanticMemory get_context_for() truncates to max_context_lines."""
    from sage.memory.semantic import SemanticMemory
    from sage.memory.memory_agent import ExtractionResult

    sem = SemanticMemory(max_context_lines=3)
    # Add many relations mentioning "Alpha"
    rels = [("Alpha", f"action_{i}", f"Target_{i}") for i in range(20)]
    sem.add_extraction(ExtractionResult(
        entities=["Alpha"] + [f"Target_{i}" for i in range(20)],
        relationships=rels,
    ))

    ctx = sem.get_context_for("process Alpha data")
    assert ctx != ""
    lines = ctx.strip().split("\n")
    assert len(lines) <= 3
