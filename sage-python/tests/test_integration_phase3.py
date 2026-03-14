"""Integration test — Phase 3 components wired together.

Tests the full pipeline: TaskPlanner -> DynamicRouter -> DAGExecutor
-> RepairLoop -> CausalMemory -> WriteGate, all with mock runners.
"""
from __future__ import annotations

import pytest
from sage.contracts.task_node import TaskNode, IOSchema, BudgetConstraint, SecurityLabel
from sage.contracts.dag import TaskDAG
from sage.contracts.planner import TaskPlanner, PlanResult
from sage.contracts.executor import DAGExecutor
from sage.contracts.repair import RepairLoop
from sage.contracts.verification import pre_check, post_check
from sage.contracts.policy import PolicyVerifier
from sage.contracts.z3_verify import (
    verify_capability_coverage,
    verify_budget_feasibility,
    verify_type_compatibility,
)
from sage.providers.capabilities import CapabilityMatrix, ProviderCapabilities
from sage.memory.causal import CausalMemory
from sage.memory.write_gate import WriteGate

z3 = pytest.importorskip("z3", reason="z3-solver not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def capability_matrix():
    m = CapabilityMatrix()
    m.register(ProviderCapabilities(
        provider="google", structured_output=True, tool_role=True,
        file_search=True, grounding=True, streaming=True,
    ))
    m.register(ProviderCapabilities(
        provider="openai", structured_output=True, tool_role=True,
        streaming=True,
    ))
    m.register(ProviderCapabilities(
        provider="budget", structured_output=False, streaming=True,
    ))
    return m


# ---------------------------------------------------------------------------
# E2E: Plan -> Verify -> Route -> Execute
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="DynamicRouter removed — superseded by CognitiveOrchestrator")
@pytest.mark.asyncio
async def test_e2e_plan_verify_route_execute(router, capability_matrix):
    """Full pipeline: plan static DAG, verify contracts, route, execute."""

    # 1. Plan
    planner = TaskPlanner()
    plan = planner.plan_static([
        {
            "id": "extract",
            "description": "Extract key facts",
            "output": {"facts": "string"},
            "capabilities": ["structured_output"],
            "budget_usd": 0.05,
        },
        {
            "id": "summarize",
            "description": "Summarize facts",
            "input": {"facts": "string"},
            "output": {"summary": "string"},
            "depends_on": ["extract"],
            "budget_usd": 0.03,
        },
    ])
    assert plan.node_count == 2
    assert plan.edge_count == 1
    assert plan.warnings == []

    # 2. Z3 Verify contracts
    available_caps = {"structured_output", "tool_role", "streaming", "file_search", "grounding"}
    cap_verdict = verify_capability_coverage(plan.dag, available_caps)
    assert cap_verdict.satisfied is True

    budget_verdict = verify_budget_feasibility(plan.dag, total_budget_usd=0.10)
    assert budget_verdict.satisfied is True

    type_verdict = verify_type_compatibility(plan.dag)
    assert type_verdict.satisfied is True

    # 3. Route each node
    decisions: dict[str, RoutingDecision] = {}
    for nid in plan.dag.topological_sort():
        node = plan.dag.get_node(nid)
        decision = router.route(node, cost_sensitivity=0.3)
        decisions[nid] = decision
    assert all(d.provider in ("google", "openai", "budget") for d in decisions.values())

    # 4. Execute
    async def mock_runner(nid, desc, data):
        if nid == "extract":
            return {"facts": "Python is a language. FastAPI uses Python."}
        return {"summary": "Python powers FastAPI."}

    executor = DAGExecutor(plan.dag, runner=mock_runner)
    result = await executor.execute({"document": "test doc"})
    assert result.success is True
    assert result.node_results["summarize"].output["summary"] == "Python powers FastAPI."


# ---------------------------------------------------------------------------
# E2E with repair loop
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_e2e_with_repair():
    """Execute DAG with flaky runner — repair loop retries and succeeds."""
    planner = TaskPlanner()
    plan = planner.plan_static([
        {"id": "gen", "description": "Generate answer", "output": {"answer": "string"}},
    ])

    attempts = 0

    async def flaky(nid, desc, data):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return {"wrong": "field"}
        return {"answer": "42"}

    loop = RepairLoop(plan.dag, runner=flaky, max_retries=5)
    result = await loop.execute({})
    assert result.success is True
    assert attempts == 3


# ---------------------------------------------------------------------------
# E2E with causal memory + write gate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_e2e_with_memory():
    """Execute DAG and record results in causal memory with write gating."""
    planner = TaskPlanner()
    plan = planner.plan_static([
        {"id": "research", "description": "Research topic", "output": {"findings": "string"}},
        {"id": "write", "description": "Write report", "input": {"findings": "string"}, "depends_on": ["research"]},
    ])

    causal_mem = CausalMemory()
    write_gate = WriteGate(threshold=0.4)

    async def runner(nid, desc, data):
        if nid == "research":
            return {"findings": "Z3 is an SMT solver by Microsoft"}
        return {"report": "Z3 enables formal verification"}

    executor = DAGExecutor(plan.dag, runner=runner)
    result = await executor.execute({})
    assert result.success is True

    # Record in causal memory (with write gating)
    for nid in plan.dag.topological_sort():
        output = result.node_results[nid].output
        content = str(output)
        decision = write_gate.evaluate(content, confidence=0.8)
        if decision.allowed:
            causal_mem.add_entity(nid, metadata={"output": output})

    # Record causal edges from DAG structure
    for nid in plan.dag.node_ids:
        for succ in plan.dag.successors(nid):
            causal_mem.add_causal_edge(nid, succ, cause_type="feeds_into")

    assert causal_mem.entity_count() == 2
    chain = causal_mem.get_causal_chain("research")
    assert chain == ["research", "write"]
    assert write_gate.abstention_count == 0


# ---------------------------------------------------------------------------
# E2E policy rejection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_e2e_policy_blocks_execution():
    """DAG with info-flow violation is blocked before execution."""
    planner = TaskPlanner()
    plan = planner.plan_static([
        {"id": "secret", "description": "Process classified data", "security": "TOP"},
        {"id": "public", "description": "Send to public API", "security": "LOW", "depends_on": ["secret"]},
    ])

    async def runner(nid, desc, data):
        return {}

    executor = DAGExecutor(plan.dag, runner=runner)
    result = await executor.execute({})
    assert result.success is False
    assert len(result.policy_violations) > 0


# ---------------------------------------------------------------------------
# E2E Z3 catches budget overrun at plan time
# ---------------------------------------------------------------------------

def test_e2e_z3_budget_check():
    """Z3 catches budget overrun before execution starts."""
    planner = TaskPlanner()
    plan = planner.plan_static([
        {"id": "a", "description": "Expensive A", "budget_usd": 5.0},
        {"id": "b", "description": "Expensive B", "budget_usd": 5.0},
        {"id": "c", "description": "Expensive C", "budget_usd": 5.0},
    ])

    verdict = verify_budget_feasibility(plan.dag, total_budget_usd=10.0)
    assert verdict.satisfied is False
    assert "15" in verdict.counterexample or "cost" in verdict.counterexample.lower()


# ---------------------------------------------------------------------------
# E2E routing adapts after failure feedback
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="DynamicRouter removed — superseded by CognitiveOrchestrator")
def test_e2e_routing_feedback(router):
    """Router adapts after reporting failures."""
    node = TaskNode(node_id="task", description="Some task")

    # Get initial routing
    d1 = router.route(node, cost_sensitivity=0.5)
    initial_provider = d1.provider

    # Report failures for the initial provider
    for _ in range(5):
        router.report_outcome(initial_provider, success=False, latency_ms=5000)

    # Route again — score should have dropped
    d2 = router.route(node, cost_sensitivity=0.5)
    # The provider may still be selected if it's the only good option,
    # but the score should be lower
    if d2.provider == initial_provider:
        assert d2.score < d1.score
