from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from sage.contracts.cost_tracker import CostTracker
from sage.pipeline import EXECUTE_BUDGET_EXCEEDED, CognitiveOrchestrationPipeline
from sage.pipeline_v2.execute import execute


BUDGET_SENTINEL = "[sage: budget exceeded]"


class _CapturingBus:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def emit(self, event: Any) -> None:
        self.events.append(event)

    @property
    def stages(self) -> list[str]:
        return [
            event.meta.get("stage")
            for event in self.events
            if hasattr(event, "meta")
        ]


class _Router:
    def __init__(self, system: int = 1) -> None:
        self.system = system

    def assess_complexity(self, task: str) -> Any:
        return SimpleNamespace(system=self.system)

    def route(self, profile: Any) -> Any:
        return SimpleNamespace(system=profile.system)


class _Response:
    content = "Pipeline test response"
    tool_calls: list[Any] = []


class _Provider:
    async def generate(
        self,
        messages: Any,
        config: Any = None,
        tools: Any = None,
        **kwargs: Any,
    ) -> _Response:
        return _Response()


class _Node:
    def __init__(self, idx: int) -> None:
        self.role = f"role-{idx}"
        self.model_id = ""
        self.max_cost_usd = 0.0
        self.required_capabilities: list[str] = []


class _Topology:
    def __init__(self, n_nodes: int = 2) -> None:
        self._nodes = [_Node(idx) for idx in range(n_nodes)]
        self.id = "budget-test-topology"

    def node_count(self) -> int:
        return len(self._nodes)

    def get_node(self, idx: int) -> _Node:
        return self._nodes[idx]

    def set_node_model_id(self, idx: int, model_id: str) -> None:
        self._nodes[idx].model_id = model_id

    def get_predecessors(self, node_idx: int) -> list[int]:
        return [node_idx - 1] if node_idx > 0 else []


class _Engine:
    def __init__(self, topology: _Topology) -> None:
        self.topology = topology

    def generate(
        self,
        task: str,
        task_embedding: Any,
        system: int,
        budget: float,
    ) -> Any:
        return SimpleNamespace(
            topology=self.topology,
            source="test",
            confidence=1.0,
        )


class _Assigner:
    def assign_models(
        self,
        topology: _Topology,
        domain: str,
        budget: float,
        hints: Any = None,
        task_system: int | None = None,
    ) -> int:
        return topology.node_count()


class _Executor:
    def __init__(self, graph: _Topology) -> None:
        self.graph = graph
        self.completed: set[int] = set()

    def next_ready(self, graph: _Topology) -> list[int]:
        for idx in range(graph.node_count()):
            if idx not in self.completed:
                return [idx]
        return []

    def mark_completed(self, idx: int) -> None:
        self.completed.add(idx)

    def is_done(self) -> bool:
        return len(self.completed) >= self.graph.node_count()


class _CostedLoop:
    def __init__(self, node_idx: int, cost_usd: float) -> None:
        self.node_idx = node_idx
        self.total_cost_usd = cost_usd
        self.tool_call_count = 0
        self.tool_turn_count = 0
        self.executed_commands: list[str] = []

    async def run(self, task: str) -> str:
        return f"node-{self.node_idx}-result"


def _single_agent_pipeline(event_bus: Any | None = None) -> CognitiveOrchestrationPipeline:
    return CognitiveOrchestrationPipeline(
        router=_Router(system=1),
        engine=None,
        assigner=None,
        provider_pool=MagicMock(),
        event_bus=event_bus,
        llm_provider=_Provider(),
    )


def _multi_node_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    costs: dict[int, float],
    event_bus: Any | None = None,
) -> tuple[CognitiveOrchestrationPipeline, list[int]]:
    monkeypatch.setitem(
        sys.modules,
        "sage_core",
        SimpleNamespace(TopologyExecutor=_Executor),
    )

    created_loops: list[int] = []

    def create_node_agent_loop(**kwargs: Any) -> _CostedLoop:
        node_name = kwargs["node_name"]
        node_idx = int(node_name.split("-")[1])
        created_loops.append(node_idx)
        return _CostedLoop(node_idx, costs.get(node_idx, 0.0))

    monkeypatch.setattr(
        "sage.agent_loop_factory.create_node_agent_loop",
        create_node_agent_loop,
    )

    pipeline = CognitiveOrchestrationPipeline(
        router=_Router(system=3),
        engine=_Engine(_Topology(n_nodes=2)),
        assigner=_Assigner(),
        provider_pool=MagicMock(),
        event_bus=event_bus,
        llm_provider=_Provider(),
        agent_loop=MagicMock(),
        tool_registry=object(),
    )
    pipeline._build_topology_from_hint = MagicMock(return_value=None)
    return pipeline, created_loops


@pytest.mark.asyncio
async def test_pipeline_run_short_circuits_after_first_node_exceeds_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, created_loops = _multi_node_pipeline(
        monkeypatch,
        costs={0: 0.02, 1: 0.0},
    )

    result = await pipeline.run("expensive topology", budget_usd=0.01)

    assert BUDGET_SENTINEL in result
    assert created_loops == [0]
    assert pipeline.last_context.cost_tracker.total_spent == pytest.approx(0.02)


@pytest.mark.asyncio
async def test_execute_budget_exceeded_event_emits_via_pipeline_emit() -> None:
    bus = _CapturingBus()
    pipeline = _single_agent_pipeline(event_bus=bus)

    with patch.object(
        CostTracker,
        "is_over_budget",
        new_callable=PropertyMock,
        return_value=True,
    ):
        result = await pipeline.run("halt before execution", budget_usd=0.01)

    assert result == BUDGET_SENTINEL
    assert EXECUTE_BUDGET_EXCEEDED in bus.stages


@pytest.mark.asyncio
async def test_budget_zero_default_has_unlimited_tracker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the run starts with no budget cap, the orchestrator now creates
    an UNLIMITED ``CostTracker(budget_usd=0.0)`` so the CLI ``set_budget``
    command can tighten mid-run from infinity (cycle-13 K post-Phase-2.2
    Stage B lock 2026-05-07: ``CostTracker.tighten_remaining_budget`` is
    the root guard for the tightening-only invariant).

    The unlimited tracker is bookkeeping only — ``is_over_budget`` stays
    ``False``, ``remaining`` stays ``inf``, and no
    ``EXECUTE_BUDGET_EXCEEDED`` event is emitted.
    """
    import math

    monkeypatch.delenv("SAGE_TASK_BUDGET_USD", raising=False)
    bus = _CapturingBus()
    pipeline = _single_agent_pipeline(event_bus=bus)

    result = await pipeline.run("normal run")

    assert result == "Pipeline test response"
    tracker = pipeline.last_context.cost_tracker
    assert tracker is not None
    assert tracker.budget_usd == 0.0
    assert tracker.remaining == math.inf
    assert tracker.is_over_budget is False
    assert EXECUTE_BUDGET_EXCEEDED not in bus.stages


@pytest.mark.asyncio
async def test_constructor_reads_task_budget_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGE_TASK_BUDGET_USD", "5.00")
    pipeline = _single_agent_pipeline()

    await pipeline.run("env-budgeted run")

    assert pipeline.last_context.cost_tracker.budget_usd == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_budget_wins_before_strict_governance_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGE_STRICT_GOVERNANCE", "1")
    bus = _CapturingBus()
    pipeline = _single_agent_pipeline(event_bus=bus)
    ctx = SimpleNamespace(
        task="verification should not win",
        topology=None,
        system=1,
        result="",
        cost=0.0,
        tool_call_count=0,
        tool_turn_count=0,
        executed_commands=[],
        verification_passed=False,
        bandit_decision_id=None,
        cost_tracker=CostTracker(budget_usd=0.01),
    )

    with patch.object(
        CostTracker,
        "is_over_budget",
        new_callable=PropertyMock,
        return_value=True,
    ):
        result_ctx = await execute(pipeline, ctx)

    assert result_ctx.result == BUDGET_SENTINEL
    assert EXECUTE_BUDGET_EXCEEDED in bus.stages
    assert "EXECUTE_HALTED_UNVERIFIED" not in bus.stages


# ────────────────────────────────────────────────────────────────────
# Stage B — pipeline.tighten_budget façade integration
# ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tighten_budget_before_prompt_returns_budget_before_prompt() -> None:
    """No active run → ``pipeline.tighten_budget()`` rejects with
    ``budget_before_prompt`` (cgpro Stage B lock 2026-05-07)."""
    pipeline = _single_agent_pipeline()
    assert pipeline._active_context is None
    result = pipeline.tighten_budget(5.0)
    assert result.accepted is False
    assert result.reason == "budget_before_prompt"


def test_tighten_budget_with_active_context_updates_tracker() -> None:
    """Façade test: with an active context whose cost_tracker has a finite
    cap, ``pipeline.tighten_budget`` rebases the cap to the new remaining."""
    pipeline = _single_agent_pipeline()
    tracker = CostTracker(budget_usd=10.0)
    tracker.record_spend(0.0)
    pipeline._active_context = SimpleNamespace(cost_tracker=tracker)

    result = pipeline.tighten_budget(3.0)

    assert result.accepted is True
    assert result.reason == "budget_tightened"
    assert tracker.budget_usd == pytest.approx(3.0)
    assert tracker.remaining == pytest.approx(3.0)


def test_tighten_budget_loosen_attempt_rejected_keeps_old_cap() -> None:
    """Façade test: loosen attempt leaves cap unchanged + returns reason
    ``budget_loosen_rejected``."""
    pipeline = _single_agent_pipeline()
    tracker = CostTracker(budget_usd=5.0)
    pipeline._active_context = SimpleNamespace(cost_tracker=tracker)

    result = pipeline.tighten_budget(100.0)  # loosen attempt

    assert result.accepted is False
    assert result.reason == "budget_loosen_rejected"
    assert tracker.budget_usd == 5.0  # unchanged


@pytest.mark.asyncio
async def test_active_context_cleared_in_finally_after_run() -> None:
    """The orchestrator's ``finally`` block clears ``pipeline._active_context``
    so a stale ctx can't be mutated after the run finishes (Stage B lock)."""
    pipeline = _single_agent_pipeline()
    assert pipeline._active_context is None
    await pipeline.run("any task")
    assert pipeline._active_context is None
