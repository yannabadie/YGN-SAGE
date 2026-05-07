"""Regression tests for cycle-12 P6-A Phase B (cgpro DESIGN 2026-05-05).  # narrative-guard: allow historical-commit-reference

Locks the structural-isolation property that REPLACES the cycle-11
P6-B asyncio lock + ContextVar reentry guard:

  Pre P6-A Phase B (cycle-11): the singleton AgentLoop was mutated  # narrative-guard: allow historical-commit-reference
  in-place during bypass; concurrency was prevented by an
  asyncio.Lock + ContextVar reentry guard (the band-aid).

  Post P6-A Phase B (cycle-12): each bypass run gets a FRESH  # narrative-guard: allow historical-commit-reference
  AgentLoop instance from `create_bypass_agent_loop()`. The
  singleton is never mutated. Concurrency is structurally safe
  because there's no shared mutable state. Recursion is safe because
  each call creates an independent instance.

These tests prove the structural property by asserting:
  - Singleton's 12 mutation-prone fields are UNCHANGED before/after
    a bypass call (success path).
  - Same property after an exception inside the bypass.
  - Concurrent bypass calls produce N independent loops; the
    factory is called N times; the singleton's .run() is NEVER
    called (only the per-run instances' .run()).
  - Recursive bypass (nested pipeline.run() from inside a bypass)
    completes without deadlock -- there's no lock to deadlock on.

If a future commit reverts P6-A Phase B (re-introduces the singleton  # narrative-guard: allow historical-commit-reference
mutation block), these tests fail loudly.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sage.agent import AgentConfig
from sage.agent_loop import AgentLoop
from sage.llm.base import LLMConfig, LLMResponse
from sage.pipeline import CognitiveOrchestrationPipeline as Pipeline
from sage.pipeline import PipelineContext
from sage.pipeline_v2.execute import execute


@pytest.fixture(autouse=True)
def _legacy_oracle_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep full pipeline recursion on the legacy non-oracle path."""
    monkeypatch.setenv("SAGE_ORACLE", "0")


class _StaticProvider:
    name = "singleton-provider"
    model_id = "singleton-provider-model"

    async def generate(self, messages: Any, config: Any = None, **kwargs: Any) -> LLMResponse:
        return LLMResponse(content="provider-output")


class _MockRouter:
    def __init__(self, system: int = 1) -> None:
        self.system = system

    def assess_complexity(self, task: str) -> SimpleNamespace:
        return SimpleNamespace(system=self.system)

    def route(self, profile: SimpleNamespace) -> SimpleNamespace:
        return SimpleNamespace(system=profile.system, confidence=0.9, model_id="")


class _MockQualityEstimator:
    def estimate(self, task: str, result: str, latency_s: float = 0.0) -> float:
        return 0.9


class _BypassLoop:
    def __init__(
        self,
        *,
        result: str = "bypass-output",
        side_effect: Any | None = None,
    ) -> None:
        self.total_cost_usd = 0.25
        self.tool_call_count = 2
        self.tool_turn_count = 1
        self.executed_commands = ["echo structural"]
        if side_effect is not None:
            self.run = AsyncMock(side_effect=side_effect)
        else:
            self.run = AsyncMock(return_value=result)


def _make_loop_with_prior_state() -> AgentLoop:
    """Real AgentLoop singleton with distinctive pre-bypass state."""
    config = AgentConfig(
        name="boot-singleton",
        llm=LLMConfig(provider="singleton", model="singleton-model"),
        system_prompt="You are a test singleton.",
        max_steps=123,
        validation_level=99,
        stall_after_tool_steps=77,
        dangerous_tools=True,
    )
    loop = AgentLoop(config=config, llm_provider=_StaticProvider())
    loop._skip_routing = False
    loop._current_topology = object()
    loop.write_gate = object()
    loop.gate_current_task = "prior-task"
    loop.gate_source_tier = "prior-tier"
    loop._on_drift = lambda *args, **kwargs: None
    loop._run_frame_builder = object()
    loop._runtime_node_run_id = "prior-node-run"
    return loop


def _make_pipeline_for_bypass(singleton: AgentLoop) -> Pipeline:
    pipeline = Pipeline(
        router=_MockRouter(system=1),
        engine=None,
        assigner=None,
        provider_pool=None,
        bandit=None,
        quality_estimator=_MockQualityEstimator(),
        event_bus=MagicMock(),
        llm_provider=None,
        llm_config=None,
        agent_loop=singleton,
    )
    pipeline.write_gate = object()
    pipeline._last_routing_decision = None
    return pipeline


def _make_ctx(task: str = "task") -> PipelineContext:
    return PipelineContext(task=task, system=2, topology=None)


def _snap(value: Any) -> tuple[str, Any]:
    if isinstance(value, (str, int, float, bool, type(None))):
        return ("value", value)
    return ("id", id(value))


def _singleton_state(loop: AgentLoop) -> dict[str, tuple[str, Any]]:
    return {
        "_llm": _snap(loop._llm),
        "_skip_routing": _snap(loop._skip_routing),
        "_current_topology": _snap(loop._current_topology),
        "write_gate": _snap(loop.write_gate),
        "gate_current_task": _snap(loop.gate_current_task),
        "gate_source_tier": _snap(loop.gate_source_tier),
        "_on_drift": _snap(loop._on_drift),
        "_run_frame_builder": _snap(loop._run_frame_builder),
        "_runtime_node_run_id": _snap(loop._runtime_node_run_id),
        "config.llm": _snap(loop.config.llm),
        "config.validation_level": _snap(loop.config.validation_level),
        "config.max_steps": _snap(loop.config.max_steps),
        "config.stall_after_tool_steps": _snap(loop.config.stall_after_tool_steps),
    }


@pytest.mark.asyncio
async def test_singleton_unchanged_after_successful_bypass() -> None:
    singleton = _make_loop_with_prior_state()
    singleton.run = AsyncMock(side_effect=AssertionError("singleton.run must not be called"))
    pipeline = _make_pipeline_for_bypass(singleton)
    ctx = _make_ctx("success-task")
    before = _singleton_state(singleton)
    bypass_loop = _BypassLoop(result="per-run-output")

    with patch(
        "sage.pipeline_v2.execute.create_bypass_agent_loop",
        return_value=bypass_loop,
        create=True,
    ) as factory:
        result_ctx = await execute(pipeline, ctx)

    assert result_ctx.result == "per-run-output"
    assert _singleton_state(singleton) == before
    factory.assert_called_once()
    assert factory.call_args.kwargs["singleton"] is singleton
    singleton.run.assert_not_called()
    bypass_loop.run.assert_awaited_once_with("success-task")


@pytest.mark.asyncio
async def test_singleton_unchanged_after_bypass_raises() -> None:
    singleton = _make_loop_with_prior_state()
    singleton.run = AsyncMock(side_effect=AssertionError("singleton.run must not be called"))
    pipeline = _make_pipeline_for_bypass(singleton)
    before = _singleton_state(singleton)
    bypass_loop = _BypassLoop(side_effect=RuntimeError("bypass boom"))

    with patch(
        "sage.pipeline_v2.execute.create_bypass_agent_loop",
        return_value=bypass_loop,
        create=True,
    ) as factory:
        with pytest.raises(RuntimeError, match="bypass boom"):
            await execute(pipeline, _make_ctx("raise-task"))

    assert _singleton_state(singleton) == before
    factory.assert_called_once()
    singleton.run.assert_not_called()
    bypass_loop.run.assert_awaited_once_with("raise-task")


@pytest.mark.asyncio
async def test_concurrent_bypass_calls_create_independent_loops() -> None:
    singleton = _make_loop_with_prior_state()
    singleton.run = AsyncMock(side_effect=AssertionError("singleton.run must not be called"))
    pipeline = _make_pipeline_for_bypass(singleton)
    loops: list[_BypassLoop] = []

    async def _run_once(task: str) -> str:
        await asyncio.sleep(0)
        return f"done:{task}"

    def _factory_side_effect(**kwargs: Any) -> _BypassLoop:
        loop = _BypassLoop(side_effect=_run_once)
        loops.append(loop)
        return loop

    with patch(
        "sage.pipeline_v2.execute.create_bypass_agent_loop",
        side_effect=_factory_side_effect,
        create=True,
    ) as factory:
        ctxs = [_make_ctx(f"task-{idx}") for idx in range(5)]
        results = await asyncio.gather(*(execute(pipeline, ctx) for ctx in ctxs))

    assert factory.call_count == 5
    assert len(loops) == 5
    assert len({id(loop) for loop in loops}) == 5
    singleton.run.assert_not_called()
    for idx, loop in enumerate(loops):
        loop.run.assert_awaited_once_with(f"task-{idx}")
        assert results[idx].result == f"done:task-{idx}"


@pytest.mark.asyncio
async def test_recursive_bypass_no_deadlock() -> None:
    singleton = _make_loop_with_prior_state()
    pipeline = _make_pipeline_for_bypass(singleton)

    async def _legacy_singleton_run(task: str) -> str:
        return await pipeline.run("inner recursive task", budget_usd=0.0, system_hint=1)

    singleton.run = AsyncMock(side_effect=_legacy_singleton_run)

    inner_loop = _BypassLoop(result="inner-result")

    async def _outer_run(task: str) -> str:
        inner = await pipeline.run("inner recursive task", budget_usd=0.0, system_hint=1)
        assert inner == "inner-result"
        return "outer-result"

    outer_loop = _BypassLoop(side_effect=_outer_run)

    with patch(
        "sage.pipeline_v2.execute.create_bypass_agent_loop",
        side_effect=[outer_loop, inner_loop],
        create=True,
    ) as factory:
        result = await asyncio.wait_for(
            pipeline.run("outer recursive task", budget_usd=0.0, system_hint=1),
            timeout=5.0,
        )

    assert result == "outer-result"
    assert factory.call_count == 2
    singleton.run.assert_not_called()
    outer_loop.run.assert_awaited_once_with("outer recursive task")
    inner_loop.run.assert_awaited_once_with("inner recursive task")
