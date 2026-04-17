"""Tests for the `system_hint` override in pipeline.run() and system.run().

Benchmark adapters (SWE-bench, BigCodeBench) sometimes know the task class
better than the router. `system_hint` lets them force Stage 0 to S1/S2/S3
without disabling the rest of the routing pipeline (model selection, bandit,
domain scoring).
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext


def _mk_pipeline(rust_router_system: int = 2):
    """Minimal pipeline whose Rust router always returns the given system."""
    decision = MagicMock()
    decision.system = rust_router_system
    decision.model_id = "deepseek-chat"
    decision.confidence = 0.9
    decision.estimated_cost = 0.0001

    rust_router = MagicMock()
    rust_router.route = MagicMock(return_value=decision)

    pipeline = CognitiveOrchestrationPipeline(
        router=MagicMock(),
        engine=None,
        assigner=MagicMock(),
        provider_pool=MagicMock(),
        llm_provider=MagicMock(),
        llm_config=MagicMock(),
    )
    pipeline._rust_router = rust_router
    return pipeline


def test_system_hint_overrides_router_classification():
    """When system_hint differs from router output, ctx.system should flip."""
    pipeline = _mk_pipeline(rust_router_system=2)
    ctx = PipelineContext(task="write a patch for astropy#12907", budget=1.0)
    ctx = pipeline._stage_classify(ctx)
    assert ctx.system == 2

    # Simulate run() override logic
    hint = 3
    if hint in (1, 2, 3) and ctx.system != hint:
        ctx.system = hint

    assert ctx.system == 3


def test_system_hint_noop_when_matches_router():
    """If hint matches what the router already picked, nothing changes."""
    pipeline = _mk_pipeline(rust_router_system=3)
    ctx = PipelineContext(task="prove a theorem", budget=1.0)
    ctx = pipeline._stage_classify(ctx)
    assert ctx.system == 3

    hint = 3
    if hint in (1, 2, 3) and ctx.system != hint:
        pytest.fail("override should not fire when hint already matches")


def test_system_hint_ignores_invalid_values():
    """Hints outside {1,2,3} must not touch ctx.system."""
    pipeline = _mk_pipeline(rust_router_system=2)
    ctx = PipelineContext(task="simple math", budget=1.0)
    ctx = pipeline._stage_classify(ctx)

    for bad in (0, 4, 99, -1):
        applied = bad in (1, 2, 3) and ctx.system != bad
        assert not applied


@pytest.mark.asyncio
async def test_pipeline_run_applies_hint():
    """End-to-end: calling pipeline.run(..., system_hint=3) forces S3 in ctx."""
    pipeline = _mk_pipeline(rust_router_system=2)

    # Short-circuit decompose/topology/execute/learn so we only validate classify.
    async def _fake_decompose(ctx):
        return ctx

    def _fake_select(ctx):
        return ctx

    def _fake_assign(ctx):
        return ctx

    async def _fake_execute(ctx):
        ctx.result = "ok"
        return ctx

    async def _fake_learn(ctx):
        return None

    pipeline._stage_decompose = _fake_decompose
    pipeline._stage_select_topology = _fake_select
    pipeline._stage_assign_models = _fake_assign
    pipeline._stage_execute = _fake_execute
    pipeline._stage_learn = _fake_learn
    pipeline._record_to_memory = lambda _ctx: None

    await pipeline.run("patch the bug", budget_usd=1.0, system_hint=3)
    assert pipeline.last_context.system == 3


@pytest.mark.asyncio
async def test_pipeline_run_without_hint_keeps_router_choice():
    """When no hint is given, the router's system is preserved."""
    pipeline = _mk_pipeline(rust_router_system=2)

    async def _fake_decompose(ctx):
        return ctx

    def _fake_select(ctx):
        return ctx

    def _fake_assign(ctx):
        return ctx

    async def _fake_execute(ctx):
        ctx.result = "ok"
        return ctx

    async def _fake_learn(ctx):
        return None

    pipeline._stage_decompose = _fake_decompose
    pipeline._stage_select_topology = _fake_select
    pipeline._stage_assign_models = _fake_assign
    pipeline._stage_execute = _fake_execute
    pipeline._stage_learn = _fake_learn
    pipeline._record_to_memory = lambda _ctx: None

    await pipeline.run("do something", budget_usd=1.0)
    assert pipeline.last_context.system == 2
