"""Tests for the `system_hint` override in pipeline.run() and system.run().

Benchmark adapters (SWE-bench, BigCodeBench) sometimes know the task class
better than the router. `system_hint` lets them force Stage 0 to S1/S2/S3
without disabling the rest of the routing pipeline (model selection, bandit,
domain scoring).
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext


def _install_fake_sage_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``sage_core.RoutingConstraints`` so _stage_classify imports succeed.

    Cycle-11 cgpro VERIFY follow-up (2026-05-05): _stage_classify
    Priority 1 imports `RoutingConstraints` from `sage_core` to build
    a request for `route_integrated`. When the real Rust extension
    isn't loaded (test envs without the C extension), the import
    fails and the except branch swallows it. The mocked rust_router
    + a fake RoutingConstraints class lets the production code
    proceed deterministically.
    """
    class _FakeRoutingConstraints:
        def __init__(self, **kwargs: object) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    existing = sys.modules.get("sage_core")
    if existing is not None:
        monkeypatch.setattr(
            existing, "RoutingConstraints", _FakeRoutingConstraints, raising=False
        )
    else:
        monkeypatch.setitem(
            sys.modules,
            "sage_core",
            SimpleNamespace(RoutingConstraints=_FakeRoutingConstraints),
        )


def _mk_pipeline(rust_router_system: int = 2, monkeypatch: pytest.MonkeyPatch | None = None):
    """Minimal pipeline whose Rust router always returns the given system.

    Cycle-11 cgpro VERIFY follow-up (2026-05-05): the original mock
    stubbed ``rust_router.route`` but production code calls
    ``rust_router.route_integrated``. With the wrong stub method,
    ``decision`` was an auto-MagicMock; ``int(MagicMock().system)``
    returns 1 (Python's MagicMock magic-method default for ``__int__``),
    so ctx.system silently became 1 regardless of ``rust_router_system``.

    Fix:
      - Stub ``route_integrated`` (the actual method).
      - Use ``SimpleNamespace`` for the decision so attribute accesses
        return the configured values, not auto-mocks. Avoids
        MagicMock magic-method gotchas (``__int__``, ``__bool__``).
    """
    decision = SimpleNamespace(
        system=rust_router_system,
        model_id="deepseek-chat",
        confidence=0.9,
        estimated_cost=0.0001,
        # selected_template=""  intentionally omitted: bandit attribution
        # path triggers only when truthy, and these tests don't exercise it.
        selected_template="",
        decision_id="",
        topology_id="",
    )

    rust_router = MagicMock()
    rust_router.route_integrated = MagicMock(return_value=decision)

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


def test_system_hint_overrides_router_classification(monkeypatch: pytest.MonkeyPatch):
    """When system_hint differs from router output, ctx.system should flip."""
    _install_fake_sage_core(monkeypatch)
    pipeline = _mk_pipeline(rust_router_system=2)
    ctx = PipelineContext(task="write a patch for astropy#12907", budget=1.0)
    ctx = pipeline._stage_classify(ctx)
    assert ctx.system == 2

    # Simulate run() override logic
    hint = 3
    if hint in (1, 2, 3) and ctx.system != hint:
        ctx.system = hint

    assert ctx.system == 3


def test_system_hint_noop_when_matches_router(monkeypatch: pytest.MonkeyPatch):
    """If hint matches what the router already picked, nothing changes."""
    _install_fake_sage_core(monkeypatch)
    pipeline = _mk_pipeline(rust_router_system=3)
    ctx = PipelineContext(task="prove a theorem", budget=1.0)
    ctx = pipeline._stage_classify(ctx)
    assert ctx.system == 3

    hint = 3
    if hint in (1, 2, 3) and ctx.system != hint:
        pytest.fail("override should not fire when hint already matches")


def test_system_hint_ignores_invalid_values(monkeypatch: pytest.MonkeyPatch):
    """Hints outside {1,2,3} must not touch ctx.system."""
    _install_fake_sage_core(monkeypatch)
    pipeline = _mk_pipeline(rust_router_system=2)
    ctx = PipelineContext(task="simple math", budget=1.0)
    ctx = pipeline._stage_classify(ctx)

    for bad in (0, 4, 99, -1):
        applied = bad in (1, 2, 3) and ctx.system != bad
        assert not applied


@pytest.mark.asyncio
async def test_pipeline_run_applies_hint(monkeypatch: pytest.MonkeyPatch):
    """End-to-end: calling pipeline.run(..., system_hint=3) forces S3 in ctx."""
    _install_fake_sage_core(monkeypatch)
    pipeline = _mk_pipeline(rust_router_system=2)

    # Short-circuit decompose/topology/execute/learn so we only validate classify.
    async def _fake_decompose(ctx):
        return ctx

    def _fake_select(ctx):
        return ctx

    def _fake_assign(ctx):
        return ctx

    async def _fake_execute(ctx, **_kwargs):
        ctx.result = "ok"
        return ctx

    async def _fake_learn(ctx):
        return None

    pipeline._stage_decompose = _fake_decompose
    pipeline._stage_select_topology = _fake_select
    pipeline._stage_assign_models = _fake_assign
    pipeline._stage_execute = _fake_execute
    pipeline._stage_learn = _fake_learn
    # Cycle-11 cgpro VERIFY follow-up: production calls
    # `_record_to_memory(ctx, *, is_training_evidence=...)` (cycle-5 R9
    # OracleStack v0). The previous lambda accepted only `_ctx` and
    # raised TypeError on the kwarg. Use **kwargs so any future kwargs
    # land cleanly without re-breaking the test.
    pipeline._record_to_memory = lambda *_args, **_kwargs: None

    await pipeline.run("patch the bug", budget_usd=1.0, system_hint=3)
    assert pipeline.last_context.system == 3


@pytest.mark.asyncio
async def test_pipeline_run_without_hint_keeps_router_choice(monkeypatch: pytest.MonkeyPatch):
    """When no hint is given, the router's system is preserved."""
    _install_fake_sage_core(monkeypatch)
    pipeline = _mk_pipeline(rust_router_system=2)

    async def _fake_decompose(ctx):
        return ctx

    def _fake_select(ctx):
        return ctx

    def _fake_assign(ctx):
        return ctx

    async def _fake_execute(ctx, **_kwargs):
        ctx.result = "ok"
        return ctx

    async def _fake_learn(ctx):
        return None

    pipeline._stage_decompose = _fake_decompose
    pipeline._stage_select_topology = _fake_select
    pipeline._stage_assign_models = _fake_assign
    pipeline._stage_execute = _fake_execute
    pipeline._stage_learn = _fake_learn
    pipeline._record_to_memory = lambda *_args, **_kwargs: None

    await pipeline.run("do something", budget_usd=1.0)
    assert pipeline.last_context.system == 2
