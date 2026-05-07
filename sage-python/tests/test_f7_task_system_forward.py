"""Integration test for F7 — pipeline forwards overall task_system to the
Rust ModelAssigner so role-aware tier promotion fires.

Paired with the Rust-side tests in sage-core/src/routing/model_assigner.rs
(test_effective_system_*, test_s3_task_pushes_planner_to_reasoner_model).
"""
from __future__ import annotations

from typing import Any

import pytest
from unittest.mock import MagicMock

from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext
from sage.pipeline_v2.assign_models import assign_models


class _SpyAssigner:
    """Captures the exact args the pipeline forwards to assigner.assign_models."""

    def __init__(self) -> None:
        self.last_call: dict[str, Any] = {}
        self.call_count = 0

    def assign_models(
        self,
        topology: Any,
        domain: str,
        budget: float,
        hints: Any = None,
        task_system: int | None = None,
    ) -> int:
        self.call_count += 1
        self.last_call = {
            "domain": domain,
            "budget": budget,
            "hints": hints,
            "task_system": task_system,
        }
        return topology.node_count() if hasattr(topology, "node_count") else 0


class _Topology:
    def __init__(self, n: int = 3) -> None:
        self._n = n
        self._nodes = [MagicMock(model_id=f"m{i}", max_cost_usd=0.0) for i in range(n)]

    def node_count(self) -> int:
        return self._n

    def get_node(self, i: int):
        return self._nodes[i]


def _mk_pipeline_with_spy():
    spy = _SpyAssigner()
    pipeline = CognitiveOrchestrationPipeline(
        router=MagicMock(),
        engine=None,
        assigner=spy,
        provider_pool=MagicMock(),
        llm_provider=MagicMock(),
        llm_config=MagicMock(),
    )
    return pipeline, spy


def test_assigner_receives_task_system_when_ctx_is_s3():
    """F7 contract: ctx.system=3 must reach assigner.assign_models(task_system=3)."""
    pipeline, spy = _mk_pipeline_with_spy()
    ctx = PipelineContext(task="fix bug", budget=10.0)
    ctx.system = 3
    ctx.domain = "code"
    ctx.topology = _Topology(3)
    ctx = assign_models(pipeline, ctx)
    assert spy.call_count == 1
    assert spy.last_call["task_system"] == 3


def test_assigner_receives_task_system_when_ctx_is_s2():
    pipeline, spy = _mk_pipeline_with_spy()
    ctx = PipelineContext(task="add tests", budget=5.0)
    ctx.system = 2
    ctx.domain = "code"
    ctx.topology = _Topology(2)
    ctx = assign_models(pipeline, ctx)
    assert spy.last_call["task_system"] == 2


def test_assigner_receives_task_system_when_ctx_is_s1():
    pipeline, spy = _mk_pipeline_with_spy()
    ctx = PipelineContext(task="say hi", budget=1.0)
    ctx.system = 1
    ctx.domain = ""
    ctx.topology = _Topology(1)
    ctx = assign_models(pipeline, ctx)
    assert spy.last_call["task_system"] == 1


def test_assigner_receives_none_when_system_not_set():
    """ctx.system starts at 0 (unset). Pipeline forwards None, not 0."""
    pipeline, spy = _mk_pipeline_with_spy()
    ctx = PipelineContext(task="x", budget=1.0)
    # ctx.system default is 0 — out of (1,2,3), so must be None-forwarded.
    ctx.topology = _Topology(1)
    ctx = assign_models(pipeline, ctx)
    assert spy.last_call["task_system"] is None


def test_assigner_receives_none_when_system_is_garbage():
    """Defensive: unexpected ctx.system values (e.g. 4, -1) must not leak."""
    pipeline, spy = _mk_pipeline_with_spy()
    ctx = PipelineContext(task="x", budget=1.0)
    ctx.system = 99
    ctx.topology = _Topology(1)
    ctx = assign_models(pipeline, ctx)
    assert spy.last_call["task_system"] is None


def test_domain_and_budget_still_forwarded_correctly():
    """Sanity: adding task_system did not break the other kwargs."""
    pipeline, spy = _mk_pipeline_with_spy()
    ctx = PipelineContext(task="x", budget=7.5)
    ctx.system = 3
    ctx.domain = "math"
    ctx.topology = _Topology(2)
    ctx = assign_models(pipeline, ctx)
    assert spy.last_call["domain"] == "math"
    assert spy.last_call["budget"] == 7.5


# ── Option C (cgpro 2026-05-07): tier-constrained model assignment ────


def test_budget_tier_clamps_task_system_3_to_1():
    """S3 task under budget tier must receive task_system=1 (cheapest models)."""
    pipeline, spy = _mk_pipeline_with_spy()
    pipeline._llm_tier = "budget"
    ctx = PipelineContext(task="fix complex bug", budget=10.0)
    ctx.system = 3
    ctx.domain = "code"
    ctx.topology = _Topology(3)
    ctx = assign_models(pipeline, ctx)
    assert spy.last_call["task_system"] == 1, (
        f"budget tier must clamp S3→S1, got task_system={spy.last_call['task_system']}"
    )


def test_budget_tier_clamps_task_system_2_to_1():
    """S2 task under budget tier must receive task_system=1."""
    pipeline, spy = _mk_pipeline_with_spy()
    pipeline._llm_tier = "budget"
    ctx = PipelineContext(task="medium task", budget=5.0)
    ctx.system = 2
    ctx.domain = "code"
    ctx.topology = _Topology(2)
    ctx = assign_models(pipeline, ctx)
    assert spy.last_call["task_system"] == 1


def test_budget_tier_does_not_lift_task_system_1():
    """S1 task under budget tier stays S1 (no up-clamp)."""
    pipeline, spy = _mk_pipeline_with_spy()
    pipeline._llm_tier = "budget"
    ctx = PipelineContext(task="simple task", budget=1.0)
    ctx.system = 1
    ctx.domain = ""
    ctx.topology = _Topology(1)
    ctx = assign_models(pipeline, ctx)
    assert spy.last_call["task_system"] == 1


def test_fast_tier_clamps_task_system_3_to_2():
    """S3 task under fast tier must clamp to S2."""
    pipeline, spy = _mk_pipeline_with_spy()
    pipeline._llm_tier = "fast"
    ctx = PipelineContext(task="complex task", budget=10.0)
    ctx.system = 3
    ctx.domain = "code"
    ctx.topology = _Topology(3)
    ctx = assign_models(pipeline, ctx)
    assert spy.last_call["task_system"] == 2


def test_reasoner_tier_does_not_clamp_task_system_3():
    """S3 task under reasoner tier stays S3 (no clamp)."""
    pipeline, spy = _mk_pipeline_with_spy()
    pipeline._llm_tier = "reasoner"
    ctx = PipelineContext(task="complex task", budget=10.0)
    ctx.system = 3
    ctx.domain = "code"
    ctx.topology = _Topology(3)
    ctx = assign_models(pipeline, ctx)
    assert spy.last_call["task_system"] == 3


def test_no_tier_does_not_clamp_task_system():
    """Unset tier (None or empty) must not clamp task_system."""
    pipeline, spy = _mk_pipeline_with_spy()
    pipeline._llm_tier = ""
    ctx = PipelineContext(task="complex task", budget=10.0)
    ctx.system = 3
    ctx.domain = "code"
    ctx.topology = _Topology(3)
    ctx = assign_models(pipeline, ctx)
    assert spy.last_call["task_system"] == 3


def test_budget_tier_none_task_system_stays_none():
    """When ctx.system is unset (→None), budget tier must not clamp None."""
    pipeline, spy = _mk_pipeline_with_spy()
    pipeline._llm_tier = "budget"
    ctx = PipelineContext(task="x", budget=1.0)
    ctx.topology = _Topology(1)
    ctx = assign_models(pipeline, ctx)
    assert spy.last_call["task_system"] is None
