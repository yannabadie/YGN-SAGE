"""B1: pipeline-level OTel span integration tests using InMemorySpanExporter."""
from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def in_memory_exporter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAGE_OTEL_EXPORTER", "console")
    import sage.observability as obs
    importlib.reload(obs)
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    obs._TRACER = trace.get_tracer("sage", "test")
    yield exporter
    exporter.clear()


@pytest.mark.asyncio
async def test_pipeline_run_emits_top_level_span(in_memory_exporter) -> None:
    """pipeline.run() emits a top-level sage.pipeline.run span with op=invoke_agent."""
    from sage.pipeline import CognitiveOrchestrationPipeline
    pipeline = CognitiveOrchestrationPipeline.__new__(CognitiveOrchestrationPipeline)
    # Stub instance attributes accessed before / during stage calls
    pipeline.budget_usd = 0.0
    pipeline._agent_loop = None
    pipeline._build_write_gate = MagicMock(return_value=None)
    pipeline._emit = MagicMock()
    pipeline._record_to_memory = MagicMock()
    # Stub all the stages to no-op
    pipeline._stage_classify = MagicMock(side_effect=lambda ctx: ctx)
    pipeline._stage_decompose = AsyncMock(side_effect=lambda ctx: ctx)
    pipeline._stage_select_topology = MagicMock(side_effect=lambda ctx: ctx)
    pipeline._stage_assign_models = MagicMock(side_effect=lambda ctx: ctx)
    pipeline._stage_execute = AsyncMock(side_effect=lambda ctx: ctx)
    pipeline._stage_learn = AsyncMock(side_effect=lambda ctx: ctx)
    pipeline._emit_budget_exceeded = MagicMock()

    await pipeline.run("hello")

    spans = in_memory_exporter.get_finished_spans()
    top = next((s for s in spans if s.name == "sage.pipeline.run"), None)
    assert top is not None, f"missing sage.pipeline.run span; got {[s.name for s in spans]}"
    assert top.attributes["gen_ai.operation.name"] == "invoke_agent"


@pytest.mark.asyncio
async def test_pipeline_run_emits_six_stage_spans(in_memory_exporter) -> None:
    """All 6 SAGE stages emit child spans with sage.* operation names (real methods, not mocks)."""
    from sage.pipeline import CognitiveOrchestrationPipeline
    pipeline = CognitiveOrchestrationPipeline.__new__(CognitiveOrchestrationPipeline)
    # Infrastructure stubs — same 5 as the top-level span test
    pipeline.budget_usd = 0.0
    pipeline._agent_loop = None
    pipeline._build_write_gate = MagicMock(return_value=None)
    pipeline._emit = MagicMock()
    pipeline._record_to_memory = MagicMock()
    # All router/engine/bandit/provider attributes → None so every stage
    # hits its early-return / fast-path branch without external dependencies.
    pipeline._rust_router = None
    pipeline.router = None
    pipeline.engine = None
    pipeline.assigner = None
    pipeline.bandit = None
    pipeline.quality_estimator = None
    pipeline.prm = None
    pipeline.controller = None
    pipeline.provider_pool = None
    pipeline.tool_registry = None
    pipeline.event_bus = None
    pipeline.llm_provider = None
    pipeline.llm_config = None
    pipeline.consolidator = None
    pipeline._task_count = 0
    # Defensive: ensure _build_topology_from_hint can't acquire real sage_core
    pipeline._build_topology_from_hint = MagicMock(return_value=None)

    # system_hint=1 forces S1 on classify-override; S1 takes the short-circuit
    # branch in decompose (omega=1), select_topology (no topology), assign_models
    # (topology=None early-return), execute (topology=None + no llm_provider),
    # and learn (empty result → quality=0.0, no bandit/engine).
    await pipeline.run("hello", system_hint=1)

    spans = in_memory_exporter.get_finished_spans()
    span_names = {s.name for s in spans}
    expected = {
        "sage.classify",
        "sage.decompose",
        "sage.topology_select",
        "sage.assign_models",
        "sage.execute",
        "sage.learn",
    }
    missing = expected - span_names
    assert not missing, f"missing stage spans: {missing}; got {span_names}"

    # Verify gen_ai.operation.name equals the span name for each stage span
    op_for = {
        s.name: s.attributes["gen_ai.operation.name"]
        for s in spans
        if s.name in expected
    }
    for stage in expected:
        assert op_for[stage] == stage, (
            f"{stage} should have op={stage}, got {op_for.get(stage)!r}"
        )
