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
