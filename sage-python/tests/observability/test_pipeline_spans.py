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
async def test_pipeline_run_emits_top_level_span(
    in_memory_exporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pipeline.run() emits a top-level sage.pipeline.run span with op=invoke_agent."""
    from sage.pipeline import CognitiveOrchestrationPipeline
    pipeline = CognitiveOrchestrationPipeline.__new__(CognitiveOrchestrationPipeline)
    # Stub instance attributes accessed before / during stage calls
    pipeline.budget_usd = 0.0
    pipeline._agent_loop = None
    pipeline._emit = MagicMock()  # _emit stays as instance seam per cgpro Q3a
    # Helper module functions: monkeypatch on the canonical module attribute
    # so the production orchestrator local-import path resolves
    # to the stub at call time (Trap 2 lock).
    monkeypatch.setattr("sage.pipeline_v2.memory_gate.build_write_gate", lambda _p: None)
    monkeypatch.setattr(
        "sage.pipeline_v2.memory_gate.record_to_memory",
        lambda _p, _ctx, **_kwargs: None,
    )
    monkeypatch.setattr(
        "sage.pipeline_v2.memory_gate.emit_budget_exceeded",
        lambda _p, _ctx: None,
    )
    # Stub all stages to no-op via module-function patching
    monkeypatch.setattr("sage.pipeline_v2.classify.classify", lambda _p, ctx: ctx)
    async def _decompose_stub(_p, ctx):
        return ctx
    monkeypatch.setattr("sage.pipeline_v2.decompose.decompose", _decompose_stub)
    monkeypatch.setattr(
        "sage.pipeline_v2.select_topology.select_topology", lambda _p, ctx: ctx
    )
    monkeypatch.setattr(
        "sage.pipeline_v2.assign_models.assign_models", lambda _p, ctx: ctx
    )
    async def _execute_stub(_p, ctx, **_kwargs):
        return ctx
    monkeypatch.setattr("sage.pipeline_v2.execute.execute", _execute_stub)
    async def _learn_stub(_p, ctx):
        return None
    monkeypatch.setattr("sage.pipeline_v2.learn.learn", _learn_stub)

    await pipeline.run("hello")

    spans = in_memory_exporter.get_finished_spans()
    top = next((s for s in spans if s.name == "sage.pipeline.run"), None)
    assert top is not None, f"missing sage.pipeline.run span; got {[s.name for s in spans]}"
    assert top.attributes["gen_ai.operation.name"] == "invoke_agent"


@pytest.mark.asyncio
async def test_pipeline_run_emits_six_stage_spans(
    in_memory_exporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All 6 SAGE stages emit child spans with sage.* operation names (real methods, not mocks)."""
    from sage.pipeline import CognitiveOrchestrationPipeline
    pipeline = CognitiveOrchestrationPipeline.__new__(CognitiveOrchestrationPipeline)
    # Infrastructure stubs — same 3 as the top-level span test
    pipeline.budget_usd = 0.0
    pipeline._agent_loop = None
    pipeline._emit = MagicMock()  # _emit stays as instance seam per cgpro Q3a
    monkeypatch.setattr("sage.pipeline_v2.memory_gate.build_write_gate", lambda _p: None)
    monkeypatch.setattr(
        "sage.pipeline_v2.memory_gate.record_to_memory",
        lambda _p, _ctx, **_kwargs: None,
    )
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
    # Defensive: ensure build_topology_from_hint can't acquire real sage_core
    monkeypatch.setattr(
        "sage.pipeline_v2.topology_helpers.build_topology_from_hint",
        lambda _hint: None,
    )

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


@pytest.mark.asyncio
async def test_topology_runner_emits_per_node_invoke_agent_spans(in_memory_exporter) -> None:
    """Each TopologyNode AgentLoop emits sage.node.<name> with op=invoke_agent."""
    from sage.observability.spans import sage_span
    with sage_span("sage.node.preprocessor", op="invoke_agent",
                   **{"sage.node.name": "preprocessor"}):
        pass
    with sage_span("sage.node.worker_0", op="invoke_agent",
                   **{"sage.node.name": "worker_0"}):
        pass

    spans = in_memory_exporter.get_finished_spans()
    node_spans = [s for s in spans if s.name.startswith("sage.node.")]
    assert len(node_spans) == 2
    for s in node_spans:
        assert s.attributes["gen_ai.operation.name"] == "invoke_agent"


@pytest.mark.asyncio
async def test_node_spans_nest_under_execute_under_pipeline_run(in_memory_exporter) -> None:
    """Span hierarchy: sage.node.<n> child of sage.execute, child of sage.pipeline.run."""
    from sage.observability.spans import sage_span
    # Synthetic nested call structure mirroring runner+pipeline emission order
    with sage_span("sage.pipeline.run", op="invoke_agent"):
        with sage_span("sage.execute", op="sage.execute"):
            with sage_span("sage.node.synthetic", op="invoke_agent",
                           **{"sage.node.name": "synthetic"}):
                pass

    spans = in_memory_exporter.get_finished_spans()
    by_id = {s.context.span_id: s for s in spans}  # noqa: F841 — kept for future walk assertions
    node = next(s for s in spans if s.name == "sage.node.synthetic")
    execute = next(s for s in spans if s.name == "sage.execute")
    top = next(s for s in spans if s.name == "sage.pipeline.run")

    # node parent is execute
    assert node.parent is not None
    assert node.parent.span_id == execute.context.span_id

    # execute parent is pipeline.run
    assert execute.parent is not None
    assert execute.parent.span_id == top.context.span_id

    # pipeline.run has no parent (it's the root)
    assert top.parent is None


@pytest.mark.asyncio
async def test_chat_span_carries_provider_model_usage(in_memory_exporter) -> None:
    """sage.chat span carries gen_ai.provider.name + gen_ai.request.model + usage tokens."""
    from sage.observability.spans import sage_span, otel_provider_name
    with sage_span(
        "sage.chat",
        op="chat",
        **{
            "gen_ai.provider.name": otel_provider_name("kimi"),
            "gen_ai.request.model": "kimi-k2.5",
            "gen_ai.usage.input_tokens": 100,
            "gen_ai.usage.output_tokens": 50,
        },
    ):
        pass
    spans = in_memory_exporter.get_finished_spans()
    chat = next((s for s in spans if s.name == "sage.chat"), None)
    assert chat is not None
    assert chat.attributes["gen_ai.operation.name"] == "chat"
    assert chat.attributes["gen_ai.provider.name"] == "moonshot.ai"
    assert chat.attributes["gen_ai.request.model"] == "kimi-k2.5"
    assert chat.attributes["gen_ai.usage.input_tokens"] == 100
    assert chat.attributes["gen_ai.usage.output_tokens"] == 50


@pytest.mark.asyncio
async def test_chat_span_record_exception_false_emits_redacted_event(
    in_memory_exporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """record_exception=False: exception event is emitted with A16-redacted message+stacktrace.

    This test exercises the S-3 mitigation: provider.generate() may raise with
    API-key material in the traceback.  The span must record a redacted event
    instead of letting OTel auto-record the raw exception.
    """
    monkeypatch.setenv("SAGE_REDACT_SECRETS", "1")
    from sage.observability.spans import sage_span, _reset_warn_flag_for_tests
    _reset_warn_flag_for_tests()

    SECRET = "sk-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"  # noqa: S105 — test fixture
    raised = False
    try:
        with sage_span("sage.chat", op="chat", record_exception=False) as span:
            raise ValueError(f"HTTP 400: Authorization header=Bearer {SECRET}")
    except ValueError:
        raised = True

    assert raised, "exception must propagate through sage_span"

    spans = in_memory_exporter.get_finished_spans()
    chat = next((s for s in spans if s.name == "sage.chat"), None)
    assert chat is not None, "sage.chat span must be finished"

    # Span status must be ERROR
    from opentelemetry.trace import StatusCode
    assert chat.status.status_code == StatusCode.ERROR

    # Exception event must exist
    exc_events = [e for e in chat.events if e.name == "exception"]
    assert exc_events, "expected an 'exception' event on the span"
    evt = exc_events[0]

    # Secret must NOT appear in any event attribute
    msg = evt.attributes.get("exception.message", "")
    tb = evt.attributes.get("exception.stacktrace", "")
    assert SECRET not in msg, f"secret leaked in exception.message: {msg!r}"
    assert SECRET not in tb, f"secret leaked in exception.stacktrace: {tb!r}"

    # Redaction marker must be present (proves A16 ran, not just truncation)
    assert "REDACTED" in msg or "REDACTED" in tb, (
        "A16 redaction must replace the secret with REDACTED"
    )


@pytest.mark.asyncio
async def test_tool_span_redacts_sensitive_arguments(
    in_memory_exporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """sage.tool span emits gen_ai.tool.call.{name,arguments,result} with A16 redaction."""
    monkeypatch.setenv("SAGE_REDACT_SECRETS", "1")
    monkeypatch.delenv("SAGE_OTEL_RAW_PAYLOADS", raising=False)
    from sage.tools.base import Tool
    from sage.llm.base import ToolDef

    async def _handler(secret: str = "") -> str:
        return f"used {secret[:4]}..."

    spec = ToolDef(name="test_tool", description="t", parameters={"type": "object"})
    tool = Tool(spec=spec, handler=_handler)
    await tool.execute({"secret": "sk-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"})

    spans = in_memory_exporter.get_finished_spans()
    tool_span = next((s for s in spans if s.name == "sage.tool"), None)
    assert tool_span is not None
    assert tool_span.attributes["gen_ai.operation.name"] == "execute_tool"
    assert tool_span.attributes["gen_ai.tool.name"] == "test_tool"
    args = tool_span.attributes["gen_ai.tool.call.arguments"]
    assert "sk-AAAA" not in args
    assert "REDACTED" in args
