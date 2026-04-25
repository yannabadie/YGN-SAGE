"""B1 acceptance: replay the 2026-04-23 A2 cascade against an in-memory exporter.

Goal: confirm that the span hierarchy makes the cascade
(Stage 4 multi-agent → Kimi 400 → fallback empty) visible in one
trace view. This is the headline justification for B1.
"""
from __future__ import annotations

import importlib

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
async def test_a2_cascade_visible_in_one_trace(in_memory_exporter) -> None:
    """Synthetic replay of the 2026-04-23 cascade.

    Top-level pipeline.run wraps:
      - sage.execute (the 5th stage)
        - sage.node.worker_0 (Stage 4 multi-agent)
          - sage.chat (kimi-k2.5) → ERROR HTTP 400
        - sage.node.fallback (Stage 4 fallback single-agent)
          - sage.chat (gemini-3.1-flash-lite-preview) → empty content

    After replay, the exporter must contain a parent-child span tree
    where the kimi chat carries error.type and the fallback chat
    carries empty content signal.
    """
    from sage.observability.spans import otel_provider_name, sage_span

    with sage_span("sage.pipeline.run", op="invoke_agent"):
        with sage_span("sage.execute", op="sage.execute"):
            with sage_span("sage.node.worker_0", op="invoke_agent",
                           **{"sage.node.name": "worker_0"}):
                with sage_span(
                    "sage.chat",
                    op="chat",
                    **{
                        "gen_ai.provider.name": otel_provider_name("kimi"),
                        "gen_ai.request.model": "kimi-k2.5",
                        "error.type": "HTTPError",
                        "http.response.status_code": 400,
                    },
                ):
                    pass
            with sage_span("sage.node.fallback", op="invoke_agent",
                           **{"sage.node.name": "fallback"}):
                with sage_span(
                    "sage.chat",
                    op="chat",
                    **{
                        "gen_ai.provider.name": otel_provider_name("google"),
                        "gen_ai.request.model": "gemini-3.1-flash-lite-preview",
                        "gen_ai.response.finish_reasons": ["empty_content"],
                    },
                ):
                    pass

    spans = in_memory_exporter.get_finished_spans()
    by_name = {s.name: s for s in spans}

    # All required spans present
    for required in [
        "sage.pipeline.run",
        "sage.execute",
        "sage.node.worker_0",
        "sage.node.fallback",
    ]:
        assert required in by_name, f"missing span {required!r}"

    # Two chat spans, one with error.type set
    chat_spans = [s for s in spans if s.name == "sage.chat"]
    assert len(chat_spans) == 2
    kimi = next(c for c in chat_spans if c.attributes.get("gen_ai.provider.name") == "moonshot.ai")
    assert kimi.attributes.get("error.type") == "HTTPError"
    assert kimi.attributes.get("http.response.status_code") == 400

    fallback = next(c for c in chat_spans if c.attributes.get("gen_ai.provider.name") == "gcp.gemini")
    assert "empty_content" in fallback.attributes.get("gen_ai.response.finish_reasons")

    # Parent-child verification: the kimi chat is descendant of pipeline.run
    pipeline_run = by_name["sage.pipeline.run"]
    assert kimi.parent is not None
    # Walk up via spans by span_id
    by_span_id = {s.context.span_id: s for s in spans}
    cur = kimi
    seen = set()
    while cur.parent is not None and cur.parent.span_id not in seen:
        seen.add(cur.parent.span_id)
        parent = by_span_id.get(cur.parent.span_id)
        if parent is None:
            break
        cur = parent
    assert cur.name == "sage.pipeline.run", (
        f"expected to reach sage.pipeline.run as ancestor of kimi chat; got {cur.name}"
    )
