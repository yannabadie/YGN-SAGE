"""B1: OTel observability package — lazy boot, no-op when disabled."""
from __future__ import annotations

import importlib

import pytest


def test_no_tracer_when_exporter_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAGE_OTEL_EXPORTER", "none")
    # Re-import to reset module state
    import sage.observability as obs
    importlib.reload(obs)
    obs._init_tracer()
    assert obs._get_tracer() is None


def test_tracer_initialized_when_exporter_console(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAGE_OTEL_EXPORTER", "console")
    import sage.observability as obs
    importlib.reload(obs)
    obs._init_tracer()
    assert obs._get_tracer() is not None


def test_unknown_exporter_logs_warning_and_returns_none(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("SAGE_OTEL_EXPORTER", "carrier-pigeon")
    import sage.observability as obs
    importlib.reload(obs)
    with caplog.at_level("WARNING", logger="sage.observability"):
        obs._init_tracer()
    assert obs._get_tracer() is None
    assert any("carrier-pigeon" in r.message for r in caplog.records)


def test_init_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAGE_OTEL_EXPORTER", "console")
    import sage.observability as obs
    importlib.reload(obs)
    obs._init_tracer()
    first = obs._get_tracer()
    obs._init_tracer()  # second call should be a no-op
    assert obs._get_tracer() is first


def _make_in_memory_provider():
    """Helper: build a TracerProvider with InMemorySpanExporter and return both."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_sage_span_yields_none_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAGE_OTEL_EXPORTER", "none")
    import sage.observability as obs
    importlib.reload(obs)
    from sage.observability.spans import sage_span
    with sage_span("x", op="chat") as span:
        assert span is None


def test_sage_span_emits_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAGE_OTEL_EXPORTER", "console")  # forces init path
    import sage.observability as obs
    importlib.reload(obs)
    # Override the configured provider with an in-memory one for assertion
    from opentelemetry import trace
    provider, exporter = _make_in_memory_provider()
    trace.set_tracer_provider(provider)
    obs._TRACER = trace.get_tracer("sage", "test")

    from sage.observability.spans import sage_span
    with sage_span("test_span", op="chat", **{"gen_ai.request.model": "gpt-5"}) as span:
        assert span is not None

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test_span"
    assert spans[0].attributes["gen_ai.operation.name"] == "chat"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-5"


def test_safe_str_redacts_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SAGE_OTEL_RAW_PAYLOADS", raising=False)
    monkeypatch.setenv("SAGE_REDACT_SECRETS", "1")
    from sage.observability.spans import _safe_str
    leaky = "user prompt with sk-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA inside"
    out = _safe_str(leaky)
    assert "sk-AAAA" not in out
    assert "REDACTED" in out


def test_safe_str_truncates_at_4kib() -> None:
    from sage.observability.spans import _safe_str
    big = "x" * 8000
    out = _safe_str(big, max_bytes=4096)
    assert len(out.encode("utf-8")) <= 4096
    assert out.endswith("…[truncated]")


def test_safe_str_raw_payloads_skips_redaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGE_OTEL_RAW_PAYLOADS", "1")
    from sage.observability.spans import _safe_str
    leaky = "sk-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    out = _safe_str(leaky)
    assert "sk-AAAA" in out


def test_provider_name_map_covers_all_seven() -> None:
    from sage.observability.spans import otel_provider_name
    assert otel_provider_name("google") == "gcp.gemini"
    assert otel_provider_name("openai") == "openai"
    assert otel_provider_name("deepseek") == "deepseek"
    assert otel_provider_name("xai") == "x_ai"
    assert otel_provider_name("kimi") == "moonshot.ai"
    assert otel_provider_name("minimax") == "minimax.ai"
    assert otel_provider_name("openrouter") == "openrouter.ai"


def test_provider_name_map_unknown_returns_input() -> None:
    from sage.observability.spans import otel_provider_name
    assert otel_provider_name("alien-provider") == "alien-provider"


def test_safe_str_redacts_inside_list_of_dicts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Critical regression: lists/tuples must be redacted recursively.

    Pre-fix bug: `gen_ai.input.messages = [{"content": "sk-AAA..."}]`
    fell into the str(value) else-branch and leaked the API key
    verbatim into the span attribute.
    """
    monkeypatch.delenv("SAGE_OTEL_RAW_PAYLOADS", raising=False)
    monkeypatch.setenv("SAGE_REDACT_SECRETS", "1")
    from sage.observability.spans import _safe_str
    leaky = [
        {"role": "user", "content": "sk-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"},
        {"role": "assistant", "content": "ok"},
    ]
    out = _safe_str(leaky)
    assert "sk-AAAA" not in out
    assert "REDACTED" in out


def test_safe_str_redacts_dict_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Coverage gap: the dict branch had no explicit test pre-fix."""
    monkeypatch.delenv("SAGE_OTEL_RAW_PAYLOADS", raising=False)
    monkeypatch.setenv("SAGE_REDACT_SECRETS", "1")
    from sage.observability.spans import _safe_str
    out = _safe_str({"prompt": "sk-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"})
    assert "sk-AAAA" not in out
    assert "REDACTED" in out


def test_safe_str_redacts_tuple_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tuples must follow the same recursive redaction path as lists."""
    monkeypatch.delenv("SAGE_OTEL_RAW_PAYLOADS", raising=False)
    monkeypatch.setenv("SAGE_REDACT_SECRETS", "1")
    from sage.observability.spans import _safe_str
    out = _safe_str(("safe-prefix", "sk-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"))
    assert "sk-AAAA" not in out
