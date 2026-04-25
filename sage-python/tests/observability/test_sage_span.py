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
