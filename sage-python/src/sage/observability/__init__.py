"""roadmap-B1: OpenTelemetry GenAI observability for YGN-SAGE.

Lazy, env-gated tracer setup. Default off — `SAGE_OTEL_EXPORTER=none`
returns no tracer and `sage_span` becomes a no-op. Other exporters
(`console`, `otlp_http`, `logfire`) wire a TracerProvider at first
`_init_tracer()` call (idempotent).
"""
from __future__ import annotations

import importlib.metadata
import logging
import os

log = logging.getLogger(__name__)

_INITIALIZED = False
_TRACER = None


def _init_tracer() -> None:
    """Idempotent. Reads SAGE_OTEL_EXPORTER and configures a tracer."""
    global _INITIALIZED, _TRACER
    if _INITIALIZED:
        return
    _INITIALIZED = True

    exporter_kind = os.environ.get("SAGE_OTEL_EXPORTER", "none").strip().lower()
    if exporter_kind == "none":
        return  # _TRACER stays None; sage_span yields None

    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider

    try:
        version = importlib.metadata.version("ygn-sage")
    except importlib.metadata.PackageNotFoundError:
        version = "0.0.0+dev"

    resource = Resource.create({"service.name": "ygn-sage", "service.version": version})
    provider = TracerProvider(resource=resource)

    if exporter_kind == "console":
        from opentelemetry.sdk.trace.export import (
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif exporter_kind == "otlp_http":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    elif exporter_kind == "logfire":
        import logfire
        logfire.configure(service_name="ygn-sage")
        # logfire installs its own TracerProvider and bridges to OTel.
        # Resolve a tracer but do NOT call set_tracer_provider — that
        # would clobber logfire's setup.
        _TRACER = trace.get_tracer("sage", version)
        return
    else:
        log.warning(
            "Unknown SAGE_OTEL_EXPORTER=%r; no exporter active", exporter_kind
        )
        return

    trace.set_tracer_provider(provider)
    _TRACER = trace.get_tracer("sage", version)


def _get_tracer():
    """Return the configured tracer, or None if no exporter is active."""
    return _TRACER


def _reset_for_tests() -> None:
    """Test-only: reset module state so importlib.reload() re-inits cleanly."""
    global _INITIALIZED, _TRACER
    _INITIALIZED = False
    _TRACER = None
