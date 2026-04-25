"""roadmap-B1: sage_span context manager + payload-safety helpers.

Independent of AgentEvent emission — both can fire at the same call
site without coupling. See spec §3.2 + §4.
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator

from sage.observability import _get_tracer, _init_tracer
from sage.security.redaction import RedactionFilter

log = logging.getLogger(__name__)

_REDACTOR = RedactionFilter()  # honors SAGE_REDACT_SECRETS env

_OTEL_PROVIDER_NAME_MAP: dict[str, str] = {
    "google": "gcp.gemini",
    "openai": "openai",
    "deepseek": "deepseek",
    "xai": "x_ai",
    "kimi": "moonshot.ai",
    "minimax": "minimax.ai",
    "openrouter": "openrouter.ai",
}

_WARNED_SECRETS_DISABLED = False


def otel_provider_name(sage_provider_id: str) -> str:
    """Map SAGE provider id → OTel `gen_ai.provider.name`. Unknown → input verbatim."""
    return _OTEL_PROVIDER_NAME_MAP.get(sage_provider_id.lower(), sage_provider_id)


def _safe_str(value: Any, max_bytes: int = 4096) -> str:
    """Redact + truncate before emitting to a span attribute.

    Reads SAGE_OTEL_RAW_PAYLOADS at call time (env can be flipped in
    tests via monkeypatch). When raw, skip both passes. When redacting,
    delegate to A16 RedactionFilter which traverses lists/tuples/dicts
    recursively. Truncate to max_bytes UTF-8.

    Note on the truncation suffix: the slice is `encoded[: max_bytes - 16]`
    leaving 16 bytes of head room for the `…[truncated]` suffix (14 UTF-8
    bytes). When the truncated text has no spaces, `rsplit(" ", 1)[0]`
    is a no-op, so the final length is at most max_bytes - 2 — still
    under the cap. This is intentional and contract-preserving.
    """
    raw_payloads = os.environ.get("SAGE_OTEL_RAW_PAYLOADS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if raw_payloads:
        s = value if isinstance(value, str) else str(value)
    elif isinstance(value, str):
        s = _REDACTOR.redact_text(value) if _REDACTOR.enabled else value
    else:
        # Lists, tuples, dicts, and any other JSON-serializable payload
        # all flow through redact_value() which traverses recursively.
        # Required to prevent secret leakage on `gen_ai.input.messages`
        # (list of dicts), `gen_ai.tool.call.arguments` (dict), and
        # `gen_ai.tool.call.result` (could be either).
        redacted = _REDACTOR.redact_value(value) if _REDACTOR.enabled else value
        try:
            s = json.dumps(redacted, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            s = str(redacted)
    encoded = s.encode("utf-8")
    if len(encoded) > max_bytes:
        truncated = encoded[: max_bytes - 16].decode("utf-8", errors="ignore")
        return truncated.rsplit(" ", 1)[0] + "…[truncated]"
    return s


def _maybe_warn_secrets_disabled() -> None:
    """Once-per-process WARN if OTel is on but A16 redaction is disabled."""
    global _WARNED_SECRETS_DISABLED
    if _WARNED_SECRETS_DISABLED:
        return
    if not _REDACTOR.enabled:
        log.warning(
            "OTel spans active but secret redaction disabled "
            "(SAGE_REDACT_SECRETS=0) — payloads on spans may contain secrets"
        )
        _WARNED_SECRETS_DISABLED = True


def _reset_warn_flag_for_tests() -> None:
    """Test-only: reset the once-per-process WARN guard."""
    global _WARNED_SECRETS_DISABLED
    _WARNED_SECRETS_DISABLED = False


def _otel_enabled() -> bool:
    """True iff a TracerProvider is configured (any non-`none` exporter)."""
    _init_tracer()
    return _get_tracer() is not None


@contextmanager
def sage_span(
    name: str,
    op: str,
    *,
    record_exception: bool = True,
    **attrs: Any,
) -> Iterator[Any]:
    """Emit an OTel span if a tracer is configured; no-op otherwise.

    `op` populates `gen_ai.operation.name`. Other kwargs are attached
    verbatim — caller is responsible for using `_safe_str` on any
    payload-bearing values before passing them in.

    When ``record_exception=False`` (recommended for LLM provider call sites),
    OTel's auto-recording of exception stacktraces is suppressed and replaced
    with a manually-emitted redacted exception event.  This prevents A16-bypass:
    HTTP 400 tracebacks from providers can carry API-key material in headers
    that auto-record would emit verbatim to the span exporter.
    """
    if not _otel_enabled():
        yield None
        return
    _maybe_warn_secrets_disabled()
    tracer = _get_tracer()
    with tracer.start_as_current_span(
        name,
        record_exception=record_exception,
        set_status_on_exception=record_exception,
    ) as span:
        span.set_attribute("gen_ai.operation.name", op)
        for k, v in attrs.items():
            if v is not None:
                span.set_attribute(k, v)
        try:
            yield span
        except BaseException as exc:
            if not record_exception:
                import traceback as _traceback

                from opentelemetry.trace import Status, StatusCode

                msg = _REDACTOR.redact_text(str(exc)) if _REDACTOR.enabled else str(exc)
                tb = (
                    _REDACTOR.redact_text(_traceback.format_exc())
                    if _REDACTOR.enabled
                    else _traceback.format_exc()
                )
                span.add_event(
                    "exception",
                    {
                        "exception.type": type(exc).__name__,
                        "exception.message": msg,
                        "exception.stacktrace": tb,
                    },
                )
                span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            raise
