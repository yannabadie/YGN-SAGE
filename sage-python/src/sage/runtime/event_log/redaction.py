"""RuntimeEventLog redaction primitives.

Reuses sage.security.redaction.RedactionFilter with forced enabled=True for
disk logs. Does not reuse sage.observability.spans._safe_str: that helper is
OTel-specific, reads SAGE_OTEL_RAW_PAYLOADS, and truncates payloads.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from sage.security.redaction import RedactionFilter

_REDACTOR = RedactionFilter(enabled=True)


def _redact_text(value: str) -> str:
    """Strip credential-shaped patterns from arbitrary text."""
    return _REDACTOR.redact_text(value)


def _redact_payload(value: Any) -> Any:
    """Recursively redact text-shaped values inside common containers."""
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, dict):
        return {k: _redact_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_payload(v) for v in value]
    return value


def _hash_payload(event_type: str, payload: Any) -> str:
    """Canonical full SHA-256 over {schema_version, event_type, payload}."""
    from sage.runtime.event_log.schema import SCHEMA_VERSION

    envelope = {
        "schema_version": SCHEMA_VERSION,
        "event_type": event_type,
        "payload": _redact_payload(payload),
    }
    canonical = json.dumps(
        envelope,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _hash_text(value: str) -> str:
    """Full SHA-256 of a credential-redacted string. Used for task_hash."""
    redacted = _redact_text(value)
    return hashlib.sha256(redacted.encode("utf-8", errors="replace")).hexdigest()
