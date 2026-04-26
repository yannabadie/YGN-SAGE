"""Regex-based secret redaction utilities."""
from __future__ import annotations

import os
import re
from dataclasses import is_dataclass, replace
from typing import Any

REDACTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "openai_api_key": re.compile(
        r"\b(?:sk-(?:proj|live)-[A-Za-z0-9_-]{20,}|"
        r"sk-[A-Za-z0-9]{32,})\b"
    ),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "gcp_service_account": re.compile(
        r'\{[^{}]*"type"\s*:\s*"service_account"[^{}]*\}'
        r'|"type"\s*:\s*"service_account"'
        r'|"client_email"\s*:\s*"[^"]+@[^"]+\.iam\.gserviceaccount\.com"'
        r'|"private_key"\s*:\s*"-----BEGIN PRIVATE KEY-----.*?-----END PRIVATE KEY-----\\n?"',
        re.DOTALL,
    ),
    "bearer_token": re.compile(
        r"\bBearer [A-Za-z0-9\-._~+/]{20,}={0,2}(?=\s|$|[\"'`),;])"
    ),
    "jwt": re.compile(
        r"\beyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]+\."
        r"[A-Za-z0-9_-]+(?=\s|$|[\"'`),;])"
    ),
}


def redact(text: str) -> str:
    """Replace known secret-shaped substrings with typed placeholders."""
    redacted = text
    for secret_type, pattern in REDACTION_PATTERNS.items():
        redacted = pattern.sub(f"[REDACTED:{secret_type}]", redacted)
    return redacted


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return redact_dict(value)
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in value)
    return value


def redact_dict(d: dict) -> dict:
    """Return a copy of *d* with secret-shaped strings redacted recursively."""
    return {key: _redact_value(value) for key, value in d.items()}


class RedactionFilter:
    """Environment-controlled redaction helper.

    Redaction is enabled by default. Set ``SAGE_REDACT_SECRETS=0`` to bypass
    scans in hot write paths.
    """

    def __init__(self, enabled: bool | None = None) -> None:
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        if self._enabled is not None:
            return self._enabled
        return os.getenv("SAGE_REDACT_SECRETS", "1") != "0"

    def redact_text(self, text: str) -> str:
        if not self.enabled:
            return text
        return redact(text)

    def redact(self, text: str) -> str:
        return self.redact_text(text)

    def redact_value(self, value: Any) -> Any:
        if not self.enabled:
            return value
        return _redact_value(value)

    def redact_dict(self, d: dict) -> dict:
        if not self.enabled:
            return d
        return redact_dict(d)

    def redact_event(self, event: Any) -> Any:
        if not self.enabled:
            return event

        updates: dict[str, dict] = {}
        for attr in ("data", "meta"):
            value = getattr(event, attr, None)
            if isinstance(value, dict):
                redacted = redact_dict(value)
                if redacted != value:
                    updates[attr] = redacted
        if not updates:
            return event
        if is_dataclass(event) and not isinstance(event, type):
            return replace(event, **updates)
        for attr, value in updates.items():
            setattr(event, attr, value)
        return event
