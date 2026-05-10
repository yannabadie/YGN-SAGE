"""OpenAI model routing helpers shared by active and legacy providers."""
from __future__ import annotations

import re

_OPENAI_RESPONSES_PREFIXES = ("openai/responses/", "responses/")
_OPENAI_RESPONSES_ROUTE_RE = re.compile(
    r"^gpt-5(?:\.\d+)*-pro(?:-|$)",
    re.IGNORECASE,
)


def normalize_openai_model_id(model_id: str) -> str:
    """Strip SAGE-only OpenAI route aliases before handing IDs to SDKs."""
    mid = (model_id or "").strip()
    lower = mid.lower()
    for prefix in _OPENAI_RESPONSES_PREFIXES:
        if lower.startswith(prefix):
            return mid[len(prefix):]
    return mid


def route_openai_model_via_responses(model_id: str) -> bool:
    """Return whether SAGE routes this OpenAI model through Responses.

    This is a local SAGE policy/regression guard, not a live endpoint
    availability claim. It keeps GPT-5 pro variants and explicit
    Responses aliases off the chat-model constructor path.
    """
    mid = (model_id or "").strip()
    lower = mid.lower()
    if any(lower.startswith(prefix) for prefix in _OPENAI_RESPONSES_PREFIXES):
        return True
    return _OPENAI_RESPONSES_ROUTE_RE.match(normalize_openai_model_id(mid)) is not None
