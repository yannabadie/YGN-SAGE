"""Shared OTel state reset between observability tests.

Without this, `set_tracer_provider` calls inside one test leak to
later tests in the same pytest session — visible as "Overriding of
current TracerProvider is not allowed" warnings and unexpected
console-spans output. See B1 Task 1 code-review notes.
"""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def _reset_otel_globals(monkeypatch: pytest.MonkeyPatch):
    """Reset SAGE OTel module state and the OTel global provider per test."""
    yield
    # Tear-down: clear sage.observability module-level globals
    import sage.observability as obs
    obs._reset_for_tests()
    # Also clear the spans.py warn flag if loaded
    try:
        from sage.observability import spans as _spans
        _spans._reset_warn_flag_for_tests()
    except ImportError:
        pass
    # Reset OTel global provider so the next test's `set_tracer_provider`
    # doesn't trigger the "Overriding ... not allowed" path.
    # _TRACER_PROVIDER holds the current provider; _TRACER_PROVIDER_SET_ONCE
    # is a sync.Once-style guard (opentelemetry.util._once.Once) whose
    # _done flag must also be cleared, or the next set_tracer_provider call
    # silently no-ops. Both are private but stable across OTel SDK 1.x.
    from opentelemetry import trace
    if hasattr(trace, "_TRACER_PROVIDER"):
        # noinspection PyProtectedMember
        trace._TRACER_PROVIDER = None
    if hasattr(trace, "_TRACER_PROVIDER_SET_ONCE"):
        # noinspection PyProtectedMember
        trace._TRACER_PROVIDER_SET_ONCE._done = False
