"""Tests for learn_final's salvage path when result_text is empty.

Replaces the old `"Agent finished at step N"` placeholder — that string
was being packaged as a fake "patch" by SWE-bench (docs/benchmarks/
2026-04-17-swebench-smoke-debug.md). Now we salvage the last real
assistant content or emit a self-describing sentinel.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from sage.phases.learn import learn_final


def _mk_event(etype, content):
    """Mirror the real WorkingMemory event shape: object with attrs, not a tuple."""
    return SimpleNamespace(event_type=etype, content=content)


def _mk_loop(events, step_count=5, result_text=""):
    """Build a minimal loop stub with controllable working_memory events.

    `events` may be a list of tuples (legacy test signature) OR a list of
    already-built SimpleNamespaces. Tuples are auto-promoted.
    """
    normalized = [
        e if hasattr(e, "event_type") else _mk_event(e[0], e[1])
        for e in events
    ]
    wm = SimpleNamespace(_events=normalized, event_count=lambda: len(normalized))
    loop = SimpleNamespace(
        step_count=step_count,
        total_cost_usd=0.0,
        working_memory=wm,
        semantic_memory=None,
        guardrail_pipeline=None,
        consolidator=None,
        _skip_guardrails=True,
        _s2_avr_retries=0,
        _last_avr_iterations=0,
        config=SimpleNamespace(validation_level=1),
        _original_validation_level=1,
        _emit=MagicMock(),
    )
    return loop


@pytest.mark.asyncio
async def test_returns_result_text_when_present():
    """When learn_final gets a non-empty result_text, no salvage needed."""
    loop = _mk_loop(events=[])
    out = await learn_final("task", "a real answer", loop)
    assert out == "a real answer"


@pytest.mark.asyncio
async def test_salvages_last_assistant_event_when_empty():
    """Empty result_text should be replaced by the last ASSISTANT event."""
    events = [
        ("USER", "question"),
        ("ASSISTANT", "first draft"),
        ("TOOL", "grep output"),
        ("ASSISTANT", "second draft\ndetails"),
    ]
    loop = _mk_loop(events=events)
    out = await learn_final("task", "", loop)
    assert out == "second draft\ndetails"


@pytest.mark.asyncio
async def test_skips_empty_or_whitespace_assistant_events():
    """Blank ASSISTANT events shouldn't win over a real earlier one."""
    events = [
        ("ASSISTANT", "useful content"),
        ("ASSISTANT", "   "),
        ("ASSISTANT", ""),
    ]
    loop = _mk_loop(events=events)
    out = await learn_final("task", "", loop)
    assert out == "useful content"


@pytest.mark.asyncio
async def test_explicit_sentinel_when_no_assistant_content():
    """If no assistant content at all, return a self-describing sentinel.

    Important: must NOT return the old "Agent finished at step N" string
    because SWE-bench was packaging that as a fake patch (25 chars) in the
    2026-04-17 smoke runs.
    """
    loop = _mk_loop(events=[("USER", "hi"), ("TOOL", "out")], step_count=7)
    out = await learn_final("task", "", loop)
    assert "Agent finished at step" not in out
    assert "[sage:" in out or "no content" in out or "exited" in out
    assert "7" in out  # step count preserved


@pytest.mark.asyncio
async def test_robust_to_missing_events_attribute():
    """If the working memory is malformed, fallback to the sentinel."""
    wm = SimpleNamespace()  # no _events attr
    loop = SimpleNamespace(
        step_count=3,
        total_cost_usd=0.0,
        working_memory=wm,
        semantic_memory=None,
        guardrail_pipeline=None,
        consolidator=None,
        _skip_guardrails=True,
        _s2_avr_retries=0,
        _last_avr_iterations=0,
        config=SimpleNamespace(validation_level=1),
        _original_validation_level=1,
        _emit=MagicMock(),
    )
    out = await learn_final("task", "", loop)
    assert "3" in out
    assert "[sage:" in out or "no content" in out
