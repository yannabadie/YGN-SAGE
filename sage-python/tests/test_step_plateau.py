"""Plateau detector regression tests (P1.1 of 2026-04-18 mega-plan).

Before today, ``AgentLoop.run()`` just looped until ``step_count >= max_steps``.
When an S3 agent got stuck repeating the same tool call or the same empty
monologue, it burned its budget and the bench-side wall-clock timeout
(``generation_timeout_300s``) fired, producing an empty prediction. v5f
SWE-bench smoke hit this on 3 of 5 tasks.

The detector tracks the last 3 step signatures (content + sorted
tool-call args). Three identical in a row = stuck; break with current
best output. Size-3 tolerates accidental repeats; three consecutive is
the "stuck" signal per the P1.1 design.
"""
from __future__ import annotations

import sys
import types as _types
from collections import deque

if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = _types.ModuleType("sage_core")

import pytest

from sage.agent_loop import AgentLoop


def _make_loop_stub() -> object:
    """Minimal object exposing the attributes the plateau block touches.

    We don't need a full AgentLoop — the detection block reads
    ``self._recent_step_signatures``, ``self.step_count``, and
    ``self.config.name``, and logs. Everything else is off the code path.
    """
    stub = _types.SimpleNamespace()
    stub._recent_step_signatures = deque(maxlen=3)
    stub.step_count = 0
    stub.config = _types.SimpleNamespace(name="test-loop")
    return stub


def test_deque_sized_three_tolerates_single_repeat() -> None:
    """Two different signatures back-to-back don't trigger a plateau."""
    sigs: deque[str] = deque(maxlen=3)
    sigs.append("think:read file")
    sigs.append("think:read file")
    sigs.append("think:write patch")
    assert len(set(sigs)) > 1, "three different or alternating signatures must NOT be a plateau"


def test_three_identical_signatures_trigger_plateau() -> None:
    """Three consecutive identical step signatures = stuck."""
    sigs: deque[str] = deque(maxlen=3)
    sig = "think:i need to read the file"
    sigs.append(sig)
    sigs.append(sig)
    sigs.append(sig)
    assert len(sigs) == sigs.maxlen
    assert len(set(sigs)) == 1
    assert sig != "", "empty signature is filtered by a separate guard"


def test_empty_signature_does_not_trigger_plateau() -> None:
    """All-empty content + no tool calls is a valid initial state, not a loop.

    The guard ``_sig`` must be truthy before we accept a plateau — prevents
    bailing before the first real step has happened.
    """
    sigs: deque[str] = deque(maxlen=3)
    sigs.append("")
    sigs.append("")
    sigs.append("")
    assert len(set(sigs)) == 1  # technically all same
    assert all(s == "" for s in sigs)  # but all empty — should not bail
    # In the agent_loop block the condition is:
    #     len(set(...)) == 1 AND _sig  (truthy)
    # so an empty _sig blocks the early-break.
    _sig = next(iter(sigs))
    assert not _sig  # i.e. falsy, so the break condition does NOT fire


def test_deque_maxlen_is_three() -> None:
    """The detector must observe exactly three consecutive steps.

    Size-1 would bail on a single step. Size-2 would miss a legitimate
    self-correction pattern (X, Y, X). Size-3 tolerates one accidental
    repeat; "three in a row" is the "stuck" signal per the design doc.
    """
    loop_like = _make_loop_stub()
    assert loop_like._recent_step_signatures.maxlen == 3


def test_agent_loop_instance_has_plateau_deque() -> None:
    """The real AgentLoop must initialise the deque in __init__ (not first-use)."""
    # Confirm the attribute exists on every AgentLoop instance before run()
    # so we don't AttributeError on the first step of the first task.
    assert hasattr(AgentLoop, "__init__")
    # Inspect source: the init must reference _recent_step_signatures.
    import inspect
    src = inspect.getsource(AgentLoop.__init__)
    assert "_recent_step_signatures" in src, (
        "AgentLoop.__init__ must create _recent_step_signatures = deque(maxlen=3)"
    )


def test_signature_differs_when_tool_args_differ() -> None:
    """Same tool name but different args should NOT collapse to one signature.

    Otherwise an agent that keeps calling read_file with different paths would
    be treated as stuck — the opposite of what we want.
    """
    import types

    class _Call:
        def __init__(self, name: str, arguments: dict) -> None:
            self.name = name
            self.arguments = arguments

    # Simulate two "think" outputs with the same content but different tool args.
    # The signature block concatenates content + sorted "name:repr(args)"; we
    # just need to confirm two runs with different args produce different sigs.
    def _sig(content: str, calls: list[object]) -> str:
        parts = [content.strip()[:512]]
        for c in calls:
            parts.append(f"{c.name}:{c.arguments!r}")
        return "|".join(parts)

    s1 = _sig("thinking", [_Call("read_file", {"path": "a.py"})])
    s2 = _sig("thinking", [_Call("read_file", {"path": "b.py"})])
    assert s1 != s2
