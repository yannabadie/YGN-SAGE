"""Regression test for the message-truncation orphan-tool bug.

Discovered 2026-04-17 PM during the F7 SWE-bench smoke: MiniMax-m2.7
rejected requests with `400 bad_request_error: tool result's tool id
not found (2013)` because phases/act.py:282-283 naively kept the
first 2 messages plus the last (MAX_MESSAGES - 2), dropping any
assistant tool_calls in the middle but keeping their tool_result tail.
Lenient providers (OpenAI, Gemini) tolerated the orphans; MiniMax
enforced strict pairing and 400-ed.

The fix builds a set of surviving assistant tool_call ids and drops
any tool message whose tool_call_id has no counterpart.
"""
from __future__ import annotations

from typing import Any

import pytest

from sage.llm.base import Message, Role


def _mk_assistant_with_tool_call(call_id: str, fn_name: str = "execute_bash"):
    """Build an assistant message carrying one tool_call. Mirrors what
    LiteLLM returns and what phases/act.py:252-258 appends to messages.
    """
    class _TC:
        def __init__(self, id: str, name: str):
            self.id = id
            self.name = name
            self.arguments: dict[str, Any] = {}
    return Message(
        role=Role.ASSISTANT,
        content="",
        tool_calls=[_TC(call_id, fn_name)],
    )


def _mk_tool_result(call_id: str, result: str = "ok"):
    return Message(
        role=Role.TOOL,
        content=result,
        tool_call_id=call_id,
        name="execute_bash",
    )


from sage.phases.act import _truncate_messages_orphan_safe as _apply_truncation


def test_orphan_tool_dropped_when_assistant_truncated():
    """The exact scenario from the SWE-bench smoke: assistant→tool pair at
    position N gets split — assistant lands in the dropped middle, tool
    result survives in the tail. Strict providers reject. Truncation must
    drop the orphan tool result.
    """
    sys_msg = Message(role=Role.SYSTEM, content="you are an agent")
    user_msg = Message(role=Role.USER, content="fix this bug")

    # Build 50 messages: system + user + 24 (assistant→tool pairs) =
    # 2 + 48 = 50 total. With MAX_MESSAGES=10, head=2 + tail=8 means we
    # keep system + user + the LAST 8 messages. The assistant→tool pair
    # boundary cuts in the middle of pair #21 → orphan tool at position 0
    # of the tail.
    messages: list[Message] = [sys_msg, user_msg]
    for i in range(24):
        messages.append(_mk_assistant_with_tool_call(f"call_{i}"))
        messages.append(_mk_tool_result(f"call_{i}"))

    # MAX_MESSAGES=10 → keep system+user+8 last = pairs #20..#23 (4 full
    # pairs) AND the tool result of pair #19 (orphan).
    truncated = _apply_truncation(messages, max_msgs=10)

    # Verify: every tool message in `truncated` has a preceding assistant
    # in the same window with that tool_call_id.
    surviving_ids = set()
    for m in truncated:
        if m.role == Role.ASSISTANT and m.tool_calls:
            for tc in m.tool_calls:
                surviving_ids.add(tc.id)
    for m in truncated:
        if m.role == Role.TOOL and m.tool_call_id:
            assert m.tool_call_id in surviving_ids, (
                f"orphan tool_call_id `{m.tool_call_id}` survived truncation — "
                f"strict providers (MiniMax) will 400 on this request"
            )


def test_truncation_no_op_when_below_threshold():
    """Sanity: when len(messages) <= max_msgs, return unchanged."""
    msgs = [
        Message(role=Role.SYSTEM, content="sys"),
        Message(role=Role.USER, content="user"),
        _mk_assistant_with_tool_call("call_x"),
        _mk_tool_result("call_x"),
    ]
    assert _apply_truncation(msgs, max_msgs=10) == msgs


def test_truncation_keeps_paired_tools():
    """When the assistant survives in the tail, its tool result must too —
    we don't drop matched tools, only orphans.
    """
    sys_msg = Message(role=Role.SYSTEM, content="sys")
    user_msg = Message(role=Role.USER, content="user")
    msgs = [sys_msg, user_msg]
    for i in range(8):
        msgs.append(_mk_assistant_with_tool_call(f"call_{i}"))
        msgs.append(_mk_tool_result(f"call_{i}"))
    # MAX=10 → keep first 2 + last 8 = system, user, then pairs #4..#7.
    truncated = _apply_truncation(msgs, max_msgs=10)
    assert len(truncated) == 10
    # Pairs 4-7 stayed intact
    assert truncated[2].role == Role.ASSISTANT
    assert truncated[3].role == Role.TOOL
    assert truncated[3].tool_call_id == truncated[2].tool_calls[0].id


def test_truncation_handles_multi_tool_assistant():
    """An assistant message with N tool_calls produces N tool results.
    All N results must either survive (with their assistant) or be
    dropped together.
    """
    class _TC:
        def __init__(self, id: str, name: str):
            self.id = id
            self.name = name
            self.arguments: dict[str, Any] = {}

    sys_msg = Message(role=Role.SYSTEM, content="sys")
    user_msg = Message(role=Role.USER, content="user")
    multi_assistant = Message(
        role=Role.ASSISTANT,
        content="",
        tool_calls=[_TC("call_a", "fn"), _TC("call_b", "fn"), _TC("call_c", "fn")],
    )
    msgs = [
        sys_msg,
        user_msg,
        multi_assistant,
        _mk_tool_result("call_a"),
        _mk_tool_result("call_b"),
        _mk_tool_result("call_c"),
    ]
    # 6 total, no truncation
    truncated = _apply_truncation(msgs, max_msgs=10)
    assert len(truncated) == 6
    # Now force truncation that drops the multi-assistant: max=4
    # head=2 (sys+user), tail=2 (last 2: tool b + tool c) → both orphan
    truncated2 = _apply_truncation(msgs, max_msgs=4)
    surviving_ids = set()
    for m in truncated2:
        if m.role == Role.ASSISTANT and m.tool_calls:
            for tc in m.tool_calls:
                surviving_ids.add(tc.id)
    for m in truncated2:
        if m.role == Role.TOOL and m.tool_call_id:
            assert m.tool_call_id in surviving_ids, (
                f"multi-tool orphan `{m.tool_call_id}` survived"
            )
