"""roadmap-A2 regression test: preserve Kimi `reasoning_content`.

Verifies the A8 Phase 2 fix (commit `df150a2a`) by building a 4-turn
tool-call conversation with `thinking` set on assistant turns and
asserting the Pydantic AI translation emits a `ThinkingPart` BEFORE
any `ToolCallPart` on each assistant `ModelResponse`.

This is the exact shape Moonshot/Kimi requires for multi-turn tool
calls under thinking mode. The pre-fix behaviour dropped the
`Message.thinking` field when rebuilding history, leading to
HTTP 400 "thinking is enabled but reasoning_content is missing in
assistant tool call message at index N".

No live API calls — pure translation-layer test.
"""
from __future__ import annotations

import pytest

from sage.llm.base import Message, Role, ToolCall
from sage.providers.pydantic_ai_provider import _our_messages_to_pydantic


def _build_4turn_toolcall_history_with_thinking() -> list[Message]:
    """A realistic multi-turn tool-call history that reproduced the bug.

    Indexes:
      0: SYSTEM
      1: USER
      2: ASSISTANT (thinking + tool_call)     ← needs reasoning_content
      3: TOOL (return)
      4: ASSISTANT (thinking + tool_call)     ← needs reasoning_content
      5: TOOL (return)
      6: ASSISTANT (thinking + final text)    ← needs reasoning_content
    """
    return [
        Message(role=Role.SYSTEM, content="You are a repair-focused agent."),
        Message(role=Role.USER, content="Fix the bug in src/foo.py."),
        Message(
            role=Role.ASSISTANT,
            content="",
            thinking="I should first read the file to understand context.",
            tool_calls=[
                ToolCall(id="t0", name="read_file", arguments={"path": "src/foo.py"}),
            ],
        ),
        Message(
            role=Role.TOOL,
            content="def foo():\n    return 1\n",
            name="read_file",
            tool_call_id="t0",
        ),
        Message(
            role=Role.ASSISTANT,
            content="",
            thinking="The bug is on line 2. I'll search for callers first.",
            tool_calls=[
                ToolCall(id="t1", name="search_repo", arguments={"query": "foo()"}),
            ],
        ),
        Message(
            role=Role.TOOL,
            content="src/bar.py:10: result = foo()",
            name="search_repo",
            tool_call_id="t1",
        ),
        Message(
            role=Role.ASSISTANT,
            content="The fix is to return 2 instead of 1.",
            thinking="Now I have enough context to propose the fix.",
        ),
    ]


def _assistant_response_parts(pydantic_history: list) -> list[list]:
    """Extract the parts[] of every ModelResponse from the translated history."""
    from pydantic_ai.messages import ModelResponse
    return [list(mr.parts) for mr in pydantic_history if isinstance(mr, ModelResponse)]


def test_thinking_precedes_toolcalls_on_every_assistant_turn() -> None:
    """A8 Phase 2 invariant: ThinkingPart[0] < ToolCallPart[...] on every turn.

    Moonshot's API serializer walks the parts in order. If ThinkingPart
    is missing or comes AFTER ToolCallPart, `reasoning_content` is
    dropped, and Kimi rejects the request with HTTP 400.
    """
    pytest.importorskip("pydantic_ai")
    try:
        from pydantic_ai.messages import ThinkingPart, ToolCallPart
    except ImportError:
        pytest.skip("pydantic_ai ThinkingPart unavailable in this version")

    history = _our_messages_to_pydantic(_build_4turn_toolcall_history_with_thinking())
    assistant_turns = _assistant_response_parts(history)

    assert len(assistant_turns) == 3, (
        f"Expected 3 ModelResponse (assistant turns); got {len(assistant_turns)}"
    )

    for idx, parts in enumerate(assistant_turns):
        part_types = [type(p).__name__ for p in parts]
        # Find first ThinkingPart and first ToolCallPart positions
        first_thinking = next(
            (i for i, p in enumerate(parts) if isinstance(p, ThinkingPart)),
            None,
        )
        first_tool_call = next(
            (i for i, p in enumerate(parts) if isinstance(p, ToolCallPart)),
            None,
        )

        assert first_thinking is not None, (
            f"turn #{idx}: no ThinkingPart found (A8 Phase 2 regression); "
            f"parts={part_types}"
        )
        if first_tool_call is not None:
            assert first_thinking < first_tool_call, (
                f"turn #{idx}: ThinkingPart must precede ToolCallPart "
                f"(Moonshot reasoning_content ordering requirement); "
                f"parts={part_types}"
            )


def test_thinking_preserved_for_assistant_final_text_turn() -> None:
    """The 3rd assistant turn has text (not tool_call). Thinking must still be preserved.

    Even though Kimi only enforces reasoning_content on tool-call turns,
    dropping it on text turns can confuse downstream provider code and
    makes the per-turn serialization inconsistent.
    """
    pytest.importorskip("pydantic_ai")
    try:
        from pydantic_ai.messages import ThinkingPart, TextPart
    except ImportError:
        pytest.skip("pydantic_ai ThinkingPart unavailable in this version")

    history = _our_messages_to_pydantic(_build_4turn_toolcall_history_with_thinking())
    assistant_turns = _assistant_response_parts(history)

    text_turn_parts = assistant_turns[-1]  # final assistant turn
    part_types = [type(p).__name__ for p in text_turn_parts]

    assert any(isinstance(p, ThinkingPart) for p in text_turn_parts), (
        f"final assistant turn: ThinkingPart missing; parts={part_types}"
    )
    assert any(isinstance(p, TextPart) for p in text_turn_parts), (
        f"final assistant turn: TextPart missing; parts={part_types}"
    )


def test_no_thinking_means_no_thinkingpart() -> None:
    """Non-thinking models: no Message.thinking → no ThinkingPart emitted.

    Keeps the fix back-compatible: only models that actually set
    reasoning_content in their responses pay the ThinkingPart cost.
    """
    pytest.importorskip("pydantic_ai")
    try:
        from pydantic_ai.messages import ThinkingPart, ToolCallPart
    except ImportError:
        pytest.skip("pydantic_ai ThinkingPart unavailable in this version")

    messages = [
        Message(role=Role.SYSTEM, content="s"),
        Message(role=Role.USER, content="u"),
        Message(
            role=Role.ASSISTANT,
            content="",
            # NO thinking= field
            tool_calls=[ToolCall(id="t0", name="x", arguments={})],
        ),
    ]
    history = _our_messages_to_pydantic(messages)
    assistant_turns = _assistant_response_parts(history)
    assert len(assistant_turns) == 1
    parts = assistant_turns[0]
    assert not any(isinstance(p, ThinkingPart) for p in parts)
    assert any(isinstance(p, ToolCallPart) for p in parts)


# -------- A8 Phase 3: native PydanticAI profile for Moonshot --------


def test_moonshot_native_profile_declares_thinking_field() -> None:
    """A8 Phase 3: Moonshot model uses a custom OpenAIModelProfile that
    declares `reasoning_content` as the native thinking field.

    Without this, PydanticAI's default moonshotai_model_profile returns
    a plain ModelProfile with supports_thinking=False and no OpenAI-chat
    thinking field — leaving our translation layer to work around the
    multi-turn tool-call HTTP 400 manually. The profile-driven path is
    native, so PydanticAI knows to both EXTRACT reasoning_content from
    Kimi responses AND SEND IT BACK on outgoing assistant messages.
    """
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.profiles.openai import OpenAIModelProfile

    from sage.providers.pydantic_ai_provider import _build_pydantic_model

    model = _build_pydantic_model("kimi", "kimi-k2.5", "fake-key")
    assert isinstance(model.profile, OpenAIModelProfile), (
        f"Moonshot model should use custom OpenAIModelProfile; "
        f"got {type(model.profile).__name__}"
    )
    assert model.profile.supports_thinking is True
    assert model.profile.openai_chat_thinking_field == "reasoning_content", (
        "Moonshot profile must declare 'reasoning_content' as the thinking "
        "field — PydanticAI uses this to extract Kimi's thinking output "
        "into ThinkingPart on every response."
    )
    assert model.profile.openai_chat_send_back_thinking_parts == "field", (
        "Moonshot profile must set send_back_thinking_parts='field' — "
        "PydanticAI uses this to serialize ThinkingPart back as "
        "reasoning_content on outgoing assistant messages (required for "
        "multi-turn tool calls)."
    )


def test_other_providers_not_affected_by_moonshot_profile() -> None:
    """Sanity: the Moonshot-specific profile override doesn't leak to
    other providers (OpenAI, DeepSeek, OpenRouter, etc.)."""
    pytest.importorskip("pydantic_ai")
    from sage.providers.pydantic_ai_provider import _build_pydantic_model

    # DeepSeek uses its own provider — should NOT carry Moonshot profile
    ds_model = _build_pydantic_model("deepseek", "deepseek-chat", "fake-key")
    # profile should either be the default or a DeepSeek-specific one,
    # not the Moonshot OpenAIModelProfile with reasoning_content
    profile = ds_model.profile
    # If it happens to be OpenAIModelProfile (some providers build one),
    # the thinking field should NOT be our Moonshot-specific override
    thinking_field = getattr(profile, "openai_chat_thinking_field", None)
    assert thinking_field != "reasoning_content" or profile.__class__.__module__.startswith(
        "pydantic_ai"
    ), (
        "DeepSeek carries Moonshot-specific thinking_field override — "
        "the fix should scope to the moonshot branch only"
    )
