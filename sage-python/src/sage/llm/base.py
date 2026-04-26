"""Base types for the LLM abstraction layer."""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    role: Role
    content: str
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    # A8 Phase 2 (2026-04-24): thinking-mode reasoning text. Set on
    # assistant messages when the underlying provider returns a
    # `reasoning_content` / ThinkingPart (kimi-k2.5/k2.6,
    # deepseek-v4-pro, any future thinking-enabled model). The
    # assistant's subsequent turn MUST re-serialize this back to the
    # provider or Moonshot/DeepSeek will reject the request with
    # HTTP 400 "thinking is enabled but reasoning_content is missing
    # in assistant tool call message at index N". Empty string means
    # either the model wasn't a thinking model OR thinking was
    # explicitly disabled; both are benign.
    thinking: str = ""


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] | None = None
    model: str | None = None
    stop_reason: str | None = None
    # A8 Phase 2 (2026-04-24): thinking-mode reasoning text from the
    # provider's response (Moonshot `reasoning_content`, PydanticAI
    # ThinkingPart, DeepSeek v4-pro thinking-mode). Callers that
    # construct an assistant `Message` from this response MUST copy
    # `thinking` onto the Message so it round-trips on the next turn.
    # See Message.thinking for the full contract.
    thinking: str = ""


@dataclass
class LLMConfig:
    provider: str
    model: str
    max_tokens: int = 8192
    context_window: int = 128000  # Input token limit (NOT output limit)
    temperature: float = 0.0
    top_p: float = 1.0
    api_key: str | None = None
    base_url: str | None = None
    json_schema: type | dict | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol that all LLM providers must implement."""

    name: str

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        config: LLMConfig | None = None,
        tool_choice: str | None = None,
    ) -> LLMResponse: ...


@runtime_checkable
class StreamingLLMProvider(LLMProvider, Protocol):
    """Extended protocol for providers that support token-level streaming.

    Providers implement ``generate_stream`` to yield text chunks as they
    arrive from the upstream API.  The base ``generate`` method is still
    required for non-streaming callers.
    """

    def generate_stream(
        self,
        messages: list[Message],
        config: LLMConfig | None = None,
    ) -> AsyncIterator[str]:
        """Yield response text chunks as they arrive from the LLM.

        Tools are intentionally excluded — streaming is for simple
        text generation only (Phase 1).

        The protocol method itself is not declared `async`: implementers
        return an AsyncIterator (either via an async-generator function
        body that uses `async def ... yield`, or by constructing one).
        """
        ...  # pragma: no cover
