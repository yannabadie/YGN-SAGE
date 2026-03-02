"""Base types for the LLM abstraction layer."""
from __future__ import annotations

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


@dataclass
class LLMConfig:
    provider: str
    model: str
    max_tokens: int = 8192
    temperature: float = 0.0
    top_p: float = 1.0
    api_key: str | None = None
    base_url: str | None = None
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
    ) -> LLMResponse: ...
