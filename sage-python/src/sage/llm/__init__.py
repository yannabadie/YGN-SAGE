"""LLM abstraction layer with multi-provider support."""
from sage.llm.base import LLMConfig, Message, Role, ToolDef, ToolCall, LLMResponse

__all__ = [
    "LLMConfig",
    "Message",
    "Role",
    "ToolDef",
    "ToolCall",
    "LLMResponse",
]
