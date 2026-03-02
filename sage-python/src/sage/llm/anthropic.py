"""Anthropic Claude provider."""
from __future__ import annotations

import os
from sage.llm.base import LLMConfig, LLMResponse, Message, ToolCall, ToolDef


class AnthropicProvider:
    """LLM provider for Anthropic Claude models."""

    name = "anthropic"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        config: LLMConfig | None = None,
    ) -> LLMResponse:
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install 'ygn-sage[anthropic]'")

        client = anthropic.AsyncAnthropic(api_key=self.api_key)

        # Convert messages to Anthropic format
        system_msg = ""
        api_messages = []
        for msg in messages:
            if msg.role.value == "system":
                system_msg = msg.content
            else:
                api_messages.append({"role": msg.role.value, "content": msg.content})

        model = config.model if config else "claude-sonnet-4-6"
        max_tokens = config.max_tokens if config else 8192

        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": api_messages,
        }
        if system_msg:
            kwargs["system"] = system_msg
        if tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]

        response = await client.messages.create(**kwargs)

        # Extract content and tool calls
        content_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input)
                )

        return LLMResponse(
            content="\n".join(content_parts),
            tool_calls=tool_calls,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            model=response.model,
            stop_reason=response.stop_reason,
        )
