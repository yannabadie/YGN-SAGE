"""OpenAI GPT provider."""
from __future__ import annotations

import json
import os
from sage.llm.base import LLMConfig, LLMResponse, Message, ToolCall, ToolDef


class OpenAIProvider:
    """LLM provider for OpenAI GPT models."""

    name = "openai"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        config: LLMConfig | None = None,
    ) -> LLMResponse:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("Install openai: pip install 'ygn-sage[openai]'")

        client = AsyncOpenAI(api_key=self.api_key)

        model = config.model if config else "gpt-4o"
        max_tokens = config.max_tokens if config else 8192

        api_messages = [{"role": m.role.value, "content": m.content} for m in messages]

        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": api_messages,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        return LLMResponse(
            content=choice.message.content or "",
            tool_calls=tool_calls,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            } if response.usage else None,
            model=response.model,
            stop_reason=choice.finish_reason,
        )
