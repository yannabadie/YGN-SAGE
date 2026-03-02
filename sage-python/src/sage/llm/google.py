"""Google Gemini provider."""
from __future__ import annotations

import os
from sage.llm.base import LLMConfig, LLMResponse, Message, ToolCall, ToolDef


class GoogleProvider:
    """LLM provider for Google Gemini models."""

    name = "google"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        config: LLMConfig | None = None,
    ) -> LLMResponse:
        try:
            from google import genai
        except ImportError:
            raise ImportError("Install google-genai: pip install 'ygn-sage[google]'")

        client = genai.Client(api_key=self.api_key)

        model = config.model if config else "gemini-2.0-flash"

        # Convert messages to Gemini format
        contents = []
        system_instruction = None
        for msg in messages:
            if msg.role.value == "system":
                system_instruction = msg.content
            else:
                role = "model" if msg.role.value == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": msg.content}]})

        generate_config = {}
        if config:
            generate_config["max_output_tokens"] = config.max_tokens
            generate_config["temperature"] = config.temperature

        kwargs: dict = {
            "model": model,
            "contents": contents,
            "config": generate_config,
        }
        if system_instruction:
            kwargs["config"]["system_instruction"] = system_instruction

        response = await client.aio.models.generate_content(**kwargs)

        return LLMResponse(
            content=response.text or "",
            tool_calls=[],
            model=model,
        )
