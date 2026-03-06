"""OpenAI-compatible LLM provider for multi-provider routing.

Supports any API that follows the OpenAI chat completions format:
OpenAI, xAI (Grok), DeepSeek, MiniMax, Kimi/Moonshot.
"""
from __future__ import annotations

import logging
from typing import Any

from sage.llm.base import LLMConfig, LLMResponse, Message, Role

log = logging.getLogger(__name__)


class OpenAICompatProvider:
    """Provider for any OpenAI-compatible API (OpenAI, xAI, DeepSeek, MiniMax, Kimi).

    Parameters
    ----------
    api_key:
        Bearer token for the API.
    base_url:
        API base URL (e.g. ``https://api.x.ai/v1``).  Defaults to OpenAI.
    model_id:
        Default model ID to use if none is specified in the config.
    """

    name = "openai-compat"

    def __init__(self, api_key: str, base_url: str | None = None, model_id: str = ""):
        self.api_key = api_key
        self.base_url = base_url
        self.model_id = model_id

    async def generate(
        self,
        messages: list[Message],
        tools: list | None = None,
        config: LLMConfig | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate content via OpenAI-compatible chat completions API."""
        from openai import AsyncOpenAI

        model = self.model_id
        if config and config.model:
            model = config.model

        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        client = AsyncOpenAI(**client_kwargs)

        # Convert messages to OpenAI format
        oai_messages: list[dict[str, str]] = []
        for msg in messages:
            role = msg.role.value
            if role == "tool":
                role = "user"  # Simplify for compat APIs that lack tool role
            oai_messages.append({"role": role, "content": msg.content})

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=oai_messages,  # type: ignore[arg-type]
                max_tokens=config.max_tokens if config and config.max_tokens else 4096,
                temperature=config.temperature if config else 0.3,
            )

            content = response.choices[0].message.content or ""
            return LLMResponse(content=content, model=model)

        except Exception as e:
            log.error("OpenAI-compat API error (%s): %s", self.base_url, e)
            raise
