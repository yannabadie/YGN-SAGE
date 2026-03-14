"""OpenAI-compatible LLM provider for multi-provider routing.

Supports any API that follows the OpenAI chat completions format:
OpenAI, xAI (Grok), DeepSeek, MiniMax, Kimi/Moonshot.
"""
from __future__ import annotations

import logging
from typing import Any

from sage.llm.base import LLMConfig, LLMResponse, Message

log = logging.getLogger(__name__)


class OpenAICompatProvider:
    """Provider for any OpenAI-compatible API (OpenAI, xAI, DeepSeek, MiniMax, Kimi).

    Handles provider-specific quirks:
    - DeepSeek: strip temperature for reasoner; merge reasoning_content
    - Grok (xAI): merge reasoning_content into <think> tags
    - Kimi: clamp temperature to [0, 1]
    - MiniMax: <think> tags already in content body (preserve as-is)

    Parameters
    ----------
    api_key:
        Bearer token for the API.
    base_url:
        API base URL (e.g. ``https://api.x.ai/v1``).  Defaults to OpenAI.
    model_id:
        Default model ID to use if none is specified in the config.
    provider_name:
        Explicit provider name for quirk dispatch (e.g. ``"deepseek"``).
        Auto-inferred from *base_url* when empty.
    """

    name = "openai-compat"

    def __init__(self, api_key: str, base_url: str | None = None,
                 model_id: str = "", provider_name: str = ""):
        self.api_key = api_key
        self.base_url = base_url
        self.model_id = model_id
        self.provider_name = provider_name or self._infer_provider(base_url)
        self._client: Any = None

    @staticmethod
    def _infer_provider(base_url: str | None) -> str:
        """Infer provider name from base_url for quirk dispatch."""
        if not base_url:
            return "openai"
        url = base_url.lower()
        if "deepseek" in url:
            return "deepseek"
        if "x.ai" in url:
            return "xai"
        if "minimaxi" in url:
            return "minimax"
        if "moonshot" in url:
            return "kimi"
        if "openai.com" in url:
            return "openai"
        return ""

    def capabilities(self) -> dict[str, bool]:
        """Declare what this provider actually supports."""
        return {
            "structured_output": False,
            "tool_role": False,      # Rewritten to user role
            "file_search": False,    # Silently dropped
            "grounding": False,
            "system_prompt": True,
            "streaming": False,
        }

    def _apply_quirks(self, params: dict[str, Any]) -> dict[str, Any]:
        """Apply provider-specific parameter quirks before API call."""
        model = params.get("model", self.model_id).lower()

        if self.provider_name == "deepseek":
            if "reasoner" in model and "temperature" in params:
                del params["temperature"]
        elif self.provider_name == "kimi":
            if "temperature" in params:
                params["temperature"] = min(params["temperature"], 1.0)

        return params

    def _extract_reasoning(self, message: Any) -> tuple[str, str]:
        """Extract reasoning content and main content from response.

        Returns (reasoning, content) tuple.
        """
        content = message.content or ""
        raw = getattr(message, "reasoning_content", None)
        reasoning = raw if isinstance(raw, str) else ""
        return reasoning, content

    def _format_response(self, reasoning: str, content: str) -> str:
        """Merge reasoning into content with <think> tags if present."""
        if reasoning:
            return f"<think>{reasoning}</think>\n{content}"
        return content

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, str]]:
        """Convert Message objects to OpenAI dict format."""
        oai_messages: list[dict[str, str]] = []
        for msg in messages:
            role = msg.role.value
            if role == "tool":
                log.warning(
                    "Rewriting tool role to user for OpenAI-compat API — "
                    "semantic context (tool provenance) is lost"
                )
                role = "user"
            oai_messages.append({"role": role, "content": msg.content})
        return oai_messages

    async def generate(
        self,
        messages: list[Message],
        tools: list | None = None,
        config: LLMConfig | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate content via OpenAI-compatible chat completions API."""
        if kwargs.get("file_search_store_names"):
            log.warning("file_search_store_names not supported by OpenAI-compat provider, ignored")
        from openai import AsyncOpenAI

        model = self.model_id
        if config and config.model:
            model = config.model

        if self._client is None:
            client_kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._client = AsyncOpenAI(**client_kwargs)

        client = self._client

        oai_messages = self._convert_messages(messages)

        params: dict[str, Any] = {
            "model": model,
            "messages": oai_messages,
            "max_tokens": config.max_tokens if config and config.max_tokens else 4096,
            "temperature": config.temperature if config else 0.3,
        }

        # Constrained decoding: JSON schema output (OpenAI Structured Outputs)
        if config and config.json_schema is not None:
            schema = config.json_schema
            # If it's a Pydantic model class, extract its JSON schema
            if isinstance(schema, type) and hasattr(schema, "model_json_schema"):
                schema = schema.model_json_schema()
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema, "strict": True},
            }

        params = self._apply_quirks(params)

        try:
            response = await client.chat.completions.create(**params)  # type: ignore[arg-type]
            msg = response.choices[0].message
            reasoning, content = self._extract_reasoning(msg)
            final_content = self._format_response(reasoning, content)
            return LLMResponse(content=final_content, model=model)
        except Exception as e:
            log.error("OpenAI-compat API error (%s/%s): %s", self.provider_name, self.base_url, e)
            raise
