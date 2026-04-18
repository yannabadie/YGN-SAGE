"""OpenAI-compatible LLM provider for multi-provider routing.

DEPRECATED: Use ``sage.providers.litellm_provider.LiteLLMProvider`` instead.
This module is kept for backward compatibility and will be removed in v0.2.0.

OpenAI-compatible API support (OpenAI, xAI (Grok), DeepSeek, MiniMax, Kimi/Moonshot)
is now unified via LiteLLM (v1.83+).

Legacy migration:
    Old:  OpenAICompatProvider(api_key="...", base_url="...", model_id="...", provider_name="...")
    New:  LiteLLMProvider(model_string="deepseek/deepseek-chat", api_key="...")

See: sage-python/src/sage/providers/litellm_provider.py
"""
from __future__ import annotations

import json
import logging
import re
import warnings
from typing import Any

from sage.llm.base import LLMConfig, LLMResponse, Message

log = logging.getLogger(__name__)

# Module-level deprecation warning
warnings.warn(
    "sage.providers.openai_compat.OpenAICompatProvider is DEPRECATED. "
    "Use sage.providers.litellm_provider.LiteLLMProvider instead. "
    "This module will be removed in v0.2.0.",
    DeprecationWarning,
    stacklevel=2,
)


def supports_chat_completions_model(provider_name: str, model_id: str) -> bool:
    """Return whether this provider/model pair works with chat completions.

    This provider implementation only speaks the chat-completions API.
    OpenAI GPT-5 Pro variants currently require the Responses API in practice,
    so they must be filtered out until a Responses-backed provider exists.
    """
    if (provider_name or "").lower() != "openai":
        return True
    model = (model_id or "").lower()
    return re.match(r"^gpt-5(?:\.\d+)?-pro(?:-|$)", model) is None


class OpenAICompatProvider:
    """Provider for any OpenAI-compatible API (OpenAI, xAI, DeepSeek, MiniMax, Kimi).

    DEPRECATED: Use ``LiteLLMProvider`` instead (sage.providers.litellm_provider).
    This class is kept for backward compatibility and will be removed in v0.2.0.

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
        API base URL (e.g. ``https://api.x.ai/v1``). Defaults to OpenAI.
    model_id:
        Default model ID to use if none is specified in the config.
    provider_name:
        Explicit provider name for quirk dispatch (e.g. ``"deepseek"``).
        Auto-inferred from *base_url* when empty.
    """

    name = "openai-compat"

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model_id: str = "",
        provider_name: str = "",
    ):
        warnings.warn(
            "OpenAICompatProvider is DEPRECATED. Use LiteLLMProvider instead. "
            "See: sage.providers.litellm_provider.LiteLLMProvider",
            DeprecationWarning,
            stacklevel=2,
        )
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
        if "minimax" in url:
            return "minimax"
        if "moonshot" in url:
            return "kimi"
        if "bigmodel.cn" in url:
            return "glm"
        if "openai.com" in url:
            return "openai"
        return ""

    def capabilities(self) -> dict[str, bool]:
        """Declare what this provider actually supports."""
        return {
            "structured_output": False,
            "tool_role": self.provider_name == "openai",
            "file_search": False,
            "grounding": False,
            "system_prompt": True,
            "streaming": False,
        }

    def _apply_quirks(self, params: dict[str, Any]) -> dict[str, Any]:
        """Apply provider-specific parameter quirks before API call."""
        model = params.get("model", self.model_id).lower()

        # OpenAI GPT-5+ / o-series reasoning models:
        #   - require max_completion_tokens instead of max_tokens
        #   - reject any temperature != 1 (observed 2026-04-18 on every
        #     OpenAI-fallback call during the Meta-Harness v2 real eval)
        # Mirror the litellm_provider.py clamp here because this compat
        # adapter is still used as the topology-runner fallback provider
        # and was the source of 41 temperature-rejection errors that made
        # 4/5 SWE-Lite tasks look worse than they were.
        if self.provider_name == "openai" and any(
            tag in model for tag in ("gpt-5", "o1", "o3", "o4")
        ):
            if "max_tokens" in params:
                params["max_completion_tokens"] = params.pop("max_tokens")
            # The API rejects anything except 1.0 here; just drop our value.
            if "temperature" in params and params["temperature"] != 1.0:
                params["temperature"] = 1.0

        if self.provider_name == "deepseek":
            if "reasoner" in model and "temperature" in params:
                del params["temperature"]
        elif self.provider_name == "kimi":
            # K2.5 has a fixed temperature; sending custom values returns 400.
            if "k2.5" in model or "k2-5" in model:
                params.pop("temperature", None)
            elif "temperature" in params:
                params["temperature"] = min(params["temperature"], 1.0)
        elif self.provider_name == "qwen":
            if "qwen3" in model or "qwq" in model:
                params.setdefault("extra_body", {})
                params["extra_body"]["enable_thinking"] = True

        return params

    def _extract_reasoning(self, message: Any) -> tuple[str, str]:
        """Extract reasoning content and main content from response."""
        content = message.content or ""
        raw = getattr(message, "reasoning_content", None)
        reasoning = raw if isinstance(raw, str) else ""
        return reasoning, content

    def _format_response(self, reasoning: str, content: str) -> str:
        """Merge reasoning into content with <think> tags if present."""
        if reasoning:
            return f"<think>{reasoning}</think>\n{content}"
        return content

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert Message objects to OpenAI dict format."""
        oai_messages: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.role.value

            if self.provider_name == "openai" and role == "assistant" and msg.tool_calls:
                oai_messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content or None,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )
                continue

            if role == "tool":
                if self.provider_name == "openai" and msg.tool_call_id:
                    oai_messages.append(
                        {
                            "role": "tool",
                            "content": msg.content,
                            "tool_call_id": msg.tool_call_id,
                        }
                    )
                    continue

                log.warning(
                    "Rewriting tool role to user for OpenAI-compat API (%s) - "
                    "semantic context (tool provenance) is lost",
                    self.provider_name or "unknown",
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
            import httpx
            from sage.llm._ssl import ssl_verify

            client_kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            client_kwargs["http_client"] = httpx.AsyncClient(verify=ssl_verify(), timeout=60)
            self._client = AsyncOpenAI(**client_kwargs)

        client = self._client
        oai_messages = self._convert_messages(messages)

        params: dict[str, Any] = {
            "model": model,
            "messages": oai_messages,
            "max_tokens": config.max_tokens if config and config.max_tokens else 4096,
            "temperature": config.temperature if config else 0.3,
        }

        # Only OpenAI supports response_format json_schema in this adapter family.
        if config and config.json_schema is not None and self.provider_name == "openai":
            schema = config.json_schema
            if isinstance(schema, type) and hasattr(schema, "model_json_schema"):
                schema = schema.model_json_schema()
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema, "strict": True},
            }

        if tools:
            params["tools"] = tools

        params = self._apply_quirks(params)

        try:
            response = await client.chat.completions.create(**params)  # type: ignore[arg-type]
            msg = response.choices[0].message
            reasoning, content = self._extract_reasoning(msg)
            final_content = self._format_response(reasoning, content)

            usage = None
            if response.usage:
                usage = {
                    "input_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                    "output_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(response.usage, "total_tokens", 0) or 0,
                }

            tool_calls = []
            if msg.tool_calls:
                from sage.llm.base import ToolCall

                for tc in msg.tool_calls:
                    args = tc.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (ValueError, TypeError):
                            args = {"raw": args}
                    tool_calls.append(
                        ToolCall(
                            id=tc.id or f"call_{len(tool_calls)}",
                            name=tc.function.name,
                            arguments=args,
                        )
                    )

            return LLMResponse(
                content=final_content,
                model=model,
                usage=usage,
                tool_calls=tool_calls,
            )
        except Exception as e:
            log.error("OpenAI-compat API error (%s/%s): %s", self.provider_name, self.base_url, e)
            raise
