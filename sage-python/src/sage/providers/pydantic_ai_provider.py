"""Pydantic AI-based LLM provider — replacement for LiteLLMProvider.

Phase 2 of the 2026-04-18 migration (see
`docs/plans/2026-04-18-pydantic-ai-migration.md`). Implements the same
`sage.llm.base.LLMProvider` protocol as `LiteLLMProvider` so the
migration is a single factory swap in `boot_providers.py` +
`boot_pipeline.py`.

Design choices:

* Native Pydantic AI provider classes where available (OpenAI, Google,
  xAI, MoonshotAI, OpenRouter). OpenAI-compatible custom `base_url` for
  DeepSeek and MiniMax — Pydantic AI's `OpenAIProvider(base_url=...)`
  pattern is first-class, not a hack.
* NO hand-coded parameter quirks. Pydantic AI's per-provider model
  classes (XaiModel, GoogleModel, MoonshotAIProvider, etc.) maintain
  their own up-to-date quirk tables — we trust the library.
* Cost telemetry: compute from `RequestUsage.input_tokens /
  output_tokens` × cost-per-token from **our `cards.toml`**, which we
  maintain. Both LiteLLM's registry and genai-prices lag behind reality
  (e.g. genai-prices rejects gpt-5.4 with LookupError 2026-04-18).
  cards.toml is our source of truth.
* Message translation: our `Message` dataclass → Pydantic AI
  `ModelRequest` / `ModelResponse`. Supports assistant tool_calls and
  tool-result returns for multi-turn.
* Tools passed schema-level via `ModelRequestParameters.function_tools`
  — our `ToolDef` maps 1:1 onto `ToolDefinition`.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from sage.llm.base import (
    LLMConfig,
    LLMResponse,
    Message,
    Role,
    ToolCall,
    ToolDef,
)

log = logging.getLogger(__name__)

# Provider → Pydantic AI model/provider class mapping. The `model_class`
# + `provider_class` are looked up lazily to avoid paying the import
# cost of every Pydantic AI sub-module when only one provider is used.
_PROVIDER_MAP: dict[str, dict[str, Any]] = {
    "openai": {"kind": "native_openai"},
    "google": {"kind": "native_google"},
    "xai": {"kind": "native_xai"},
    "kimi": {"kind": "moonshot"},
    "openrouter": {"kind": "native_openrouter"},
    # Custom OpenAI-compat endpoints
    "deepseek": {"kind": "custom_openai", "base_url": "https://api.deepseek.com/v1"},
    "minimax": {"kind": "custom_openai", "base_url": "https://api.minimax.io/v1"},
}


def _build_pydantic_model(provider_name: str, model_id: str, api_key: str | None) -> Any:
    """Return a Pydantic AI Model instance for the given (provider, model)."""
    cfg = _PROVIDER_MAP.get(provider_name.lower())
    if cfg is None:
        raise ValueError(
            f"Unknown SAGE provider {provider_name!r}; expected one of "
            f"{sorted(_PROVIDER_MAP)}. Add it to _PROVIDER_MAP if you added "
            f"a new provider in cards.toml."
        )
    kind = cfg["kind"]

    if kind == "native_openai":
        from pydantic_ai.providers.openai import OpenAIProvider
        # gpt-5.4-pro and other reasoning-only variants must use the
        # Responses API (`/v1/responses`) — the Chat Completions endpoint
        # returns "This is not a chat model" 404. Pydantic AI exposes
        # OpenAIResponsesModel for this. Our cards.toml convention: any
        # model starting with "gpt-5.4-pro" or ending in "-pro" among the
        # gpt-5 family routes via Responses. (The previous LiteLLM path
        # used an `openai/responses/<model>` prefix — same mechanism.)
        _responses_only = model_id.startswith("gpt-5.4-pro") or "responses" in model_id.lower()
        if _responses_only:
            from pydantic_ai.models.openai import OpenAIResponsesModel
            return OpenAIResponsesModel(
                model_id, provider=OpenAIProvider(api_key=api_key or "")
            )
        from pydantic_ai.models.openai import OpenAIChatModel
        return OpenAIChatModel(model_id, provider=OpenAIProvider(api_key=api_key or ""))

    if kind == "native_google":
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider
        return GoogleModel(
            model_id,
            provider=GoogleProvider(api_key=api_key or "", vertexai=False),
        )

    if kind == "native_xai":
        from pydantic_ai.models.xai import XaiModel
        from pydantic_ai.providers.xai import XaiProvider
        return XaiModel(model_id, provider=XaiProvider(api_key=api_key or ""))

    if kind == "moonshot":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.moonshotai import MoonshotAIProvider
        return OpenAIChatModel(
            model_id, provider=MoonshotAIProvider(api_key=api_key or "")
        )

    if kind == "native_openrouter":
        from pydantic_ai.models.openrouter import OpenRouterModel
        from pydantic_ai.providers.openrouter import OpenRouterProvider
        return OpenRouterModel(
            model_id, provider=OpenRouterProvider(api_key=api_key or "")
        )

    if kind == "custom_openai":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
        return OpenAIChatModel(
            model_id,
            provider=OpenAIProvider(base_url=cfg["base_url"], api_key=api_key or ""),
        )

    raise ValueError(f"Unhandled provider kind: {kind!r}")


_REGISTRY_CACHE: Any = None


def _get_registry() -> Any:
    """Lazy-load the Rust ModelRegistry from cards.toml.

    Searches the same paths as sage.providers.registry._toml_search_paths.
    Caches on first successful load.
    """
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is not None:
        return _REGISTRY_CACHE
    from pathlib import Path

    try:
        import sage_core  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return None

    candidates = [
        Path.cwd() / "config" / "cards.toml",
        Path(__file__).parent.parent.parent.parent / "config" / "cards.toml",
        Path(__file__).parent.parent.parent.parent.parent / "sage-core" / "config" / "cards.toml",
        Path.home() / ".sage" / "cards.toml",
    ]
    for p in candidates:
        if p.is_file():
            try:
                _REGISTRY_CACHE = sage_core.ModelRegistry.from_toml_file(str(p))
                return _REGISTRY_CACHE
            except Exception:  # noqa: BLE001
                continue
    return None


def _lookup_cost_per_token(provider_name: str, model_id: str) -> tuple[float, float]:
    """Return (input_cost_per_token, output_cost_per_token) from cards.toml.

    cards.toml is our source of truth; both LiteLLM's registry and
    genai-prices lag behind (e.g. neither has correct gpt-5.4 entries as
    of 2026-04-18). Returns (0.0, 0.0) if the card can't be found so
    callers know the cost figure is degraded, not silently wrong.

    cards.toml fields: cost_input_per_m / cost_output_per_m (USD per 1M
    tokens). Divide by 1e6 to get per-token.
    """
    registry = _get_registry()
    if registry is None:
        return (0.0, 0.0)
    try:
        card = registry.get(model_id)
        if card is None:
            return (0.0, 0.0)
        in_cost = float(getattr(card, "cost_input_per_m", 0.0) or 0.0) / 1_000_000
        out_cost = float(getattr(card, "cost_output_per_m", 0.0) or 0.0) / 1_000_000
        return (in_cost, out_cost)
    except Exception:  # noqa: BLE001
        return (0.0, 0.0)


def _our_messages_to_pydantic(messages: list[Message]) -> list[Any]:
    """Translate our ``Message`` list to Pydantic AI ModelMessage list.

    System prompts are prepended as ``SystemPromptPart`` in the first
    ModelRequest. Pydantic AI's low-level ``direct.model_request`` does
    not take an ``instructions`` kwarg (unlike the Agent API).
    """
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        SystemPromptPart,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )

    out: list[Any] = []
    # We bucket consecutive user/tool parts into a single ModelRequest,
    # and emit a ModelResponse for every assistant message. This mirrors
    # Pydantic AI's request/response alternation model.
    pending_request_parts: list[Any] = []

    def _flush_request() -> None:
        if pending_request_parts:
            out.append(ModelRequest(parts=list(pending_request_parts)))
            pending_request_parts.clear()

    for msg in messages:
        if msg.role == Role.SYSTEM:
            pending_request_parts.append(SystemPromptPart(content=msg.content))
            continue

        if msg.role == Role.USER:
            pending_request_parts.append(UserPromptPart(content=msg.content))
            continue

        if msg.role == Role.TOOL:
            # Tool return bundled into the next request block
            pending_request_parts.append(
                ToolReturnPart(
                    tool_name=msg.name or "tool",
                    content=msg.content,
                    tool_call_id=msg.tool_call_id or "",
                )
            )
            continue

        if msg.role == Role.ASSISTANT:
            _flush_request()
            response_parts: list[Any] = []
            if msg.content:
                response_parts.append(TextPart(content=msg.content))
            for tc in msg.tool_calls or []:
                response_parts.append(
                    ToolCallPart(
                        tool_name=tc.name,
                        args=tc.arguments,
                        tool_call_id=tc.id,
                    )
                )
            if response_parts:
                out.append(ModelResponse(parts=response_parts))
            continue

        # Unhandled role — best-effort as user prompt
        log.debug("Unhandled role %r in Pydantic AI translation", msg.role)
        pending_request_parts.append(UserPromptPart(content=msg.content))

    _flush_request()
    return out


def _pydantic_response_to_ours(
    response: Any,
    model_name: str,
    provider_name: str,
    model_id: str,
) -> LLMResponse:
    """Translate Pydantic AI ModelResponse → our LLMResponse."""
    content_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for part in getattr(response, "parts", []) or []:
        part_type = type(part).__name__
        if part_type == "TextPart":
            content_parts.append(getattr(part, "content", "") or "")
        elif part_type == "ToolCallPart":
            args = getattr(part, "args", None)
            # Pydantic AI delivers args as either dict or JSON string
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args else {}
                except json.JSONDecodeError:
                    args = {"_raw": args}
            tool_calls.append(
                ToolCall(
                    id=str(getattr(part, "tool_call_id", "") or ""),
                    name=str(getattr(part, "tool_name", "") or ""),
                    arguments=dict(args or {}),
                )
            )
        # ThinkingPart is dropped — our LLMResponse has no thinking field

    usage: dict[str, int | float] = {}
    req_usage = getattr(response, "usage", None)
    if req_usage is not None:
        in_tok = int(getattr(req_usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(req_usage, "output_tokens", 0) or 0)
        usage["input_tokens"] = in_tok
        usage["output_tokens"] = out_tok
        usage["total_tokens"] = in_tok + out_tok
        # Cost from our cards.toml
        in_rate, out_rate = _lookup_cost_per_token(provider_name, model_id)
        if in_rate > 0 or out_rate > 0:
            usage["cost_usd"] = in_tok * in_rate + out_tok * out_rate

    return LLMResponse(
        content="".join(content_parts),
        tool_calls=tool_calls,
        usage=usage or None,  # type: ignore[arg-type]
        model=model_name,
        stop_reason=getattr(response, "finish_reason", None),
    )


def _our_tools_to_pydantic(tools: list[Any]) -> list[Any]:
    """Translate tool defs to Pydantic AI ``ToolDefinition`` list.

    Accepts BOTH our ``ToolDef`` dataclass AND raw OpenAI-format
    function-tool dicts — LiteLLMProvider accepted both shapes and
    some callers still pass the dict variant (observed 2026-04-18 in
    test_e2e_real on phases/think.py). Normalize here.
    """
    from pydantic_ai import ToolDefinition

    out: list[Any] = []
    for t in tools:
        if isinstance(t, dict):
            # OpenAI-format: {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}
            # OR bare: {"name": ..., "description": ..., "parameters": {...}}
            fn = t.get("function") if t.get("type") == "function" else t
            out.append(
                ToolDefinition(
                    name=fn.get("name", ""),
                    description=fn.get("description", ""),
                    parameters_json_schema=fn.get("parameters", {}),
                )
            )
        else:
            # ToolDef dataclass
            out.append(
                ToolDefinition(
                    name=t.name,
                    description=t.description,
                    parameters_json_schema=t.parameters,
                )
            )
    return out


class PydanticAIProvider:
    """LLMProvider implementation backed by Pydantic AI.

    Parameters
    ----------
    provider_name: our SAGE provider name ("openai", "deepseek", ...)
    model_id: the model id as stored in cards.toml ("gpt-5.4", ...)
    api_key: upstream API key
    """

    name = "pydantic_ai"

    def __init__(
        self,
        provider_name: str,
        model_id: str,
        api_key: str | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.model_id = model_id
        self.api_key = api_key
        # Cache of built Pydantic AI models keyed by their model_id. We
        # build lazily-on-first-use for each distinct id the caller asks
        # for via `config.model` — matches LiteLLMProvider's per-call
        # model routing (commit c9ff902) where ModelAssigner picks a
        # different model per topology node.
        self._model_cache: dict[str, Any] = {}
        self._model = self._get_or_build_model(model_id)
        # Set by ProviderPool for FrugalGPT-on-rate-limit runtime circuit
        # breaking. Keep the same attribute name LiteLLMProvider exposed.
        self._pool_ref: Any = None

    def _get_or_build_model(self, model_id: str) -> Any:
        """Return a Pydantic AI model for the given id, cached."""
        mid = model_id or self.model_id
        if mid in self._model_cache:
            return self._model_cache[mid]
        built = _build_pydantic_model(self.provider_name, mid, self.api_key)
        self._model_cache[mid] = built
        return built

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def for_sage_provider(
        cls,
        provider_name: str,
        model_id: str,
        api_key: str | None = None,
    ) -> PydanticAIProvider:
        """Construct a provider for a given (SAGE provider, model id) pair.

        Matches the factory signature of ``LiteLLMProvider.for_sage_provider``
        so migration is a single s/LiteLLMProvider/PydanticAIProvider in
        ``boot_providers.py`` + ``boot_pipeline.py``.
        """
        return cls(provider_name=provider_name, model_id=model_id, api_key=api_key)

    # ------------------------------------------------------------------
    # LLMProvider protocol
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        config: LLMConfig | None = None,
        tool_choice: str | None = None,
    ) -> LLMResponse:
        from pydantic_ai.direct import model_request
        from pydantic_ai.models import ModelRequestParameters

        # Honor per-call model override — topology runners (ModelAssigner
        # commit c9ff902) pick a different model per node and pass it
        # via config.model. Without this the boot-time model is always
        # used, which for openrouter (no default_model in connector.py)
        # built OpenRouterModel("") and crashed on profile access
        # (ValueError: not enough values to unpack). Mirrors LiteLLM-
        # Provider's effective_model logic.
        effective_model_id = (config.model if config and config.model else None) or self.model_id
        model_obj = self._get_or_build_model(effective_model_id)
        resolved_model_name_for_cost = effective_model_id

        history = _our_messages_to_pydantic(messages)

        params = ModelRequestParameters(
            function_tools=_our_tools_to_pydantic(tools) if tools else [],
            allow_text_output=True,
        )

        try:
            response = await model_request(
                model_obj,
                history,
                model_request_parameters=params,
            )
        except Exception as exc:
            # Runtime rate-limit / quota signal → ProviderPool. Matches
            # the LiteLLMProvider behaviour so TopologyRunner's AgentLoop
            # circuit-breaker wiring (P1.2, commit 8cb719e) still fires.
            _exc_str = str(exc).lower()
            _is_rate_limit = "ratelimit" in type(exc).__name__.lower() or "429" in _exc_str
            _is_quota = _is_rate_limit and any(
                s in _exc_str for s in
                ["insufficient_quota", "quota", "billing", "credit",
                 "exceeded your current quota", "payment"]
            )
            if _is_rate_limit and self._pool_ref is not None:
                try:
                    record = getattr(self._pool_ref, "record_failure", None)
                    if callable(record):
                        record(self.provider_name, exc)
                    if _is_quota:
                        mark_dead = getattr(self._pool_ref, "mark_dead", None)
                        if callable(mark_dead):
                            mark_dead(self.provider_name)
                except Exception:  # noqa: BLE001
                    pass  # never let breaker bookkeeping mask the real error
            raise

        model_name = getattr(response, "model_name", None) or resolved_model_name_for_cost
        return _pydantic_response_to_ours(
            response,
            model_name=model_name,
            provider_name=self.provider_name,
            model_id=resolved_model_name_for_cost,
        )

    async def generate_stream(
        self,
        messages: list[Message],
        config: LLMConfig | None = None,
    ):
        """Stream text chunks from the LLM.

        Pydantic AI streams via ``Agent.run_stream`` rather than through
        the low-level ``direct`` API. We wrap one Agent per call to keep
        the streaming surface simple — this is the same Phase-1 model
        the LiteLLM path uses (text-only, no tools).
        """
        from pydantic_ai import Agent

        # Assemble a single user prompt from the message list. Streaming
        # is only used for non-AVR, non-tool paths (see AgentLoop.stream),
        # so collapsing to one prompt is safe.
        history = _our_messages_to_pydantic(messages)
        # The last UserPromptPart is the actual prompt; everything before
        # becomes message_history.
        user_prompt = ""
        if history:
            last = history[-1]
            parts = getattr(last, "parts", [])
            for p in reversed(parts):
                if type(p).__name__ == "UserPromptPart":
                    user_prompt = getattr(p, "content", "") or ""
                    break

        agent = Agent(self._model)
        async with agent.run_stream(
            user_prompt,
            message_history=history[:-1] if len(history) > 1 else None,
        ) as run:
            async for chunk in run.stream_text():
                yield chunk
