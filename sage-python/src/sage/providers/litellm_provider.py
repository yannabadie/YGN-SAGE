"""LiteLLM-based LLM provider — unified interface for all upstream APIs.

Replaces per-provider custom plumbing with a single adapter that delegates
to ``litellm.acompletion``.  Provider-specific routing (base URL, model
prefix, API key env var) is handled by the ``for_sage_provider`` factory.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import litellm

from sage.llm.base import LLMConfig, LLMResponse, Message, ToolCall, ToolDef

log = logging.getLogger(__name__)

# LiteLLM handles provider-specific parameter quirks natively when
# `drop_params=True` is set — unsupported parameters (e.g. `temperature`
# on GPT-5 reasoning models) are silently filtered instead of raising
# `BadRequestError`. This removes the need for our own per-model drop
# tables. Documented in
# https://github.com/berriai/litellm/blob/main/docs/my-website/docs/completion/drop_params.md
#
# Also gives us automatic handling of:
#   - GPT-5 / o-series `max_tokens → max_completion_tokens` swap
#     (LiteLLM's provider_specific_params translation; verified via
#     Context7 /berriai/litellm).
#   - Moonshot (Kimi) temperature clamp to [0, 1] — only when routed
#     via the `moonshot/` prefix, not our custom "openai/kimi-k2.5".
#   - Gemini 3 default temperature=1.0 injection.
#
# Set ONCE at module import, not per-call — litellm reads the flag on
# every `acompletion`. Per directive #7: we rely on upstream LiteLLM
# for model-quirk behaviour; we only keep quirks for models/providers
# LiteLLM doesn't know about (so far: none we care about).
litellm.drop_params = True

# ---------------------------------------------------------------------------
# Provider → LiteLLM model-string mapping
# ---------------------------------------------------------------------------

_PROVIDER_PREFIX: dict[str, str] = {
    "openai": "openai",
    "deepseek": "deepseek",
    "google": "gemini",
    "xai": "xai",
    "openrouter": "openrouter",
    "minimax": "minimax",
}

# Providers that need a custom base_url and use the generic "openai/" prefix.
_CUSTOM_BASE_PROVIDERS: dict[str, str] = {
    "kimi": "https://api.moonshot.ai/v1",
}

# TTL for provider marked DEAD by runtime rate-limit / quota signal.
# Short for pure rate-limit (they typically recover in ~60s); longer for
# quota exhaustion (typically midnight UTC reset). Reused by TTL refresh.
DEFAULT_POOL_DEAD_TTL_RL_SEC = 300


def _infer_provider_from_model_id(model_id: str) -> str:
    """Best-effort provider inference from a bare model id.

    Used as a last resort when config.provider is missing/"unknown" and the
    adapter default can't help (neither has the right prefix). Patterns match
    actual model_id conventions in sage-core/config/cards.toml (April 2026).
    Empty return means "caller should fall back to adapter default".
    """
    if not model_id:
        return ""
    m = model_id.lower()
    if m.startswith("gemini-"):
        return "google"
    # OpenAI: cards.toml only wires gpt-5.x variants as of 2026-04-18.
    # Keep gpt- prefix broad so gpt-5.4 / gpt-5.2 / gpt-5.4-{pro,mini,nano}
    # all route correctly and any future gpt-5.N rev is covered without
    # a re-edit. Do NOT add o1/o3/o4 — see
    # `docs/patterns/knowledge-cutoff-checks.md`.
    if m.startswith("gpt-"):
        return "openai"
    if m.startswith("deepseek"):
        return "deepseek"
    if m.startswith("grok"):
        return "xai"
    if m.startswith("minimax") or m.startswith("MiniMax".lower()):
        return "minimax"
    if m.startswith("kimi"):
        return "kimi"
    if "/" in model_id:  # openrouter-style "qwen/qwen-plus"
        return "openrouter"
    return ""


def _litellm_model_string(provider: str, model_id: str) -> str:
    """Build the ``model`` argument expected by ``litellm.acompletion``.

    For most providers this is ``"<prefix>/<model_id>"``.  OpenAI GPT-5.4-pro
    models require the Responses-API prefix ``"openai/responses/<model_id>"``.
    """
    provider_lower = provider.lower()

    # Custom-base providers use the plain "openai/" prefix.
    if provider_lower in _CUSTOM_BASE_PROVIDERS:
        return f"openai/{model_id}"

    prefix = _PROVIDER_PREFIX.get(provider_lower, provider_lower)

    # OpenAI gpt-5.4-pro variants need the Responses API path.
    if prefix == "openai" and model_id.startswith("gpt-5.4-pro"):
        return f"openai/responses/{model_id}"

    return f"{prefix}/{model_id}"


class LiteLLMProvider:
    """Unified LLM provider backed by LiteLLM.

    Parameters
    ----------
    model_string:
        Full LiteLLM model identifier (e.g. ``"deepseek/deepseek-chat"``).
    api_key:
        API key passed to the upstream provider.
    api_base:
        Optional custom API base URL (for Kimi, MiniMax, etc.).
    """

    name = "litellm"

    def __init__(
        self,
        model_string: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self.model_string = model_string
        self.api_key = api_key
        self.api_base = api_base
        # Set by ProviderPool after construction to enable
        # runtime rate-limit / quota circuit-breaking. Can remain None;
        # failures then fall through to the caller.
        self._pool_ref: Any = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def for_sage_provider(
        cls,
        provider_name: str,
        model_id: str,
        api_key: str | None = None,
    ) -> LiteLLMProvider:
        """Create a provider instance pre-configured for a SAGE provider name.

        >>> p = LiteLLMProvider.for_sage_provider("deepseek", "deepseek-chat", "sk-...")
        >>> p.model_string
        'deepseek/deepseek-chat'
        """
        model_string = _litellm_model_string(provider_name, model_id)
        api_base = _CUSTOM_BASE_PROVIDERS.get(provider_name.lower())
        return cls(model_string=model_string, api_key=api_key, api_base=api_base)

    # ------------------------------------------------------------------
    # Message conversion (SAGE → OpenAI dict format for LiteLLM)
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_messages(messages: list[Message]) -> list[dict[str, Any]]:
        """Convert SAGE ``Message`` objects to OpenAI-format dicts."""
        out: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.role.value

            # Assistant message carrying tool_calls
            if role == "assistant" and msg.tool_calls:
                out.append(
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

            # Tool-result message
            if role == "tool" and msg.tool_call_id:
                entry: dict[str, Any] = {
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                }
                if msg.name:
                    entry["name"] = msg.name
                out.append(entry)
                continue

            out.append({"role": role, "content": msg.content})

        return out

    @staticmethod
    def _convert_tools(tools: list[ToolDef] | list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert SAGE ``ToolDef`` objects to OpenAI function-tool dicts.

        Handles both ToolDef objects and pre-formatted dicts (from ToolRegistry.get_tool_defs).
        """
        result = []
        for t in tools:
            if isinstance(t, dict):
                # Already in OpenAI format
                result.append(t)
            else:
                # ToolDef object
                result.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                })
        return result

    # ------------------------------------------------------------------
    # Core generate
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        config: LLMConfig | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion via LiteLLM.

        Converts SAGE types to/from the OpenAI wire format that LiteLLM
        expects and returns a populated ``LLMResponse``.

        `tool_choice` (added 2026-04-17): when set to "required" / "auto" /
        "none" or a specific function name, forwarded to the provider's
        OpenAI-compatible API. Used by phases/think.py to FORCE coder/actor
        roles to call execute_bash on early steps, since the F6 _CODER text
        mandate ("AT LEAST 3 execute_bash") was empirically ignored by all
        models (smoke v3: 0 tool calls across 3 tasks despite the mandate).
        Most providers accept "required"; a few may reject — exception
        propagates and caller can drop the gate and retry.
        """
        oai_messages = self._convert_messages(messages)

        # Per-model routing: honor config.model if it names a different model
        # than the adapter default. Without this, ModelAssigner decisions are
        # silently dropped — LiteLLMProvider was always calling self.model_string
        # (the provider default) regardless of what the pipeline assigned.
        # Flagged by Codex review 2026-04-18.
        effective_model = self.model_string
        if config and getattr(config, "model", None):
            _cfg_model = config.model
            # config.model is typically a bare id like "gemini-3.1-flash-lite-preview"
            # or "deepseek-reasoner"; LiteLLM needs "<provider>/<model>". If the
            # config specifies a different model_id, re-prefix via provider.
            #
            # "/" in model_id is ambiguous: could be a LiteLLM prefix
            # ("openai/gpt-4") or part of a multi-slash HF-style id
            # ("qwen/qwen3.5-plus-02-15" — openrouter). Only treat as
            # pre-formatted when the first segment is a known LiteLLM prefix.
            _KNOWN_PREFIXES = set(_PROVIDER_PREFIX.values()) | {"openai", "vertex_ai"}
            _first_seg = _cfg_model.split("/", 1)[0]
            if "/" in _cfg_model and _first_seg in _KNOWN_PREFIXES:
                effective_model = _cfg_model  # already formatted
            else:
                # Prefer explicit provider on config; treat "unknown"/"" as
                # missing (registry defaults to "unknown" when TOML lacks a
                # provider key — don't propagate that to litellm or it emits
                # model="unknown/…" which litellm rejects).
                _cfg_provider_raw = getattr(config, "provider", "") or ""
                _cfg_provider = (
                    _cfg_provider_raw
                    if _cfg_provider_raw and _cfg_provider_raw.lower() != "unknown"
                    else ""
                )
                # Model-id-based inference beats adapter default. If the
                # model id clearly belongs to a specific provider (e.g.
                # "gemini-3.1-flash-lite-preview" → google), we should not
                # route it via the deepseek adapter, even if that's the
                # default we were built with.
                if not _cfg_provider:
                    _cfg_provider = _infer_provider_from_model_id(_cfg_model)
                # Last resort: adapter default. Used only when model_id
                # doesn't match any known pattern (e.g. a custom model name).
                if not _cfg_provider and "/" in self.model_string:
                    _cfg_provider = self.model_string.split("/", 1)[0]
                if _cfg_provider:
                    effective_model = _litellm_model_string(_cfg_provider, _cfg_model)

        # Build the request "naïvely" and let LiteLLM filter unsupported
        # params via the module-level `litellm.drop_params = True`. This
        # covers GPT-5 / o-series reasoning models rejecting `temperature`,
        # the `max_tokens → max_completion_tokens` swap, Moonshot/Kimi's
        # temperature clamp, and Gemini 3's default-to-1.0 policy — all
        # of which are maintained upstream in litellm.llms.<provider>/
        # transformation modules. Per directive #7: stop hand-coding
        # per-model quirks when the library already owns them.
        _requested_temp = config.temperature if config else 0.0
        params: dict[str, Any] = {
            "model": effective_model,
            "messages": oai_messages,
            "max_tokens": config.max_tokens if config else 4096,
            "temperature": _requested_temp,
        }

        if self.api_key:
            params["api_key"] = self.api_key
        if self.api_base:
            params["api_base"] = self.api_base

        if tools:
            params["tools"] = self._convert_tools(tools)
            # Only meaningful when tools are present — providers will 400
            # if tool_choice is set with no tools list.
            if tool_choice:
                params["tool_choice"] = tool_choice

        try:
            response = await litellm.acompletion(**params)
        except Exception as exc:
            # Runtime rate-limit / quota signal → ProviderPool (FrugalGPT-on-
            # rate-limit, Apr 18). Without this, a provider that starts
            # rate-limiting mid-run keeps getting requests until every node
            # times out. We trip the breaker + mark dead so subsequent nodes
            # route elsewhere and the next batch-start refresh re-probes.
            _exc_str = str(exc).lower()
            _exc_name = type(exc).__name__
            _is_rate_limit = ("ratelimit" in _exc_name.lower() or "429" in _exc_str)
            _is_quota = _is_rate_limit and any(
                s in _exc_str for s in [
                    "insufficient_quota", "quota", "billing", "credit",
                    "exceeded your current quota", "payment",
                ]
            )
            if self._pool_ref is not None and (_is_rate_limit or _is_quota):
                # Derive provider name from the effective model prefix
                _prov = (effective_model.split("/", 1)[0]
                         if "/" in effective_model else self.model_string.split("/", 1)[0]
                         if "/" in self.model_string else "")
                # Reverse-map LiteLLM prefix → SAGE provider name (gemini → google)
                _litellm_to_sage = {v: k for k, v in _PROVIDER_PREFIX.items()}
                _prov_sage = _litellm_to_sage.get(_prov, _prov)
                if _prov_sage:
                    try:
                        self._pool_ref.record_failure(_prov_sage, exc)
                        if _is_quota:
                            # Quota exhaustion: long TTL (typically midnight UTC reset)
                            import time as _time
                            self._pool_ref._dead_at[_prov_sage] = _time.time()
                            log.warning(
                                "LiteLLM: quota exhaustion on %s → marked DEAD for %ds TTL",
                                _prov_sage, DEFAULT_POOL_DEAD_TTL_RL_SEC,
                            )
                        else:
                            log.info(
                                "LiteLLM: rate-limit on %s → circuit-breaker failure recorded",
                                _prov_sage,
                            )
                    except Exception as _inner:  # noqa: BLE001 - signal is best-effort
                        log.debug("pool_ref.record_failure failed: %s", _inner)
            log.error("LiteLLM error for model %s", self.model_string, exc_info=True)
            raise

        # --- Parse response ---------------------------------------------------
        choice = response.choices[0]
        msg = choice.message

        content = msg.content or ""

        # Tool calls
        parsed_tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (ValueError, TypeError):
                        args = {"raw": args}
                parsed_tool_calls.append(
                    ToolCall(
                        id=tc.id or f"call_{len(parsed_tool_calls)}",
                        name=tc.function.name,
                        arguments=args,
                    )
                )

        # Usage
        usage: dict[str, int] | None = None
        if response.usage:
            usage = {
                "input_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(response.usage, "total_tokens", 0) or 0,
            }

        # Cost (LiteLLM tracks cost in hidden params)
        cost = None
        hidden = getattr(response, "_hidden_params", None)
        if isinstance(hidden, dict):
            cost = hidden.get("response_cost")
        if cost is not None and usage is not None:
            usage["cost_usd"] = cost

        # Stop reason
        stop_reason = getattr(choice, "finish_reason", None)

        return LLMResponse(
            content=content,
            tool_calls=parsed_tool_calls,
            usage=usage,
            model=self.model_string,
            stop_reason=stop_reason,
        )
