"""SOTA March 2026 Model Router for YGN-SAGE.

Tested models (ChatGPT Pro account via Codex CLI):
  gpt-5.3-codex  — most powerful coding model, supports xhigh effort
  gpt-5.2        — most powerful general model, supports xhigh effort
  gpt-5          — reasoning model, max effort=high

Google Gemini (via API key):
  gemini-3.1-pro-preview       — complex reasoning
  gemini-3-flash-preview       — fast code mutation
  gemini-3.1-flash-lite-preview — lowest latency
  gemini-2.5-flash-lite        — cheapest
  gemini-2.5-flash             — stable fallback
"""
from typing import Literal
from sage.llm.base import LLMConfig

Tier = Literal[
    "fast", "mutator", "reasoner", "codex", "codex_max",
    "budget", "critical", "fallback",
]


class ModelRouter:
    """Routes requests to the optimal model for each task type."""

    MODELS = {
        # OpenAI via Codex CLI (most powerful)
        "codex": "gpt-5.3-codex",       # SOTA coding, xhigh effort
        "codex_max": "gpt-5.2",         # SOTA general reasoning, xhigh effort
        # Google Gemini 3.x (primary API)
        "fast": "gemini-3.1-flash-lite-preview",
        "mutator": "gemini-3-flash-preview",
        "reasoner": "gemini-3.1-pro-preview",
        # Google Gemini 2.5 (stable fallback)
        "budget": "gemini-2.5-flash-lite",
        "fallback": "gemini-2.5-flash",
    }

    @staticmethod
    def get_config(
        tier: Tier = "fast",
        temperature: float = 0.7,
        json_schema: type | dict | None = None,
    ) -> LLMConfig:
        if tier == "codex":
            return LLMConfig(
                provider="codex", model=ModelRouter.MODELS["codex"],
                max_tokens=8192, temperature=temperature, json_schema=json_schema,
            )
        elif tier == "codex_max":
            return LLMConfig(
                provider="codex", model=ModelRouter.MODELS["codex_max"],
                max_tokens=16384, temperature=temperature, json_schema=json_schema,
                extra={"reasoning_effort": "xhigh"},
            )
        elif tier in ("reasoner", "critical"):
            return LLMConfig(
                provider="google", model=ModelRouter.MODELS["reasoner"],
                max_tokens=8192, temperature=temperature, json_schema=json_schema,
            )
        elif tier == "mutator":
            return LLMConfig(
                provider="google", model=ModelRouter.MODELS["mutator"],
                max_tokens=4096, temperature=temperature, json_schema=json_schema,
            )
        elif tier == "budget":
            return LLMConfig(
                provider="google", model=ModelRouter.MODELS["budget"],
                max_tokens=2048, temperature=temperature, json_schema=json_schema,
            )
        elif tier == "fallback":
            return LLMConfig(
                provider="google", model=ModelRouter.MODELS["fallback"],
                max_tokens=4096, temperature=temperature, json_schema=json_schema,
            )
        else:  # fast
            return LLMConfig(
                provider="google", model=ModelRouter.MODELS["fast"],
                max_tokens=4096, temperature=temperature, json_schema=json_schema,
            )
