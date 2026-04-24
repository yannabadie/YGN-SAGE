"""Live-API matrix: one ping per cards.toml model via PydanticAIProvider.

Phase 3 of LiteLLM → Pydantic AI migration. Validates that the wrapper
works for every model we ship in `sage-core/config/cards.toml`. Each
test:

  1. Instantiates `PydanticAIProvider.for_sage_provider(provider, model_id, api_key)`.
  2. Sends one `Say 'hi' in 3 words.` user prompt.
  3. Asserts content is non-empty, input/output tokens populated.
  4. Logs discovered cost (from cards.toml) and any provider quirk.

Opt-in because these tests hit real APIs and cost real money.
Set `SAGE_LIVE_PROVIDERS=1` to enable, or run with
`-m live_provider`. Deselected by default via the
`live_provider` marker. Skipped per-test if the required API key env
var is missing.

Estimated total cost per run: ~$0.10–$0.50 (depends on model tiers).
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from sage.llm.base import LLMResponse

# ---------------------------------------------------------------------------
# Env loader (same CRLF-stripping pattern as meta_harness.py)
# ---------------------------------------------------------------------------

_ENV_PATH = Path(__file__).resolve().parents[3] / ".env"


def _load_env() -> None:
    if not _ENV_PATH.is_file():
        return
    for line in _ENV_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    # Pydantic AI env var aliases: our .env ships one name, it expects another.
    if os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
    for target in ("MOONSHOTAI_API_KEY", "MOONSHOT_API_KEY"):
        if os.environ.get("KIMI_API_KEY") and not os.environ.get(target):
            os.environ[target] = os.environ["KIMI_API_KEY"]


_load_env()

# ---------------------------------------------------------------------------
# Opt-in gate — these tests cost real money. Off by default.
# ---------------------------------------------------------------------------

_SKIP_LIVE = not os.environ.get("SAGE_LIVE_PROVIDERS")

pytestmark = [
    pytest.mark.live_provider,
    pytest.mark.skipif(
        _SKIP_LIVE,
        reason="Live provider tests opt-in — set SAGE_LIVE_PROVIDERS=1",
    ),
]


# ---------------------------------------------------------------------------
# Matrix: (provider_name, model_id, api_key_env)
# ---------------------------------------------------------------------------
#
# Source of truth: sage-core/config/cards.toml (listed via
# `grep '^id =' sage-core/config/cards.toml` on 2026-04-18).
#
# Pydantic AI model-id alignment notes:
#   - kimi-k2.6 (A8 migration 2026-04-24) → Pydantic AI's MoonshotAI
#     provider accepts kimi-k2.6 directly per the Moonshot OpenAI-
#     compat endpoint. Live call will succeed if KIMI_API_KEY is set
#     AND the current PydanticAI build recognises the model id. If
#     PydanticAI's moonshot provider still hard-codes a pre-k2.6
#     allowlist, the test will 404 — keeping this visible in Phase 3
#     surfaces the breakage rather than silently skipping.
#   - qwen/qwen3.5-plus-02-15 routes via openrouter/ prefix in
#     Pydantic AI — already handled by OpenRouterModel.
_MATRIX: list[tuple[str, str, str]] = [
    # (provider, model_id, api_key_env_var)
    ("google", "gemini-3.1-pro-preview", "GOOGLE_API_KEY"),
    ("google", "gemini-3.1-flash-lite-preview", "GOOGLE_API_KEY"),
    ("google", "gemini-3-flash-preview", "GOOGLE_API_KEY"),
    ("google", "gemini-2.5-flash", "GOOGLE_API_KEY"),
    ("openai", "gpt-5.5", "OPENAI_API_KEY"),
    ("openai", "gpt-5.5-pro", "OPENAI_API_KEY"),
    ("openai", "gpt-5.4", "OPENAI_API_KEY"),
    ("openai", "gpt-5.4-pro", "OPENAI_API_KEY"),
    ("openai", "gpt-5.2", "OPENAI_API_KEY"),
    ("openai", "gpt-5.4-mini", "OPENAI_API_KEY"),
    ("openai", "gpt-5.4-nano", "OPENAI_API_KEY"),
    ("xai", "grok-4-1-fast-reasoning", "GROK_API_KEY"),
    ("xai", "grok-code-fast-1", "GROK_API_KEY"),
    ("xai", "grok-3", "GROK_API_KEY"),
    ("deepseek", "deepseek-v4-flash", "DEEPSEEK_API_KEY"),
    ("deepseek", "deepseek-v4-pro", "DEEPSEEK_API_KEY"),
    ("deepseek", "deepseek-chat", "DEEPSEEK_API_KEY"),  # LEGACY sunsets 2026-07-24
    ("deepseek", "deepseek-reasoner", "DEEPSEEK_API_KEY"),  # LEGACY sunsets 2026-07-24
    ("minimax", "minimax-m2.7", "MINIMAX_API_KEY"),
    ("minimax", "MiniMax-M2.5", "MINIMAX_API_KEY"),
    ("minimax", "MiniMax-M2.5-highspeed", "MINIMAX_API_KEY"),
    ("openrouter", "qwen/qwen3.5-plus-02-15", "OPEN_ROUTER_API_KEY"),
    ("kimi", "kimi-k2.6", "KIMI_API_KEY"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_name,model_id,api_key_env",
    _MATRIX,
    ids=[f"{p}-{m}" for p, m, _ in _MATRIX],
)
async def test_live_provider_hello(provider_name: str, model_id: str, api_key_env: str) -> None:
    """Hello-world call against a live model; asserts content + usage."""
    api_key = os.environ.get(api_key_env)
    if not api_key:
        pytest.skip(f"{api_key_env} not set")

    from sage.llm.base import LLMConfig, Message, Role
    from sage.providers.pydantic_ai_provider import PydanticAIProvider

    provider = PydanticAIProvider.for_sage_provider(provider_name, model_id, api_key)

    messages = [
        Message(
            role=Role.USER,
            content="Reply with exactly 'ok' and nothing else.",
        ),
    ]
    cfg = LLMConfig(provider=provider_name, model=model_id, max_tokens=20)

    resp: LLMResponse = await asyncio.wait_for(
        provider.generate(messages, config=cfg),
        timeout=60.0,
    )

    # Core contract: content + usage
    assert resp.content, f"empty content from {provider_name}:{model_id}"
    assert resp.usage is not None, f"no usage from {provider_name}:{model_id}"
    assert resp.usage.get("input_tokens", 0) > 0, "no input_tokens"
    assert resp.usage.get("output_tokens", 0) > 0, "no output_tokens"

    # Cost should be non-zero if cards.toml has pricing for this model.
    cost = resp.usage.get("cost_usd", 0.0)
    # Print for the summary (pytest captures — will show on -s or on failure).
    print(
        f"  {provider_name}/{model_id}: "
        f"{resp.usage['input_tokens']}/{resp.usage['output_tokens']} tok, "
        f"cost=${cost:.6f}, content={resp.content!r}"
    )


if __name__ == "__main__":
    # Allow running this file directly for visible output (pytest captures).
    os.environ["SAGE_LIVE_PROVIDERS"] = "1"
    sys.exit(pytest.main([__file__, "-v", "-s", "--tb=short"]))
