# LiteLLM Provider Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace 1355 lines of custom provider plumbing (OpenAICompatProvider, GoogleProvider, ProviderPool, connector) with LiteLLM v1.83+ — unified tool-calling, Responses API bridge for gpt-5.4-pro, cost tracking, automatic provider quirks.

**Architecture:** LiteLLM SDK (not proxy) wraps all 7 providers behind `litellm.completion()`. A thin `LiteLLMProvider` adapter implements the existing `LLMProvider` interface so the rest of SAGE (pipeline, TopologyRunner, boot) stays unchanged. The Rust ModelAssigner and cards.toml remain the source of truth for model selection — LiteLLM handles the API call, not the routing decision.

**Tech Stack:** `litellm>=1.83`, Python 3.13, existing `LLMProvider`/`LLMResponse`/`LLMConfig` interfaces.

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| **Create** | `sage-python/src/sage/providers/litellm_provider.py` | LiteLLMProvider class implementing LLMProvider interface |
| **Create** | `sage-python/tests/test_litellm_provider.py` | Unit tests (mocked litellm.acompletion) |
| **Modify** | `sage-python/src/sage/boot_providers.py` | Replace OpenAICompat/Google init with LiteLLMProvider |
| **Modify** | `sage-python/src/sage/boot_pipeline.py` | Wire LiteLLMProvider into pipeline |
| **Modify** | `sage-python/src/sage/llm/provider_pool.py` | Simplify — LiteLLM handles provider resolution |
| **Modify** | `sage-python/src/sage/llm/base.py` | No change needed (LLMProvider interface stays) |
| **Modify** | `sage-python/pyproject.toml` | Add `litellm>=1.83` dependency |
| **Keep** | `sage-python/src/sage/providers/openai_compat.py` | Keep as fallback (don't delete yet) |
| **Keep** | `sage-python/src/sage/llm/google.py` | Keep as fallback |
| **Keep** | `sage-core/config/cards.toml` | Unchanged — Rust ModelAssigner still uses it |

---

### Task 1: Add LiteLLM Dependency + Verify Installation

**Files:**
- Modify: `sage-python/pyproject.toml:18-25`

- [ ] **Step 1: Add litellm to dependencies**

```toml
dependencies = [
    "httpx>=0.28",
    "pydantic>=2.10",
    "rich>=13",
    "anyio>=4",
    "aiosqlite>=0.20",
    "numpy>=1.26",
    "truststore>=0.9",
    "litellm>=1.83",
]
```

- [ ] **Step 2: Install and verify**

Run: `pip install -e ".[all,dev]"`
Then: `python -c "import litellm; print(litellm.__version__)"`
Expected: `1.83.x`

- [ ] **Step 3: Verify LiteLLM sees our API keys**

```python
python -c "
import litellm, os
# LiteLLM reads standard env vars automatically
for key in ['OPENAI_API_KEY', 'DEEPSEEK_API_KEY', 'GOOGLE_API_KEY']:
    print(f'{key}: {\"set\" if os.environ.get(key) else \"missing\"}')
# Quick test
resp = litellm.completion(model='deepseek/deepseek-chat', messages=[{'role':'user','content':'ping'}], max_tokens=5)
print(f'DeepSeek OK: {resp.choices[0].message.content[:20]}')
"
```

Expected: DeepSeek responds.

- [ ] **Step 4: Commit**

```bash
git add sage-python/pyproject.toml
git commit -m "deps: add litellm>=1.83 for unified multi-provider LLM gateway"
```

---

### Task 2: Create LiteLLMProvider Adapter

**Files:**
- Create: `sage-python/src/sage/providers/litellm_provider.py`
- Create: `sage-python/tests/test_litellm_provider.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_litellm_provider.py
"""Tests for LiteLLMProvider adapter."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from sage.llm.base import LLMConfig, Message, Role, ToolCall


@pytest.mark.asyncio
async def test_generate_basic():
    """LiteLLMProvider.generate returns LLMResponse with content."""
    from sage.providers.litellm_provider import LiteLLMProvider

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello world"
    mock_response.choices[0].message.tool_calls = None
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_response._hidden_params = {"response_cost": 0.001}

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
        provider = LiteLLMProvider(litellm_model="deepseek/deepseek-chat")
        config = LLMConfig(provider="deepseek", model="deepseek-chat")
        resp = await provider.generate(
            messages=[Message(role=Role.USER, content="test")],
            config=config,
        )
        assert resp.content == "Hello world"
        assert resp.usage["total_tokens"] == 15
        assert resp.tool_calls == []


@pytest.mark.asyncio
async def test_generate_with_tools():
    """LiteLLMProvider.generate parses tool_calls."""
    from sage.providers.litellm_provider import LiteLLMProvider

    mock_tc = MagicMock()
    mock_tc.id = "call_123"
    mock_tc.function.name = "execute_bash"
    mock_tc.function.arguments = '{"command": "ls"}'

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = ""
    mock_response.choices[0].message.tool_calls = [mock_tc]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_response._hidden_params = {"response_cost": 0.001}

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
        provider = LiteLLMProvider(litellm_model="openai/gpt-5.4")
        config = LLMConfig(provider="openai", model="gpt-5.4")
        tools = [{"type": "function", "function": {"name": "execute_bash", "parameters": {}}}]
        resp = await provider.generate(
            messages=[Message(role=Role.USER, content="test")],
            config=config,
            tools=tools,
        )
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "execute_bash"
        assert resp.tool_calls[0].id == "call_123"


@pytest.mark.asyncio
async def test_generate_maps_messages():
    """Messages with tool_call_id are correctly mapped."""
    from sage.providers.litellm_provider import LiteLLMProvider

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "done"
    mock_response.choices[0].message.tool_calls = None
    mock_response.usage.prompt_tokens = 20
    mock_response.usage.completion_tokens = 3
    mock_response.usage.total_tokens = 23
    mock_response._hidden_params = {"response_cost": 0.002}

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response) as mock_call:
        provider = LiteLLMProvider(litellm_model="openai/gpt-5.4")
        config = LLMConfig(provider="openai", model="gpt-5.4")
        messages = [
            Message(role=Role.USER, content="hello"),
            Message(role=Role.ASSISTANT, content="", tool_calls=[
                ToolCall(id="call_1", name="execute_bash", arguments={"command": "ls"})
            ]),
            Message(role=Role.TOOL, content="file.txt", tool_call_id="call_1", name="execute_bash"),
        ]
        await provider.generate(messages=messages, config=config)

        # Verify LiteLLM received properly formatted messages
        call_args = mock_call.call_args
        sent_messages = call_args.kwargs["messages"]
        assert sent_messages[1]["role"] == "assistant"
        assert sent_messages[1]["tool_calls"][0]["id"] == "call_1"
        assert sent_messages[2]["role"] == "tool"
        assert sent_messages[2]["tool_call_id"] == "call_1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_litellm_provider.py -v`
Expected: FAIL — `ImportError: cannot import name 'LiteLLMProvider'`

- [ ] **Step 3: Implement LiteLLMProvider**

```python
# sage-python/src/sage/providers/litellm_provider.py
"""LiteLLM-backed provider — unified interface for all LLM APIs.

Replaces OpenAICompatProvider + GoogleProvider with a single adapter.
LiteLLM handles provider quirks (tool formats, temperature limits,
json_schema support, Responses API bridge) transparently.

Model prefix determines the provider:
  openai/gpt-5.4            → OpenAI Chat Completions
  openai/responses/gpt-5.4-pro → OpenAI Responses API
  deepseek/deepseek-chat    → DeepSeek
  gemini/gemini-3.1-pro     → Google GenAI
  xai/grok-4-1-fast-reasoning → xAI
"""
from __future__ import annotations

import json
import logging
from typing import Any

from sage.llm.base import LLMConfig, LLMProvider, LLMResponse, Message, ToolCall

log = logging.getLogger(__name__)

# Map SAGE provider names to LiteLLM prefixes
_PROVIDER_PREFIX = {
    "openai": "openai",
    "deepseek": "deepseek",
    "google": "gemini",
    "xai": "xai",
    "kimi": "openai",  # Kimi uses OpenAI-compat
    "minimax": "openai",  # MiniMax uses OpenAI-compat
    "openrouter": "openrouter",
}

# Kimi and MiniMax need explicit base_url since litellm prefix "openai" defaults to api.openai.com
_CUSTOM_BASE_URLS = {
    "kimi": "https://api.moonshot.ai/v1",
    "minimax": "https://api.minimax.io/v1",
}


class LiteLLMProvider(LLMProvider):
    """Provider adapter backed by LiteLLM.

    Parameters
    ----------
    litellm_model : str
        Full LiteLLM model string (e.g. "deepseek/deepseek-chat").
    api_key : str, optional
        API key. If None, LiteLLM reads from env vars.
    base_url : str, optional
        Custom base URL (for Kimi, MiniMax, etc.).
    """

    name = "litellm"

    def __init__(
        self,
        litellm_model: str = "",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.litellm_model = litellm_model
        self.api_key = api_key
        self.base_url = base_url
        # Extract model_id for compatibility (e.g. "deepseek/deepseek-chat" → "deepseek-chat")
        self.model_id = litellm_model.split("/", 1)[-1] if "/" in litellm_model else litellm_model

    async def generate(
        self,
        messages: list[Message],
        tools: list | None = None,
        config: LLMConfig | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate via LiteLLM. Handles all provider quirks transparently."""
        import litellm

        # Build model string: use config if provided, else default
        model = self.litellm_model
        if config and config.model:
            provider_name = config.provider or ""
            prefix = _PROVIDER_PREFIX.get(provider_name, "")
            if prefix and not config.model.startswith(prefix):
                model = f"{prefix}/{config.model}"
            else:
                model = config.model

        # Convert SAGE Messages to LiteLLM format (OpenAI dicts)
        litellm_messages = self._convert_messages(messages)

        # Build kwargs
        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": litellm_messages,
        }
        if config:
            if config.max_tokens:
                call_kwargs["max_tokens"] = config.max_tokens
            if config.temperature is not None:
                call_kwargs["temperature"] = config.temperature
        if tools:
            call_kwargs["tools"] = tools
        if self.api_key:
            call_kwargs["api_key"] = self.api_key
        if self.base_url:
            call_kwargs["api_base"] = self.base_url

        try:
            response = await litellm.acompletion(**call_kwargs)
        except Exception as e:
            log.error("LiteLLM error (%s): %s", model, e)
            raise

        # Parse response
        msg = response.choices[0].message
        content = msg.content or ""

        # Token usage (always populated by LiteLLM)
        usage = None
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens or 0,
                "output_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }

        # Cost tracking (LiteLLM-specific)
        cost = getattr(response, "_hidden_params", {}).get("response_cost", 0.0)
        if usage:
            usage["cost_usd"] = cost

        # Parse tool_calls
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (ValueError, TypeError):
                        args = {"raw": args}
                tool_calls.append(ToolCall(
                    id=tc.id or f"call_{len(tool_calls)}",
                    name=tc.function.name,
                    arguments=args,
                ))

        return LLMResponse(
            content=content,
            model=model,
            usage=usage,
            tool_calls=tool_calls,
        )

    @staticmethod
    def _convert_messages(messages: list[Message]) -> list[dict[str, Any]]:
        """Convert SAGE Messages to OpenAI dict format.

        LiteLLM handles provider-specific translation internally,
        so we just produce standard OpenAI format.
        """
        result: list[dict[str, Any]] = []
        for msg in messages:
            d: dict[str, Any] = {"role": msg.role.value, "content": msg.content}

            # Assistant with tool_calls
            if msg.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
                if not d["content"]:
                    d["content"] = None  # OpenAI requires null not ""

            # Tool result
            if msg.tool_call_id:
                d["tool_call_id"] = msg.tool_call_id
            if msg.name:
                d["name"] = msg.name

            result.append(d)
        return result

    @staticmethod
    def for_sage_provider(
        provider_name: str,
        model_id: str,
        api_key: str | None = None,
    ) -> "LiteLLMProvider":
        """Factory: create a LiteLLMProvider for a SAGE provider/model pair."""
        prefix = _PROVIDER_PREFIX.get(provider_name, "openai")
        litellm_model = f"{prefix}/{model_id}"
        base_url = _CUSTOM_BASE_URLS.get(provider_name)
        return LiteLLMProvider(
            litellm_model=litellm_model,
            api_key=api_key,
            base_url=base_url,
        )
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_litellm_provider.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add sage-python/src/sage/providers/litellm_provider.py sage-python/tests/test_litellm_provider.py
git commit -m "feat: LiteLLMProvider adapter — unified interface for all providers"
```

---

### Task 3: Wire LiteLLMProvider into Boot

**Files:**
- Modify: `sage-python/src/sage/boot_providers.py`
- Modify: `sage-python/src/sage/boot_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# In tests/test_boot.py — add this test
@pytest.mark.asyncio
async def test_boot_uses_litellm_provider():
    """Boot creates LiteLLMProvider instead of OpenAICompatProvider."""
    from sage.boot import boot_agent_system
    system = boot_agent_system()
    pipe = system.pipeline
    if pipe and pipe.llm_provider:
        assert "LiteLLM" in type(pipe.llm_provider).__name__ or "litellm" in type(pipe.llm_provider).__name__
```

- [ ] **Step 2: Modify boot_providers.py to create LiteLLMProvider instances**

In `boot_providers.py`, find where `OpenAICompatProvider` instances are created for each provider. Replace with `LiteLLMProvider.for_sage_provider()`:

```python
# Replace OpenAICompatProvider creation loop with:
from sage.providers.litellm_provider import LiteLLMProvider

for cfg in provider_configs:
    provider_name = cfg["provider"]
    api_key = os.environ.get(cfg["api_key_env"], "")
    if not api_key:
        continue
    model_id = cfg.get("default_model", "")
    _runtime_adapters[provider_name] = LiteLLMProvider.for_sage_provider(
        provider_name=provider_name,
        model_id=model_id,
        api_key=api_key,
    )
```

Also replace the default provider (currently `OpenAICompatProvider` for DeepSeek):

```python
# Default provider
default_provider = LiteLLMProvider.for_sage_provider(
    provider_name="deepseek",
    model_id="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
)
```

- [ ] **Step 3: Modify boot_pipeline.py — pass LiteLLM providers to ProviderPool**

No structural change needed — `_runtime_adapters` dict is already passed to `ProviderPool(providers=_runtime_adapters)`. LiteLLMProvider implements `LLMProvider`, so it plugs in directly.

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/test_boot.py tests/test_pipeline.py tests/test_litellm_provider.py -v`
Expected: All pass

- [ ] **Step 5: Smoke test with real API**

```bash
set -a && source .env && set +a
python -c "
from sage.boot import boot_agent_system
import asyncio
s = boot_agent_system()
result = asyncio.get_event_loop().run_until_complete(s.run('What is 2+2?'))
print(f'Result: {result[:100]}')
print(f'Provider type: {type(s.pipeline.llm_provider).__name__}')
"
```

Expected: Response from DeepSeek via LiteLLM. Provider type: LiteLLMProvider.

- [ ] **Step 6: Commit**

```bash
git add sage-python/src/sage/boot_providers.py sage-python/src/sage/boot_pipeline.py
git commit -m "feat: boot uses LiteLLMProvider for all 7 providers"
```

---

### Task 4: Simplify ProviderPool — LiteLLM Handles Resolution

**Files:**
- Modify: `sage-python/src/sage/llm/provider_pool.py`

- [ ] **Step 1: Simplify resolve() — LiteLLMProvider handles prefix routing**

The `resolve()` method currently does complex provider name inference and fallback logic. With LiteLLM, each `LiteLLMProvider` already knows its model prefix. The pool just needs to find the right provider instance by provider name.

Simplify `resolve()`:
```python
def resolve(self, model_id: str) -> tuple[LLMProvider, LLMConfig]:
    """Resolve model_id to (provider, config).

    With LiteLLM, the provider handles routing via its prefix.
    We just need to find which LiteLLMProvider to use.
    """
    # Check cache
    if model_id in self._cache:
        return self._cache[model_id]

    # Infer provider name
    pname = self.infer_provider(model_id)

    # Get provider instance (or default)
    provider = self._providers.get(pname, self._default)

    # Check circuit breaker
    if pname and not self.is_available(pname):
        provider = self._default

    config = LLMConfig(provider=pname or "default", model=model_id)
    result = (provider, config)
    self._cache[model_id] = result
    return result
```

- [ ] **Step 2: Remove supports_chat_completions_model — LiteLLM bridges Responses API**

The `supports_chat_completions_model()` hack is no longer needed. LiteLLM handles gpt-5.4-pro via `openai/responses/gpt-5.4-pro` prefix. Remove the import and checks from `is_model_available()` and `resolve()`.

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_pipeline.py tests/test_provider_pool.py -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add sage-python/src/sage/llm/provider_pool.py
git commit -m "refactor: simplify ProviderPool — LiteLLM handles provider quirks"
```

---

### Task 5: Update cards.toml Model IDs for Responses API

**Files:**
- Modify: `sage-core/config/cards.toml`

- [ ] **Step 1: Update gpt-5.4-pro to use Responses API prefix**

In `cards.toml`, the model ID `gpt-5.4-pro` is currently filtered out. With LiteLLM, it works via the `openai/responses/` prefix. Update the card so `select_for_system()` can return it and LiteLLM will route correctly:

The `LiteLLMProvider.for_sage_provider("openai", "gpt-5.4-pro")` will produce `openai/gpt-5.4-pro`. For Responses API, we need a way to indicate this model needs the responses prefix. Options:
- Add a `responses_api = true` flag to cards.toml
- Or handle in LiteLLMProvider._build_model_string()

For now, handle in the provider: if model matches `gpt-5*-pro`, use `openai/responses/` prefix.

- [ ] **Step 2: Commit**

```bash
git add sage-core/config/cards.toml sage-python/src/sage/providers/litellm_provider.py
git commit -m "feat: gpt-5.4-pro accessible via LiteLLM Responses API bridge"
```

---

### Task 6: Integration Test — SWE-bench with LiteLLM

**Files:**
- No new files — run existing SWE-bench

- [ ] **Step 1: Run SWE-bench Lite 1 task with LiteLLM**

```bash
set -a && source .env && set +a
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
python -u -m sage.bench --type swebench --dataset lite --limit 1 --generate-only
```

- [ ] **Step 2: Monitor per protocol**

```bash
# Check: model used, tool calls, cost
grep "upgraded to\|Tool call:\|LiteLLM\|cost" output.log
```

Expected: LiteLLMProvider used, tool_calls work, cost tracked.

- [ ] **Step 3: Commit results**

```bash
git add docs/benchmarks/
git commit -m "data: SWE-bench Lite with LiteLLM provider — first run"
```

---

### Task 7: Cleanup — Remove Dead Provider Code (Optional, Post-Validation)

**Files:**
- Delete (after 1 week of LiteLLM stability): `sage-python/src/sage/providers/openai_compat.py`
- Delete: `sage-python/src/sage/llm/google.py`
- Simplify: `sage-python/src/sage/providers/connector.py` (discovery still useful)

Do NOT delete in this PR. Keep as fallback. Tag with `# DEPRECATED: replaced by LiteLLMProvider` comment.

- [ ] **Step 1: Add deprecation comments**

- [ ] **Step 2: Commit**

```bash
git commit -m "docs: mark OpenAICompatProvider + GoogleProvider as deprecated (LiteLLM replaces)"
```
