# LiteLLM → Pydantic AI Migration Plan (2026-04-18)

**Status**: Phase 1 (Discovery) — this doc is the output.
**Trigger**: User pushed back after noticing I was re-implementing
LiteLLM's quirk table by trial and error. LiteLLM 1.83.9 registry still
incorrectly lists `temperature` as supported for `openai/gpt-5.4`, and
`register_model` doesn't let us override `supported_openai_params`.
Pydantic AI has native providers for every model in cards.toml and a
cleaner custom-base pattern for edge cases.

## 1. Contract our new provider must honor

`sage-python/src/sage/llm/base.py::LLMProvider` Protocol:

```python
@runtime_checkable
class LLMProvider(Protocol):
    name: str

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        config: LLMConfig | None = None,
        tool_choice: str | None = None,
    ) -> LLMResponse: ...


class StreamingLLMProvider(LLMProvider, Protocol):
    async def generate_stream(
        self,
        messages: list[Message],
        config: LLMConfig | None = None,
    ) -> AsyncIterator[str]: ...
```

`LLMResponse` fields: `content`, `tool_calls: list[ToolCall]`, `usage:
dict[str, int]`, `model`, `stop_reason`.

**The new `PydanticAIProvider` must expose exactly this shape** — no
boot-site changes beyond a single factory swap.

## 2. LiteLLMProvider call sites (enumerated)

Live production callers (2 files, 4 sites):

| File | Line | Purpose |
|------|------|---------|
| `sage-python/src/sage/boot_providers.py` | 53 | import |
| `sage-python/src/sage/boot_providers.py` | 65 | `LiteLLMProvider.for_sage_provider(prov_name, model_id, api_key)` in ProviderPool factory |
| `sage-python/src/sage/boot_providers.py` | 72 | second factory branch for discovered providers |
| `sage-python/src/sage/boot_pipeline.py` | 74 | import |
| `sage-python/src/sage/boot_pipeline.py` | 88 | `_runtime_adapters[_pname] = LiteLLMProvider.for_sage_provider(...)` |

Docstring / comment references (no code change needed):
- `sage-python/src/sage/phases/think.py:34` — explaining cost telemetry wiring
- `sage-python/src/sage/providers/openai_compat.py` — deprecated, points at LiteLLMProvider

**Migration surface: 4 call sites + 1 new provider class + 1 factory method.**

## 3. Pydantic AI API for our 7 providers

Verified via Context7 `/pydantic/pydantic-ai` (v1.71.0):

| cards.toml provider | Pydantic AI path | Native? |
|---|---|---|
| `openai` | `Agent('openai:gpt-5.2')` or `OpenAIChatModel + OpenAIProvider` | ✅ native |
| `google` | `Agent('google-gla:gemini-3-flash-preview')` or `GoogleModel + GoogleProvider(vertexai=False)` | ✅ native |
| `xai` | `Agent('groq:...')` — **verify** — we want grok. Listed as supported in search results | ⚠️ verify in Phase 2 |
| `deepseek` | Unclear from Context7 — DeepSeek is OpenAI-compatible, use custom base_url via `OpenAIProvider(base_url='https://api.deepseek.com/v1', ...)` | ✅ via openai-compat |
| `minimax` | Not in native provider list — use `OpenAIProvider(base_url='https://api.minimax.io/v1', ...)` if OpenAI-compatible. **Verify in Phase 2.** | ⚠️ via openai-compat |
| `kimi` | `Agent('moonshotai:kimi-k2-0711-preview')` — native Moonshot provider | ✅ native |
| `openrouter` | `Agent('openrouter:...')` native per earlier web search | ✅ native |

**Custom base_url pattern** (for anything OpenAI-compatible without a native provider):

```python
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIChatModel(
    'deepseek-chat',
    provider=OpenAIProvider(
        base_url='https://api.deepseek.com/v1',
        api_key='sk-...'
    ),
)
agent = Agent(model)
```

Clean, first-class, no substring detection or quirk tables.

## 4. Streaming API

`agent.run_stream(prompt)` async context manager, `.stream_text()`
yields text chunks. Full event API (`run_stream_events`) exposes
`PartStartEvent`, `PartDeltaEvent` (with `TextPartDelta`,
`ThinkingPartDelta`, `ToolCallPartDelta`), `FunctionToolCallEvent`,
`FunctionToolResultEvent`, `FinalResultEvent`. Our `generate_stream`
returns plain text chunks — map `.stream_text()` to that.

## 5. Tool calling API

Two patterns:

* **Decorator** (Python-native functions, the default):
  ```python
  @agent.tool
  async def weather_forecast(ctx, location: str, forecast_date: date) -> str: ...
  ```
* **Schema-level** (for our use case — topologies define tools at
  runtime, not at decoration time):
  ```python
  from pydantic_ai import ModelRequestParameters, ToolDefinition
  params = ModelRequestParameters(function_tools=[ToolDefinition(
      name='...', description='...', parameters_json_schema={...}
  )])
  response = await model_request('openai:gpt-5.2', messages, model_request_parameters=params)
  ```

Our `ToolDef` dataclass maps 1:1 onto `ToolDefinition`. Tool-call
extraction: `ModelResponse.parts` contains `ToolCallPart`s with
`tool_name`, `tool_call_id`, `args` (already parsed JSON, dict).

`tool_choice` parameter: Pydantic AI's schema-level API doesn't
directly expose tool_choice in the top-level; it's handled via
`ModelRequestParameters.output_mode` and `allow_text_output`. **Open
question for Phase 2** — how to force "required" tool call behavior.

## 6. Usage and cost — OPEN QUESTION

**This is the biggest unknown.** Pydantic AI exposes:

```python
result.usage()  # RunUsage(input_tokens=66, output_tokens=16, requests=1)
```

That's **token counts only, no per-call provider-reported cost**.
Cost computation is via an optional plugin (`genai-prices` via
Logfire integration) that uses a local price table — structurally
identical to LiteLLM's `_COST_PER_1K` which we had to patch around in
P0.3.

**Risk**: migrating to Pydantic AI regresses our P0.3 "prefer
provider-reported cost" fix. We'd be back to token × local table
estimates.

**Mitigation options for Phase 2**:
1. Check if `ModelResponse` exposes the raw provider response where
   OpenAI's `usage.total_cost` or Moonshot's metering lives.
2. If not, contribute to Pydantic AI upstream — propose a
   `provider_reported_cost` attribute (they have a good PR culture).
3. Accept the regression and move cost telemetry back to estimate,
   documenting the tradeoff.

## 7. Plan the execution

Phases already in task list (#35–#39):

1. **Phase 1 (this doc)** — done.
2. **Phase 2: prototype** — new file
   `sage-python/src/sage/providers/pydantic_ai_provider.py`
   implementing `LLMProvider` protocol. Test on DeepSeek first (least
   quirky, single model, no temp restrictions).
3. **Phase 3: live tests** — 19 models × 1 "hello" test each. Tag
   `live_provider`, opt-in in CI. Document quirks per provider.
4. **Phase 4: Rust flow validation** — ProviderPool.resolve() +
   ModelAssigner.assign() + SystemRouter smoke still pass for every
   provider now backed by PydanticAIProvider.
5. **Phase 5: cleanup** — swap factory in `boot_providers.py` +
   `boot_pipeline.py`, delete `litellm_provider.py` + `openai_compat.py`,
   remove `litellm` from `pyproject.toml`.

## 8. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Cost telemetry regression (§6) | Investigate in Phase 2 before committing; may keep LiteLLM alongside for cost-only path |
| xAI (Grok) native support unclear | Phase 2 live test on grok-3 confirms. Custom base_url fallback if needed (xAI is OpenAI-compatible) |
| MiniMax not in native provider list | Use custom `OpenAIProvider(base_url=...)` — same pattern we've been using, but cleaner |
| Tool-choice "required" missing | Phase 2 verify — if missing, file upstream issue or monkey-patch `ModelRequestParameters` |
| Breaking sage-core contracts | Phase 4 smoke tests. Rust provider-pool / ModelAssigner doesn't care about the provider CLASS, just the protocol — should be transparent |
| Migration disrupts current iter5 SWE-Lite work | Do NOT migrate until baseline infra is stable (task #34). Meta-Harness is paused anyway |

## 9. Success criteria

Migration is complete when ALL of:

- `grep -r litellm sage-python/src/` returns zero matches
- `pyproject.toml` no longer lists `litellm` as a dependency
- `pytest tests/` is green on pre-migration count ± 0
- For each of 7 providers, a live-tagged test passes with a real API
  call (19 model smoke tests pass)
- ProviderPool circuit-breaker integration test still passes
- One SWE-Lite 5-task smoke bench runs with 0 infra errors (the
  baseline-stability criterion from task #34 — orthogonal but a good
  sanity check)
