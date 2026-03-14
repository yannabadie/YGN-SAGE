# llm

LLM provider abstraction layer with multi-provider support and tiered model routing.

## Modules

### `base.py` -- Core Types

Defines the fundamental types used across all LLM interactions.

- **Key exports**:
  - `Role` -- enum: SYSTEM, USER, ASSISTANT, TOOL
  - `Message` -- role + content + optional tool_call_id/name
  - `ToolDef` -- tool name, description, JSON schema parameters
  - `ToolCall` -- id, name, parsed arguments dict
  - `LLMResponse` -- content, tool_calls, usage stats, model ID, stop_reason
  - `LLMConfig` -- provider name + model ID + optional parameters
  - `LLMProvider` -- Protocol with `async generate(messages, tools, config) -> LLMResponse`

### `router.py` -- ModelRouter

Maps tier names to `LLMConfig` objects. Resolution order: environment variable `SAGE_MODEL_<TIER>` > `config/models.toml` > hardcoded defaults.

- **Key exports**: `ModelRouter`, `Tier`
- **Tiers**: `codex`, `codex_max`, `reasoner`, `mutator`, `fast`, `budget`, `fallback`
- `ModelRouter.get_config(tier)` -- returns `LLMConfig` for the requested tier

### `config_loader.py` -- TOML Config Loader

Loads model configuration from TOML files and resolves model IDs with environment variable overrides.

- **Key exports**: `load_model_config()`, `resolve_model_id()`
- TOML search paths: `cwd/config/`, package `config/`, `~/.sage/`

### `google.py` -- GoogleProvider

Provider for Google Gemini models (3.x series) via the `google-genai` SDK. Supports native grounding, File Search (ExoCortex), and structured output. File Search and Google Search grounding are mutually exclusive.

- **Key exports**: `GoogleProvider`
- `capabilities()` returns: structured_output, tool_role, file_search, grounding, system_prompt, streaming

### `codex.py` -- CodexProvider

Provider for OpenAI Codex CLI (`gpt-4.1` default). Wraps `codex exec` for non-interactive agent calls. Supports structured JSON output via `--output-schema`. Falls back to `GoogleProvider` if the CLI is missing or fails.

- **Key exports**: `CodexProvider`, `DEFAULT_MODEL`

### `mock.py` -- MockProvider

Two mock providers for testing and benchmarking:
- `MockLLMProvider` -- simulates latency for AIO benchmarking
- `MockProvider` -- returns configurable response sequences for unit tests

- **Key exports**: `MockLLMProvider`, `MockProvider`
