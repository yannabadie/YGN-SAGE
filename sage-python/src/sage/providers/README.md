# providers

Provider discovery, capability management, and multi-provider routing infrastructure. Separates model discovery (which LLMs are available) from model selection (which LLM to use for a given task).

## Modules

### `capabilities.py` -- CapabilityMatrix

Semantic capability matrix for hard-fail filtering. Each provider declares its capabilities (structured output, tool role, file search, grounding, streaming). The matrix exposes `providers_for(**requirements)` to find compatible providers and `require(**requirements)` to fail fast when none match.

- **Key exports**: `CapabilityMatrix`, `ProviderCapabilities`

### `connector.py` -- ProviderConnector

Auto-discovers available LLM models by querying provider APIs at boot time. Supports Google (`google-genai` SDK), OpenAI, and Codex CLI. Failures are isolated per-provider -- one provider going down never crashes boot.

- **Key exports**: `ProviderConnector`, `DiscoveredModel`, `PROVIDER_CONFIGS`

### `openai_compat.py` -- OpenAICompatProvider

Generic provider for any OpenAI-compatible API: OpenAI, xAI (Grok), DeepSeek, MiniMax, Kimi/Moonshot. Logs a WARNING when converting TOOL messages to USER role (semantic loss for providers that lack native tool role support).

- **Key exports**: `OpenAICompatProvider`

### `registry.py` -- ModelRegistry

Combines live provider discovery (via `ProviderConnector`) with curated benchmark data from `config/model_profiles.toml`. Produces `ModelProfile` objects with scoring fields (reasoning, coding, speed) and economics (cost per 1K tokens). Used by `CognitiveOrchestrator` for capability-based model selection.

- **Key exports**: `ModelRegistry`, `ModelProfile`

## Architecture

```
ProviderConnector          ModelRegistry
  |-- discover() --------->  merge with TOML profiles
  |-- Google API             |
  |-- OpenAI API             +--> ModelProfile[]
  |-- Codex CLI              |
                          CapabilityMatrix
                             |-- require(tool_role=True)
                             +--> filtered provider list
```

The `boot_agent_system()` function in `sage/boot.py` wires `ModelRegistry` and `CognitiveOrchestrator` together. In mock mode (tests), the legacy `ModelRouter` path is used directly.
