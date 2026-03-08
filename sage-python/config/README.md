# Configuration

TOML configuration files for LLM model routing and provider profiles.

## Files

### `models.toml` -- Model Tier Definitions

Defines the 7 model tiers used by the routing system:

| Tier | Purpose | Default Model |
|------|---------|---------------|
| `codex` | Primary coding | gpt-5.3-codex |
| `codex_max` | Maximum reasoning | gpt-5.2 |
| `reasoner` | Complex evaluation | gemini-3.1-pro-preview |
| `mutator` | Code mutation | gemini-3-flash-preview |
| `fast` | Low-latency | gemini-3.1-flash-lite-preview |
| `budget` | Cheapest | gemini-2.5-flash-lite |
| `fallback` | If primary unavailable | gemini-2.5-flash |

Each tier entry specifies: model ID, provider, and optional parameters (temperature, max tokens).

### `model_profiles.toml` -- Provider Profiles and Capabilities

Declares per-provider capability metadata used by the DynamicRouter for constraint-based model selection. Each profile includes:

- Supported capabilities (coding, reasoning, formal_verification)
- Cost per token (input/output)
- Latency estimates
- Context window size

## Resolution Order

Model IDs are resolved in priority order:

1. Environment variable `SAGE_MODEL_<TIER>` (e.g., `SAGE_MODEL_FAST=gemini-2.5-flash`)
2. TOML config file (`models.toml`)
3. Hardcoded defaults in `llm/config_loader.py`

TOML files are searched in: `cwd/config/`, `sage-python/config/` (package), `~/.sage/`.
