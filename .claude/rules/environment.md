---
paths:
  - "**/*.py"
  - "**/*.toml"
  - "**/*.yml"
---

# Environment & LLM Configuration

## NO Corporate Proxy
Standard HTTPS. Never add verify=False. See critical-directives.md.

## Active Models (March 2026)
| Tier | Model ID | Provider | Notes |
|------|----------|----------|-------|
| codex | gpt-5.3-codex | Codex CLI | SOTA coding, provider="codex" in cards.toml |
| reasoner | gemini-3.1-pro-preview | Google | Complex evaluation |
| fast | gemini-3.1-flash-lite-preview | Google | Low-latency |
| budget | gemini-2.5-flash-lite | Google | Cheapest |

## API Keys (all 6 + Codex CLI)
```
GOOGLE_API_KEY      # Required
OPENAI_API_KEY      # Optional
DEEPSEEK_API_KEY    # Optional (NOT DEEP_SEEK_API_KEY)
GROK_API_KEY        # Optional
KIMI_API_KEY        # Optional
MINIMAX_API_KEY     # Optional
# Codex CLI: codex login (ChatGPT Pro account)
```

## cards.toml (27 models, 7 providers)
- Rust reads: sage-core/config/cards.toml
- Python reads: sage-python/config/cards.toml + config/model_profiles.toml
- BOTH must be in sync for codex provider="codex" (not "openai")

## Discovery Cache
Provider discovery cached 24h at ~/.sage/discovery_cache/. Delete to force refresh.
Boot time: ~2s (cached) vs 60s+ (cold).

## ExoCortex
Auto-configured: `fileSearchStores/ygnsageresearch-wii7kwkqozrd`
Query via: `search_exocortex` agent tool or direct Google GenAI File Search API.
