---
paths:
  - "**/*.py"
  - "**/*.toml"
  - "**/*.yml"
---

# Environment & LLM Configuration

## SSL / Corporate Proxy Certificates
CA certificates at `C:\Code\certs\`:
- `ca-bundle.pem` — full CA bundle (use for REQUESTS_CA_BUNDLE, SSL_CERT_FILE)
- `ca.cacrt` — corporate CA root certificate
- `client.crt` + `client.key` + `client.pem` — client certificates

For Python scripts, set these env vars:
```bash
export REQUESTS_CA_BUNDLE=C:/Code/certs/ca-bundle.pem
export SSL_CERT_FILE=C:/Code/certs/ca-bundle.pem
export CURL_CA_BUNDLE=C:/Code/certs/ca-bundle.pem
```

For httpx (HuggingFace Hub), also pass `verify="C:/Code/certs/ca-bundle.pem"` to httpx.Client().

**NEVER use verify=False when ca-bundle.pem is available.** Use the proper certificate instead.

## Active Models (May 10, 2026)
| Tier | Model ID | Provider | Notes |
|------|----------|----------|-------|
| codex | gpt-5.4 | OpenAI | Runtime chat/coding tier in `sage-python/src/sage/llm/router.py` |
| codex_max | gpt-5.4-pro | OpenAI | Higher-cost OpenAI tier |
| reasoner | gemini-3.1-pro-preview | Google | Complex evaluation ($2.00/$12.00) |
| fast | gemini-3.1-flash-lite-preview | Google | Low-latency ($0.25/$1.50) |
| budget | deepseek-v4-flash | DeepSeek | Budget default; thinking disabled via runtime settings |
| budget-pro | deepseek-v4-pro | DeepSeek | Active V4 Pro; thinking enabled via runtime settings |
| budget-alt | grok-4-1-fast-reasoning | xAI | 2M context, $0.20/$0.50 |
| topology-sft | gpt-5.4 | OpenAI | SFT data generation |
| topology-policy | nvidia/Nemotron-Orchestrator-8B | veRL training | NVIDIA Open Model License, Qwen3 architecture, GRPO-trained orchestrator. GiGPO on RunPod H100 |
| new | MiniMax-M2.7 | MiniMax | Official capitalization; 204.8k text context |
| new | qwen/qwen3.5-plus-02-15 | OpenRouter | Qwen3.5-Plus via OpenRouter ($0.26/$1.56) |
| new | kimi-k2.6 | Kimi/Moonshot | Current Kimi model card |
| new | gpt-5.4-mini | OpenAI | Budget frontier ($0.75/$4.50) |

DeepSeek legacy aliases `deepseek-chat` and `deepseek-reasoner` are not
runtime-selectable. They rewrite to `deepseek-v4-flash` per
`sage-core/config/cards.toml`.

## API Keys (7 API providers; Codex/OpenAI tiers use OpenAI routing)
```
GOOGLE_API_KEY        # Required
OPENAI_API_KEY        # Required
DEEPSEEK_API_KEY      # Required (primary training provider)
GROK_API_KEY          # Optional
KIMI_API_KEY          # Optional
MINIMAX_API_KEY       # Optional
OPEN_ROUTER_API_KEY   # For Qwen3.5-Plus
```

## cards.toml (24 model cards, 7 API providers)
- Single source of truth for model IDs, costs, context windows, runtime selectability, and runtime settings: sage-core/config/cards.toml
- Provider connection settings live in sage-python/src/sage/providers/connector.py
- sage-python/config/cards.toml is a SYMLINK
- Providers: google, openai, deepseek, xai, minimax, kimi, openrouter

## Discovery Cache
~/.sage/discovery_cache/ — 24h TTL. Delete to force refresh.

## ExoCortex
Store: `fileSearchStores/ygnsageresearch-wii7kwkqozrd`

**Not auto-defaulted since 2026-04-18** (P1.3 multi-tenant fix, commit
e338b7e). Set `SAGE_EXOCORTEX_STORE` in your `.env`:

```
SAGE_EXOCORTEX_STORE=fileSearchStores/ygnsageresearch-wii7kwkqozrd
```

If unset: ExoCortex features no-op silently (one-shot WARN). Fine for
SWE-bench / code-repair benches (they use repo-local tools, not File
Search). Required for `search_exocortex` agent tool to return papers.

See `.env.example` for the full reference template.
