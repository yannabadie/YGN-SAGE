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

## Active Models (March 20, 2026)
| Tier | Model ID | Provider | Notes |
|------|----------|----------|-------|
| codex | gpt-5.3-codex | OpenAI | SOTA coding |
| reasoner | gemini-3.1-pro-preview | Google | Complex evaluation ($2.00/$12.00) |
| fast | gemini-3.1-flash-lite-preview | Google | Low-latency ($0.25/$1.50) |
| budget | deepseek-chat | DeepSeek | Best cost/quality ($0.28/$0.42, no rate limits) |
| budget-alt | grok-4-1-fast-reasoning | xAI | 2M context, $0.20/$0.50 |
| topology-sft | gpt-5.4 | OpenAI | SFT data generation |
| topology-policy | Qwen/Qwen3.5-9B | veRL training | Replacing Phi-4-mini via GRPO on RunPod H100 |
| new | minimax-m2.7 | MiniMax | Self-evolving, $0.30/$1.20 |
| new | qwen/qwen3.5-plus-02-15 | OpenRouter | Qwen3.5-Plus via OpenRouter ($0.26/$1.56) |
| new | gpt-5.4-mini | OpenAI | Budget frontier ($0.75/$4.50) |

## API Keys (all 7 + Codex CLI)
```
GOOGLE_API_KEY        # Required
OPENAI_API_KEY        # Required
DEEPSEEK_API_KEY      # Required (primary training provider)
GROK_API_KEY          # Optional
KIMI_API_KEY          # Optional
MINIMAX_API_KEY       # Optional
OPEN_ROUTER_API_KEY   # For Qwen3.5-Plus
```

## cards.toml (20 models, 8 providers)
- Single source of truth: sage-core/config/cards.toml
- sage-python/config/cards.toml is a SYMLINK
- Providers: google, openai, deepseek, xai, minimax, kimi, openrouter, codex

## Discovery Cache
~/.sage/discovery_cache/ — 24h TTL. Delete to force refresh.

## ExoCortex
Store: `fileSearchStores/ygnsageresearch-wii7kwkqozrd`
