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

## Active Models (March 2026)
| Tier | Model ID | Provider | Notes |
|------|----------|----------|-------|
| codex | gpt-5.3-codex | Codex CLI | SOTA coding, provider="codex" in cards.toml |
| reasoner | gemini-3.1-pro-preview | Google | Complex evaluation |
| fast | gemini-3.1-flash-lite-preview | Google | Low-latency |
| budget | gemini-2.5-flash-lite | Google | Cheapest |
| topology-sft | gpt-5.4 | OpenAI API | Best for SFT data generation (reasoning=high) |
| topology-policy | microsoft/Phi-4-mini-instruct | Local ONNX | 3.8B, MIT, trained via GRPO |

## API Keys (all 6 + Codex CLI)
```
GOOGLE_API_KEY      # Required
OPENAI_API_KEY      # Required for SFT data generation
DEEPSEEK_API_KEY    # Optional (NOT DEEP_SEEK_API_KEY)
GROK_API_KEY        # Optional
KIMI_API_KEY        # Optional
MINIMAX_API_KEY     # Optional
```

## cards.toml (27 models, 7 providers)
- Single source of truth: sage-core/config/cards.toml
- sage-python/config/cards.toml is a SYMLINK
- codex provider = "codex" (not "openai")

## Discovery Cache
~/.sage/discovery_cache/ — 24h TTL. Delete to force refresh.

## ExoCortex
Store: `fileSearchStores/ygnsageresearch-wii7kwkqozrd`
