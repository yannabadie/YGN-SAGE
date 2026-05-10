---
name: project_may09_provider_status
description: Provider reliability status as of 2026-05-09 — which models work and which fail
type: project
originSessionId: 98c5d292-7098-4166-9e24-6d083e057a81
---
# Provider Status — 2026-05-09

**Working (verified via API calls):**
- Google gemini-2.5-flash: fast, reliable (~587ms)
- DeepSeek deepseek-chat: reliable (~884ms)
- DeepSeek deepseek-v4-flash: works but slow for SWE-bench tasks
- Minimax minimax-m2.7: Yann confirms it works normally

**Silently failing (cost=$0, no API call made):**
- OpenAI gpt-5.4: routing picks it but API calls fail silently
- OpenAI gpt-5.5-pro: **may not be available via API** — Yann not sure it's accessible via API key. The routing picks it but API calls fail.
- OpenRouter qwen/qwen3.5-plus: unknown reliability

**Consequence for canary:**
- Multi-provider debate deadlocks because one failing provider blocks asyncio.gather()
- Fix: restrict providers to confirmed-working ones for canary (google, deepseek, minimax)
- Per-node timeout (SAGE_TOPOLOGY_NODE_TIMEOUT_SEC=120) added but process may crash before timeout