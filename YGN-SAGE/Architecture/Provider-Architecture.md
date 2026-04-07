---
title: Provider Architecture
type: architecture
tags:
  - architecture
  - providers
updated: 2026-04-07
---

# 7 Providers — Architecture Multi-Fournisseur

Source de verite : `sage-python/src/sage/providers/connector.py`

## Providers

| Provider | API URL | Modele par defaut | Role |
|----------|---------|-------------------|------|
| DeepSeek | api.deepseek.com/v1 | deepseek-chat | **Primaire** (moins cher, pas de rate limits) |
| Google | native SDK | gemini-3.1-flash-lite | Fallback fiable |
| OpenAI | api.openai.com/v1 | gpt-5.4 | Meilleure qualite |
| xAI | api.x.ai/v1 | grok-4-1-fast-reasoning | Raisonnement rapide |
| Kimi | api.moonshot.ai/v1 | kimi-k2.5 | Vision + raisonnement |
| MiniMax | api.minimax.io/v1 | minimax-m2.7 | Contexte 4M tokens — **BUG 400** "invalid chat setting" |
| OpenRouter | openrouter.ai/api/v1 | qwen/qwen3.5-plus | Acces 200+ modeles |

## Fonctionnement

- Chaque noeud de topologie peut utiliser un provider different
- La policy model peut exprimer `provider_hint` pour biaiser la selection (+0.15)
- **Circuit breaker** : auto-failover quand un provider tombe
- **ProviderPool** : resolution per-node, pas globale

## Configuration

Variables d'environnement (au moins une requise) :
- `DEEPSEEK_API_KEY`
- `GOOGLE_API_KEY`
- `OPENAI_API_KEY`
- `GROK_API_KEY`
- `KIMI_API_KEY`
- `MINIMAX_API_KEY`
- `OPEN_ROUTER_API_KEY`

## Modeles (cards.toml)

20 modeles configures dans `sage-core/config/cards.toml` avec :
- Scores d'affinite (s1, s2, s3)
- Scores de domaine (math, code, reasoning, etc.)
- Cout, latence, fenetre de contexte
