---
title: Provider Architecture
type: architecture
tags:
  - architecture
  - providers
updated: 2026-04-18
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
| MiniMax | api.minimax.io/v1 | minimax-m2.7 | Contexte 4M tokens, SWE-bench coder (Apr 17+) |
| OpenRouter | openrouter.ai/api/v1 | qwen/qwen3.5-plus | Acces 200+ modeles — require `openrouter/` prefix (Apr 18 fix) |

## Fonctionnement

- Chaque noeud de topologie peut utiliser un provider different
- La policy model peut exprimer `provider_hint` pour biaiser la selection (+0.15)
- **Per-model routing réel** (Apr 18, c9ff902) : `LiteLLMProvider.generate()` honore `config.model` ; avant, `self.model_string` (adapter default) était utilisé, ignorant silencieusement les décisions du `ModelAssigner` depuis `cards.toml`
- **Provider inference** (Apr 18, 4a2c038 + f754535) : `_infer_provider_from_model_id()` reconnaît `gemini-*`/`gpt-*`/`deepseek-*`/`grok-*`/`minimax-*`/`kimi-*`/`x/y` → openrouter. `"unknown"` de ModelRegistry est remplacé par inférence.
- **Health check au boot** : probe tous les providers, circuit breaker pour les morts
- **Health check quota-aware** (Apr 18, fe66d52) : connexion error + 429 `insufficient_quota` → DEAD ; 401/400/429 transient sans quota wording → ALIVE (probe params misconfig, pas outage)
- **TTL exclusion + re-probe** (Apr 18, 3148667) : providers morts excluded 300s puis re-probés ; `ProviderPool.refresh_exclusion_list(assigner)` appelé au début de chaque batch (bench, pipeline) pour recovery automatique. **Exclusion n'est pas permanente.**
- **truststore** : SSL proxy corporate (*.adgroupe.com) gere via Windows Certificate Store
- **ModelAssigner.exclude_providers()** : providers morts exclus du scoring Rust (liste mise à jour dynamiquement)
- **FrugalGPT cascade** : valide provider avant upgrade modele
- **json_schema** : seulement pour OpenAI (DeepSeek/xAI/etc. le rejettent)

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
