---
title: Provider Architecture
type: architecture
tags:
  - architecture
  - providers
updated: 2026-05-10
---

# 7 Providers - Architecture Multi-Fournisseur

Source de verite modele : `sage-core/config/cards.toml`

Source de verite connexion : `sage-python/src/sage/providers/connector.py`

Etat courant detaille : `docs/status/2026-05-10-current-state.md`

## Providers

| Provider | API URL | Modele par defaut | Role |
|----------|---------|-------------------|------|
| DeepSeek | api.deepseek.com/v1 | deepseek-v4-flash | Budget primaire; thinking disabled |
| Google | native SDK | gemini-3.1-flash-lite-preview | Fallback fiable |
| OpenAI | api.openai.com/v1 | gpt-5.4 | Meilleure qualite |
| xAI | api.x.ai/v1 | grok-4-1-fast-reasoning | Raisonnement rapide |
| Kimi | api.moonshot.ai/v1 | kimi-k2.6 | Vision + raisonnement |
| MiniMax | api.minimax.io/v1 | MiniMax-M2.7 | Contexte texte 204.8k |
| OpenRouter | openrouter.ai/api/v1 | qwen/qwen3.5-plus-02-15 | Acces 200+ modeles |

## Etat verifie 2026-05-10

- `cards.toml` contient 24 model cards et 7 API providers.
- DeepSeek expose actuellement `deepseek-v4-flash` et `deepseek-v4-pro`.
- `deepseek-chat` et `deepseek-reasoner` sont des aliases legacy non
  selectionnables au runtime; ils sont remappes vers `deepseek-v4-flash`.
- MiniMax utilise l'orthographe officielle `MiniMax-M2.7` et un contexte texte
  204.8k; `models.list` MiniMax renvoie 0 ID mais le smoke generation live
  fonctionne.
- L'ancien `minimax-m2.7` reste dans `cards.toml` comme alias de compatibilite
  non selectionnable et remappe vers `MiniMax-M2.7`.
- Smoke provider live: 10/10 OK dans
  `docs/benchmarks/2026-05-10-provider-preflight-post-model-catalog.json`.
  Chaque ligne est `evidence_scope=liveness_only`. Cela prouve la
  connectivite/configuration, pas la qualite benchmark ni le respect strict
  d'instruction.

## Fonctionnement

- Chaque noeud de topologie peut utiliser un provider different
- La policy model peut exprimer `provider_hint` pour biaiser la selection (+0.15)
- **Per-model routing réel** (Apr 18, c9ff902) : le provider honore `config.model`; avant, l'adapter default ignorait silencieusement les décisions du `ModelAssigner` depuis `cards.toml`
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

24 modeles configures dans `sage-core/config/cards.toml` avec :
- Scores d'affinite (s1, s2, s3)
- Scores de domaine (math, code, reasoning, etc.)
- Cout, latence, fenetre de contexte
- Selectabilite runtime et remplacements d'aliases legacy
