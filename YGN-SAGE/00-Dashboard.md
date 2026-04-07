---
title: YGN-SAGE Dashboard
type: moc
updated: 2026-04-07
---

# YGN-SAGE — Self-Adaptive Generation Engine

Agent Development Kit qui **apprend** quelle topologie multi-agent utiliser pour chaque tache.
Rust core (sage-core) + Python SDK (sage-python) + Knowledge Pipeline (sage-discover).

## Etat du projet (7 Avril 2026)

| Metrique | Valeur | Notes |
|----------|--------|-------|
| Tests Python | **2001/2001** | 0 failures (Phase A fix, commit 23ab78b) |
| Tests Rust | **429/429** | 100% pass |
| kNN Routing GT | **93.3%** (56/60) | Routeur principal, 60 exemplaires |
| MASBENCH delta | +27pp (non-pondere) | breadth +22, depth +2, horizon +4, parallel **-6** |
| BigCodeBench Hard | 37.8% (verification en cours) | Leaderboard gele (33.1%) |
| Providers | 7 | DeepSeek, Google, OpenAI, xAI, Kimi, MiniMax, OpenRouter |
| Templates topo | 11 | sequential, parallel, AVR, debate, hub, etc. |
| Modeles entraines | 2 | Qwen3-4B local, Nemotron-8B pod |
| PyPI | `pip install ygn-sage` | v0.1.0-alpha |
| A2A | a2a-sdk 0.3.25 | 6 tests, streaming + cancellation |
| ToolForge | E2E valide | gap -> synthese -> registration -> use |

## Navigation

### Architecture
- [[00-Architecture-MOC|Architecture]] — 5 piliers cognitifs, pipeline 6 etapes
- [[Pipeline|Pipeline CLASSIFY-LEARN]] — Le flux complet d'une tache
- [[Provider-Architecture|Providers]] — 7 providers, circuit breaker, failover

### Recherche
- [[00-Papers-MOC|Papers]] — 25+ papers de recherche backing le projet
- [[00-Training-MOC|Training]] — Qwen3-4B local + Nemotron-8B pod

### Resultats
- [[00-Benchmarks-MOC|Benchmarks]] — MASBENCH, BigCodeBench, Routing GT
- [[00-Issues-MOC|Issues connues]] — Problemes ouverts et fixes
- [[00-Decisions-MOC|Decisions]] — ADRs et choix architecturaux

## Verites inconfortables

> [!warning] Ce qui ne marche pas ou est incomplet
> - **MASBENCH parallel -6pp** : la topologie regresse sur les taches paralleles
> - **MASBENCH robustness 0%** : 0% bare ET SAGE — a debugger
> - **MiniMax 400 error** : "invalid chat setting (2013)" — provider inutilisable, a debugger
> - **Path 6 (learned policy)** : opt-in (`SAGE_ENABLE_PATH6=1`), pas dans le pipeline par defaut
> - **sage-discover** : gateway partiellement cassee (imports fixes, runtime non verifie)
> - **Memory consolidation** : design documente, implementation incomplete
> - **Sandbox meta-tools** : durci (regex) mais pas formellement sur — `subprocess` avec blocklist
> - **Evolution** : opt-in, pas active par defaut dans `system.run()`
> - **Benchmarks** : leaderboard BigCodeBench gele avril 2025, comparaison biaisee

> [!info] Architecture vs Realite
> ~80% de l'architecture documentee est implementee et integree.
> Les 20% restants : evolution (opt-in), consolidation memoire, sage-discover, preuves formelles sandbox.

> [!success] Recemment fixe (7 avril 2026, Phase A)
> - **A2A** : imports migres de phantom v1.0 vers a2a-sdk 0.3.25 — 6 tests passent
> - **ToolForge** : bug "Tool not callable" corrige, `Tool.run()` ajoute, 4 tests E2E
> - **Codex CLI** : supprime — tous les modeles via API (DeepSeek primaire)
> - **17 tests casses** : codex provider assertions, SSL bypass, env leaks, boot rename
> - **boot()** renomme `boot_agent_system()` dans toute la codebase
