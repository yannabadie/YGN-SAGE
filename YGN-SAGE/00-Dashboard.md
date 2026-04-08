---
title: YGN-SAGE Dashboard
type: moc
updated: 2026-04-07
---

# YGN-SAGE — Self-Adaptive Generation Engine

Agent Development Kit qui **apprend** quelle topologie multi-agent utiliser pour chaque tache.
Rust core (sage-core) + Python SDK (sage-python) + Knowledge Pipeline (sage-discover).

## Etat du projet (8 Avril 2026)

| Metrique | Valeur | Notes |
|----------|--------|-------|
| Tests Python | **2001/2001** | 0 failures (Phase A fix, commit 23ab78b) |
| Tests Rust | **429/429** | 100% pass |
| kNN Routing GT | **93.3%** (56/60) | Routeur principal, 60 exemplaires |
| MASBENCH breadth | **+22pp (p=0.015)** | Seul axe statistiquement significatif |
| MASBENCH depth/horizon | +2/+4pp | Non significatifs (p>0.05) |
| MASBENCH parallel | **-6pp** | Regression — topologie nuit |
| BigCodeBench Hard | v1: 37.2%, v3b: 35.8%, **v4 en cours** | v4: bypass + repair + escalation |
| Adaptive bypass | S2+sequential → single-agent | AdaptOrch omega thresholds |
| EvolutionMemory | CORAL Phase 1 implemente | SQLite WAL, skills persistantes |
| Providers | **7 alive** (truststore SSL) | Health check au boot, circuit breaker, ModelAssigner.exclude_providers() |
| Tools agent | **13** (execute_bash + meta + memory) | Tool-calling loop dans pipeline (single entry point) |
| Model selection | Rust affinity scoring | ModelRegistry.select_for_system() depuis cards.toml, pas hardcode |
| Templates topo | 11 | sequential, parallel, AVR, debate, hub, etc. |
| PyPI | `pip install ygn-sage` | v0.1.0-alpha |
| A2A | a2a-sdk 0.3.25 | 6 tests, streaming + cancellation |
| ToolForge | Wire dans agent loop | GapDetector + BuildLoop + tool-calling integre |
| SWE-bench | Premier patch genere | Agent lit le code via execute_bash, clone repo |

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
> - **MASBENCH 4/5 axes non significatifs** : seul breadth (p=0.015) prouve topology > model
> - **MASBENCH parallel -6pp** : la topologie regresse sur les taches paralleles
> - **MASBENCH robustness 0%** : 0% bare ET SAGE — a debugger
> - **BigCodeBench omega=1.3** : topologie n'est PAS le levier (ADR-006). Gains viennent du repair
> - **MiniMax** : exclu au boot via health check + ModelAssigner.exclude_providers() (SSL proxy fixe par truststore)
> - **Path 6 (learned policy)** : opt-in (`SAGE_ENABLE_PATH6=1`), pas dans le pipeline par defaut
> - **sage-discover** : gateway partiellement cassee (imports fixes, runtime non verifie)
> - **Memory consolidation** : design documente, implementation incomplete
> - **Sandbox meta-tools** : durci (regex) mais pas formellement sur — `subprocess` avec blocklist
> - **Benchmarks** : leaderboard BigCodeBench gele avril 2025, comparaison biaisee

> [!info] Architecture vs Realite
> ~80% de l'architecture documentee est implementee et integree.
> Les 20% restants : evolution (opt-in), consolidation memoire, sage-discover, preuves formelles sandbox.

> [!success] Recemment fixe (7-8 avril 2026, 23 commits)
> - **Phase A** : A2A v0.3.25, ToolForge E2E, 17 test fixes (2001 pass/0 fail)
> - **Pipeline tracing** : routing, topology, models, cost par tache dans JSONL
> - **MiniMax pre-filtre** : providers sans API key exclus avant assignment
> - **AVR repair** : reasoner tier (gemini-3.1-pro) au lieu du meme modele
> - **Adaptive bypass** : S2+sequential → single-agent (AdaptOrch omega thresholds)
> - **Topology escalation** : bypass → repair → topology fallback (Conductor-inspired)
> - **EvolutionMemory** : CORAL Phase 1 — SQLite WAL persistent mutations/skills
> - **MASBENCH stats** : McNemar + Cohen's d sur N=50 existant — breadth p=0.015
> - **Obsidian vault** : 48 fichiers trackes, 20+ corrections, 6 ADRs
