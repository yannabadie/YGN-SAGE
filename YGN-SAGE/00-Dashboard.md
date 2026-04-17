---
title: YGN-SAGE Dashboard
type: moc
updated: 2026-04-17
---

# YGN-SAGE — Self-Adaptive Generation Engine

Agent Development Kit qui **apprend** quelle topologie multi-agent utiliser pour chaque tache.
Rust core (sage-core) + Python SDK (sage-python) + Knowledge Pipeline (sage-discover).

## Etat du projet (17 Avril 2026)

| Metrique | Valeur | Notes |
|----------|--------|-------|
| Tests Python | **1896/1896** | 0 failures (training tests retires Apr 15, b2f59ee) |
| Tests Rust | **441/441** | +12 vs Apr 7 (F7 floor + sink + drift tests) |
| kNN Routing GT | **100%** (60/60) | exact-match override depuis Apr 17 (CORAL ff41e53) |
| MASBENCH breadth | **+22pp (p=0.015)** | Seul axe statistiquement significatif |
| MASBENCH depth/horizon | +2/+4pp | Non significatifs (p>0.05) |
| MASBENCH parallel | **-6pp** | Regression — topologie nuit |
| BigCodeBench Hard | **v4 final: 45.9%** (68/148) | bypass + repair reasoner + escalation |
| Adaptive bypass | **Removed Apr 17** (CORAL 30ee004) | topology-first par defaut, opt-out via SAGE_BYPASS_S2_SEQUENTIAL=1 |
| EvolutionMemory | CORAL **integrated** | SQLite WAL + persistent skills, lazy init au boot |
| Providers | **7 alive** + ModelAssigner.exclude_providers() | Health check au boot, circuit breaker |
| Tools agent | **14** (+sage_recurse Apr 17) | Tool-calling loop dans pipeline (single entry point) |
| Model selection | Rust affinity scoring + **F7 floor** | ModelAssigner forwarde `task_system` pour role-aware promotion (voir [[ADR-007-F7-Routing]]) |
| Templates topo | 12 | +`formal_solver` (Apr 17), 6 sink roles maintenus en sync |
| PyPI | `pip install ygn-sage` | v0.1.0-alpha |
| A2A | a2a-sdk 0.3.25 | 6 tests, streaming + cancellation |
| ToolForge | Wire dans agent loop + **E2E** (Sprint 3) | execute_tool_call ouvre CreationTicket et retry |
| SWE-bench Lite | smoke 2/3 patches (post F7+PRM-gate) | gating sur Docker eval (en attente) |
| Architecture | **Unified entry point Phases 1-3 mergees** | system.run() → pipeline.run() unique path (-944 LOC) |
| Training | **Parked Apr 15** | code retiré (-4.3GB), checkpoints sur HF, branche dédiée |
| ExoCortex | **3 bugs fixes Apr 17** | rattrapage v3 active (manifest 3 → 39+ papers) |
| F7 routing | **Apr 17 PM** — domain-aware floor | math/formal S3 → S3, autres S3 → S2 (sink classification audit-protected) |
| PRM gate | **Apr 17 PM** — domain-symetrique | Z3 PRM uniquement sur math/formal, voir [[ADR-008-PRM-Gate-Domain]] |
| tool_choice | **Apr 17 PM** — `required` sur coder/actor steps 1-2 | Force F6 mandate empirique (sinon ignored) |

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
> - **MiniMax** : ACTIF dans le pipeline (Apr 17, contrairement à ce qui était documenté). Bug 400 orphan-tool fixe `591d3c4`
> - **Path 6 (learned policy)** : opt-in (`SAGE_ENABLE_PATH6=1`), pas dans le pipeline par defaut
> - **sage-discover ExoCortex** : 3 bugs latents fixes Apr 17 (if/else, upload timeout, manifest persistence). Rattrapage v3 active
> - **Memory consolidation** : design documente, implementation incomplete
> - **Sandbox meta-tools** : durci (regex) + structured argv allowlist (CORAL 84fee02), pas formellement sur
> - **Benchmarks** : leaderboard BigCodeBench gele avril 2025, comparaison biaisee
> - **SWE-bench Lite contamination** : smoke v3 a `_tool_call_count=0` sur 3/3 tasks → patches "réussis" sont du recall LLM, pas de l'investigation. tool_choice="required" Apr 17 PM force l'usage des outils
> - **Z3 PRM** : applicable uniquement à math/formal (voir [[ADR-008-PRM-Gate-Domain]]). Sur code l'AVR niveau 2 suffit

> [!info] Architecture vs Realite
> ~85% de l'architecture documentee est implementee et integree (Apr 17).
> Les 15% restants : evolution (opt-in), consolidation memoire, preuves formelles sandbox, learned prompt registry (discuté, pas implémenté).

> [!success] Recemment fixe (Apr 9 → 17, 50+ commits, voir [[Changelog-Apr9-17]])
> - **Unified entry point Phases 1-3 mergees** (Apr 9-10) — single execution path, -944 LOC
> - **Training pipeline retire** (Apr 15, b2f59ee) — code sur branche dediee, checkpoints HF
> - **Sprints 1-6 autonomes** (Apr 17 AM) — Sprint 5 ablation scaffolding, Sprint 6 decision gate
> - **CORAL integration** (cherry-picks Apr 17) — kNN exact-match (100% GT), TopologyController, S2+sequential bypass removed
> - **F7 routing** (Apr 17 PM) — role-aware tier promotion, sink audit (-formal_solver regression), FrugalGPT cascade wiring
> - **PRM domain gate** (Apr 17 PM) — Z3 PRM uniquement sur math/formal (smoke v3 : 0 → 17 CEGAR, 2/3 patches debloques)
> - **tool_choice required** (Apr 17 PM) — force F6 mandate sur coder/actor steps 1-2
> - **ExoCortex bugs** (Apr 17 PM) — 3 bugs reels + 1 perf, rattrapage v3 active
> - **Obsidian vault** : 48 → 50+ fichiers, 8 ADRs, [[Changelog-Apr9-17]]
