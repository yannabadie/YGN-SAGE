---
title: YGN-SAGE Dashboard
type: moc
updated: 2026-04-18
---

# YGN-SAGE — Self-Adaptive Generation Engine

Agent Development Kit qui **apprend** quelle topologie multi-agent utiliser pour chaque tache.
Rust core (sage-core) + Python SDK (sage-python) + Knowledge Pipeline (sage-discover).

## Etat du projet (18 Avril 2026)

| Metrique | Valeur | Notes |
|----------|--------|-------|
| Tests Python | **1906/1906** | +10 Apr 18 (telemetry, per-model routing, health TTL) |
| Tests Rust | **441/441** | Stable depuis Apr 17 |
| kNN Routing GT | **100%** (60/60) | exact-match override (CORAL ff41e53) |
| MASBENCH breadth | **+22pp (p=0.015)** | Seul axe statistiquement significatif |
| MASBENCH depth/horizon | +2/+4pp | Non significatifs (p>0.05) |
| MASBENCH parallel | **-6pp** | Regression — topologie nuit |
| BigCodeBench Hard | **v4 final: 45.9%** (68/148) | bypass + repair reasoner + escalation |
| **SWE-bench Lite smoke** | **v5d 4/5 (80%) v5e 3/5 — moyenne 70%** | (vs v4 1/10=10%, Apr 17). Vrais fixes multi-fichiers avec 31-62 tool_calls/task |
| EvolutionMemory | CORAL **integrated** | SQLite WAL + persistent skills, lazy init au boot |
| Providers | **7 alive + TTL exclusion Apr 18** | Re-probe 300s, recover automatiquement (OpenAI quota/Gemini brown-out) |
| Tools agent | **14** (+sage_recurse) | Tool-calling loop — **62 bash/task confirmé Apr 18** |
| Model selection | Rust affinity + **F7 floor** + **per-model routing réel Apr 18** | LiteLLMProvider honore config.model (était silencieusement ignoré depuis cards.toml) |
| Templates topo | 12 | +`formal_solver` (Apr 17), 6 sink roles maintenus en sync |
| PyPI | `pip install ygn-sage` | v0.1.0-alpha |
| A2A | a2a-sdk 0.3.25 | 6 tests, streaming + cancellation |
| ToolForge | Wire dans agent loop + **E2E** | execute_tool_call ouvre CreationTicket et retry |
| **Telemetry tool_call_count** | **Réel Apr 18** | compteur mort depuis Apr 9, re-wired bypass + topology aggregation |
| **Bench reporter** | **Classifier real/sentinel/empty Apr 18** | header ne peut plus mentir (sentinels étaient comptés comme patches) |
| **Sentinel cascade** | **Résolue Apr 18** (85282e0) | `EMPTY_STEP_SENTINEL` filtré avant d'être transmis en aval |
| Architecture | **Unified entry point Phases 1-3 mergees** | system.run() → pipeline.run() unique path (-944 LOC) |
| Training | **Parked Apr 15** | code retiré (-4.3GB), checkpoints sur HF, branche dédiée |
| ExoCortex | **3 bugs fixes Apr 17** | rattrapage terminé (manifest 132 papers, 2026-04-18) |
| F7 routing | **Apr 17 PM** — domain-aware floor | math/formal S3 → S3, autres S3 → S2 (sink classification audit-protected) |
| PRM gate | **Apr 17 PM** — domain-symetrique | Z3 PRM uniquement sur math/formal, voir [[ADR-008-PRM-Gate-Domain]] |
| tool_choice="required" | **Reverted Apr 18** (e69cb7f) | causait sentinels cascades ; diagnostic basé sur compteur mort |

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
> - **`_tool_call_count=0` interprétation mensongère (résolu Apr 18)** : le compteur était mort (champ `PipelineContext` déclaré jamais incrémenté). v3/v4/v5a conclusions "modèles refusent outils" fausses. Avec telemetry réel (988aa99 + 0677376) : 31-62 bash/task, vraie investigation multi-fichiers
> - **Z3 PRM** : applicable uniquement à math/formal (voir [[ADR-008-PRM-Gate-Domain]]). Sur code l'AVR niveau 2 suffit
> - **Budget steps S3=20** : encore trop tight quand planner consomme 20+ tool_calls → sentinel. Prochaine direction : dynamic scaling par plateau detection

> [!info] Architecture vs Realite
> ~90% de l'architecture documentee est implementee et integree (Apr 18).
> Les 10% restants : evolution (opt-in), consolidation memoire, preuves formelles sandbox, learned prompt registry (discuté, pas implémenté), dynamic step budget.

> [!success] Recemment fixe (Apr 18, **13 commits plumbing** + 5 commits Apr 17)
> - **Bench reporter classifier** (4a33faa) — real/sentinel/empty, le header cesse de mentir
> - **Sentinel cascade strip** (85282e0) — `EMPTY_STEP_SENTINEL` filtré en predecessor context
> - **Telemetry re-wiring** (988aa99 bypass + 0677376 topology) — `_tool_call_count` cesse d'être mort
> - **Per-model routing réel** (c9ff902) — LiteLLMProvider honore `config.model` (était ignoré)
> - **Provider inference intelligente** (4a2c038 + f754535) — "unknown" → inférence par model_id ; qwen/* → openrouter
> - **Quota-aware + TTL health_check** (fe66d52 + 3148667) — 429 insufficient_quota = DEAD ; **re-probe 300s pour recovery auto** (pas permanent)
> - **`--offset` CLI** (97fc64f) — mesure hygiénique hors tâches mémorisées
> - **Planner injection opt-in** (ea09dd6) — `SAGE_PLANNER_INJECTION=1`, MASS arXiv 2502.02533
> - **tool_choice required reverted** (e69cb7f) — diagnostic basé sur compteur mort, causait sentinels
> - **Résultats v5d/v5e** : 4/5 puis 3/5 real patches (moyenne 70%), 0 sentinels, 62 bash/task (vs v4: 1/10=10%)

> [!success] Recemment fixe (Apr 9 → 17, 50+ commits, voir [[Changelog-Apr9-17]])
> - **Unified entry point Phases 1-3 mergees** (Apr 9-10) — single execution path, -944 LOC
> - **Training pipeline retire** (Apr 15, b2f59ee) — code sur branche dediee, checkpoints HF
> - **Sprints 1-6 autonomes** (Apr 17 AM) — Sprint 5 ablation scaffolding, Sprint 6 decision gate
> - **CORAL integration** (cherry-picks Apr 17) — kNN exact-match (100% GT), TopologyController, S2+sequential bypass removed
> - **F7 routing** (Apr 17 PM) — role-aware tier promotion, sink audit (-formal_solver regression), FrugalGPT cascade wiring
> - **PRM domain gate** (Apr 17 PM) — Z3 PRM uniquement sur math/formal (smoke v3 : 0 → 17 CEGAR, 2/3 patches debloques)
> - **ExoCortex bugs** (Apr 17 PM) — 3 bugs reels + 1 perf, rattrapage terminé (manifest 132 papers)
