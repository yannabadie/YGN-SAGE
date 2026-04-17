---
title: Issues MOC
type: moc
tags:
  - issues
  - moc
updated: 2026-04-17
---

# Issues Connues

## P1 — Critiques

| Issue | Status | Detail |
|-------|--------|--------|
| [[Sandbox-Safety]] | Durci + structured argv (Apr 17) | CORAL 84fee02 ajoute allowlist argv, pas encore formel |
| [[Sage-Discover-Broken]] | **Resolu Apr 17** | 3 bugs ExoCortex fixes (`9b8d91c`, `2c994e0`, `60793b7`, `2780bbb`, `d40dc4e`). Rattrapage v3 active |

## P2 — Importantes

| Issue | Status | Detail |
|-------|--------|--------|
| [[Parallel-Axis-Regression]] | Ouvert | MASBENCH parallel -6pp |
| [[Robustness-Zero]] | Ouvert | 0% bare ET SAGE |
| [[Memory-Consolidation-Incomplete]] | Ouvert | Design ok, implementation partielle |
| [[Path6-Not-Default]] | Design choice | Opt-in, pas dans pipeline defaut |
| SWE-bench data contamination | **Decouvert Apr 17 PM** | smoke v3 `_tool_call_count=0` sur 3/3 — patches issus du recall LLM. Mitige par `tool_choice="required"` (`da839dc`) sur coder/actor steps 1-2 |
| Custom prompt registry | Discute, non implemente | Indexation (task_embed, role, template) — alternative cheaper proposee : injecter plan du planner dans system_prompt aval |

## Recemment fixes (Apr 17 PM — F7 + ExoCortex)

| Issue | Severite | Fix |
|-------|----------|-----|
| MiniMax 400 "tool id not found" | P0 | `591d3c4` — orphan-tool guard dans `phases/act.py` truncation. Set-based filtre ID surviving |
| F2 sentinel 25-char fake patches | P0 | `b19e744` (F1 max_steps scale), `a1ed919` (F2 salvage), `76bae64` (events as objects pas tuples) |
| F6 generic role prompts | P1 | `53e2aac` — per-role system prompts (planner/coder/synthesizer/verifier/source/worker) |
| F7 absent task tier promotion | P1 | `029701c` (base) + `2839d95` (domain floor) + `4efa37d` (sink audit) + `4c1b52a` (FrugalGPT) + `6d198db` (review apply). Voir [[ADR-007-F7-Routing]] |
| F8 Gemini 3.x temperature | P2 | `081812d` — force temp=1.0 (degenerate au-dessous) |
| CEGAR thrash sur code S3 | P0 | `97dfb2b` — PRM domain gate, Z3 uniquement math/formal. Voir [[ADR-008-PRM-Gate-Domain]] |
| ExoCortex pipeline if/else | P0 | `9b8d91c` — both backends (Qdrant + ExoCortex), sum semantics |
| ExoCortex upload polling no timeout | P0 | `2c994e0` (90s) → `60793b7` (300s) → `2780bbb` (600s) — bumped 3 fois selon evidence |
| ExoCortex manifest store_name vide | P2 | `2c994e0` — populate from live client on first paper |
| ExoCortex genai.Client recreation | P2 | `d40dc4e` — reuse cached `_get_client()` (~63s economises sur 211 uploads) |
| Test mock signatures F7 | P3 | `ae9d10b` — `_MockAssigner` heritage signature dans `test_oxiz_pipeline.py` |

## Recemment fixes (Apr 17 AM — Sprints autonomes)

| Issue | Severite | Fix |
|-------|----------|-----|
| SWE-bench 3 diagnostic gaps | P1 | `bb70502` — system_hint=3 forwarding, prompt fusion, multi-turn mandate |
| ToolForge non wire E2E | P1 | `93f911d` — execute_tool_call ouvre CreationTicket + retry transparent |
| Sage recursion absent | P2 | `13463fb` — sage_recurse tool, depth via contextvars, SAGE_RECURSION_MAX |
| SWE-bench Pro support | P2 | `3d898a3` — `pro → ScaleAI/SWE-bench_Pro` mapping + 4-config ablation runner |
| Decision gate framework | P2 | `2878265` — `decide_next_phase.py` lit ablation JSON, gate A/B/C |

## Recemment fixes (Apr 9-15 — Architecture)

| Issue | Severite | Fix |
|-------|----------|-----|
| Legacy execution path | P1 | `f97ddc4` — _run_legacy + SAGE_AGENT_LOOP_LEGACY supprimes (-550 LOC) |
| Topology nodes pas vrais agents | P1 | `786838a`, `e8401dd`, `f265dba` — agent_loop factory + TopologyRunner dispatch (Phases 1-3) |
| Training pipeline en main | P1 | `b2f59ee` — supprime verl/scripts/data/models (-4.3GB), branche dediee |
| ComplexityRouter dead | P2 | `00097d8` (CORAL) — supprime ComplexityRouter (34% GT) + ShadowRouter (-1794 LOC) |
| kNN exact-match miss | P2 | `ff41e53` (CORAL) — exact-match override → 100% GT sur 60 exemplaires |
| S2+sequential bypass trop agressif | P2 | `30ee004` (CORAL) — supprime, opt-in via SAGE_BYPASS_S2_SEQUENTIAL=1 |
| TopologyController inter-node | P1 | `47784c7`, `d73cfa2` (CORAL) — wire VPRMs (arXiv 2601.17223) |

## Recemment fixes (Avril 8, 2026 — Architecture)

| Issue | Severite | Fix |
|-------|----------|-----|
| Pipeline tool-calling | P0 | Single entry point, agents utilisent execute_bash via tool-calling loop |
| ToolForge non wire | P1 | Wire dans agent loop au boot |
| execute_bash manquant | P0 | Outil enregistre au boot (git bash sur Windows) |
| Model selection hardcode | P1 | Rust ModelRegistry.select_for_system() affinity scoring |
| Provider health check | P0 | Probe au boot, circuit breaker pour providers morts |
| Rust exclude_providers | P1 | ModelAssigner exclut providers morts a la source |
| truststore SSL | P1 | Corporate proxy fixe via Windows Certificate Store |
| json_schema DeepSeek | P2 | Seulement envoye a OpenAI |
| FrugalGPT model mismatch | P2 | Valide provider avant upgrade |
| SWE-bench repo clone | P1 | Clone au bon commit + chdir avant execution |
| Token usage capture | P2 | LLMResponse.usage rempli depuis les reponses API |
| EvolutionMemory | P1 | CORAL Phase 1, SQLite WAL, lazy init au boot |

## Recemment fixes (Avril 7, 2026 — Phase A)

| Issue | Severite | Fix |
|-------|----------|-----|
| A2A phantom v1.0 imports | P1 | Migre vers a2a-sdk 0.3.25 (10 imports), 6 tests passent |
| ToolForge "Tool not callable" | P1 | `Tool.run()` ajoute, forge.py corrige, 4 tests E2E |
| Codex CLI removal fallout | P1 | 3 tests attendaient provider="codex" → "openai" |
| SSL bypass dans tests | P2 | verify=False et CA_BUNDLE="" supprimes (directive #3) |
| Env leaks GOOGLE_API_KEY | P2 | os.environ → monkeypatch dans 2 fichiers |
| boot() rename incomplet | P2 | boot() → boot_agent_system() dans provider_pool_wiring |
| _provider_pool rename | P2 | _provider_pool → provider_pool (attribut public) |
| BigCodeBench logging muet | P2 | basicConfig ajoute dans bench CLI |

## Fixes anterieurs (Avril 3, 2026)

| Issue | Severite | Fix |
|-------|----------|-----|
| Sandbox bash blocklist | P1 | Regex validator ajoute |
| EpisodicMemory race condition | P1 | Async queue supprimee, SQLite direct |
| Pipeline mock signatures | P1 | Params task_embedding, hints, max_cost_usd |
| veRL reward V8 scoring | P1 | Format tool_call mis a jour |
| kNN exemplar count 50→60 | P2 | Tests mis a jour |
| sage-discover ebpf crash | P1 | References mortes supprimees |
| UI dashboard race condition | P1 | run_lock ajoute |
| a2a_server test skips | P2 | Completement fixe (Phase A, pas juste skip) |
