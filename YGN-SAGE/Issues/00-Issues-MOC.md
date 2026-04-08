---
title: Issues MOC
type: moc
tags:
  - issues
  - moc
updated: 2026-04-07
---

# Issues Connues

## P1 — Critiques

| Issue | Status | Detail |
|-------|--------|--------|
| [[Sandbox-Safety]] | Durci, pas resolu | Regex blocklist, pas formel |
| [[Sage-Discover-Broken]] | Partiellement fixe | Imports fixes, runtime non verifie |

## P2 — Importantes

| Issue | Status | Detail |
|-------|--------|--------|
| [[Parallel-Axis-Regression]] | Ouvert | MASBENCH parallel -6pp |
| [[Robustness-Zero]] | Ouvert | 0% bare ET SAGE |
| [[Memory-Consolidation-Incomplete]] | Ouvert | Design ok, implementation partielle |
| [[Path6-Not-Default]] | Design choice | Opt-in, pas dans pipeline defaut |
| MiniMax 400 error | **Mitige** | SSL proxy → truststore, exclu via health check + ModelAssigner.exclude_providers() |

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
