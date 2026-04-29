---
title: ADR-014 Runtime Contracts (Cycle 1, R0-R4)
type: adr
status: shipped
date: 2026-04-28
commits: ["b66b07bd", "ebba1d56", "2d346f96", "42234c4d"]
tags: [runtime, topology, runner, sandbox, p0]
---

# ADR-014 — Cycle 1 Runtime Contracts (R0..R4)

## Context

Pre-cycle-1 main branch had four latent runtime invariants violated in `TopologyRunner` and `_execute_code_node`. These were verified live by reading `runner.py` HEAD `e73990ae`, not relying on cgpro's CGPRO.md snapshot:

1. **Controller mutation duplication.** `_execute_node_via_agent_loop:625-647` called `evaluate_and_decide` "for debug log", then `run()`/`run_traced()`/`run_stream()` called it again to apply. `topology_controller.py:206-347` mutates Rust state on every invocation (abstain_count, reroute_count, retry counters, `_gate_loops`). Net: counters bumped 0→2 per logical node, biasing routing decisions and triggering false reroutes/prunes.
2. **Three-path execution divergence.** `run` handled 5 controller actions, `run_traced` handled only `upgrade_model` (silently dropped 4/5), `run_stream` handled 3/5 in single-node and **skipped controller entirely in parallel branch**. Plus `run_traced` ran parallel batches sequentially via `for` loop (no `asyncio.gather`) — performance regression on parallel topologies.
3. **Capability-blind provider fallback.** `runner.py:1043-1054` used `get_available_providers()[0]` on primary failure — picked the first available provider with NO capability check. A node with `required_capabilities=["tools"]` could be assigned a `supports_tools=False` model. This was the kimi-k2.5/k2.6 incident class (roadmap-A2). Plus a stale `if fallback_cfg.get("sdk") != "google-genai"` heuristic.
4. **Sandbox fail-open on non-Linux.** `_execute_code_node` fell back to raw `subprocess.run` on `from sage.sandbox.isolated_executor import execute_isolated` ImportError. cgpro added: even when the import succeeds, `BWRAP_AVAILABLE = platform.system() == "Linux" and shutil.which("bwrap") is not None` — on Windows or Linux without bwrap, `execute_isolated` internally falls back to plain subprocess. So gating only on ImportError leaves a fail-open hole on every non-Linux platform.

## Decision

Sequenced 4 P0 fixes as Cycle 1, each with a separate cgpro DESIGN → codex IMPLEMENT → Claude verify-local → cgpro VERIFY → SHIP cycle.

### R1 — Controller single-commit (`b66b07bd`, 2 files +139/-26)

Pure removal of the debug-eval block at `runner.py:624-647`. No `preview()` API (would need state snapshot/restore or duplicated read-only logic — both wider than this P0). The real `run/run_traced/run_stream` decision sites already log their actions. New `tests/test_topology_runner_controller_once.py` with `CountingController` mock pinning ≤1 call per node.

### R2 — Unify `_run_core` (`ebba1d56`, 2 files +624/-243)

Replaced 3 divergent execution loops with one private async-generator `_run_core()` yielding 6 typed events (`_NodeStartEvent`, `_NodeDoneEvent`, `_ControllerDecisionEvent`, `_RerouteEvent`, `_BudgetExceededEvent`, `_TopologyDoneEvent`). Shared `_apply_controller_decision()` helper handles all 5 actions in ONE place (upgrade_model / spawn_subagent / reroute_topology / prune_node / open_gate). `run/run_traced/run_stream` reduced to thin wrappers (~10-25 lines each). NodeStarted/NodeCompleted emitted in **executor ready order**, not wall-clock completion (deterministic for tests). Public API shapes unchanged: `run` → str, `run_traced` → list[dict] with legacy `latency` in seconds, `run_stream` → AsyncIterator[dict] with `type` discriminator.

### R3 — Capability-aware fallback (`2d346f96`, 3 files +614/-44)

`TopologyRunner.__init__` extended with `assigner` + `task_domain` + `budget_usd` kwargs. `_remaining_budget_usd()` helper returns `min(cost_tracker.remaining, budget_usd - total_cost_usd)` or `float("inf")` (never 0.0 — 0.0 is fail-closed signal that rejects all paid models, not "unlimited"). `_capability_aware_fallback_generate()` retry-loop calls `assigner.assign_single_node(graph, node_idx, task_domain=..., budget_usd=remaining, exclude_model_ids=[failed], task_system=node.system)` + validates via `provider_pool.is_model_available(model_id)` (TTL'd circuit breaker) + retries with dead model excluded + restores original_model_id on exhaustion. cgpro VERIFY round-trip caught 2 micro-fixes: (1) cost_tracker fall-through bug at exhaustion, (2) `if provider is not self._llm` guard preserved old fail-open. Both fixed before SHIP. Connector + OpenAICompatProvider hand-roll deleted.

### R4 — Sandbox fail-closed (`42234c4d`, 6 files +376/-20)

NEW typed `SandboxUnavailable(RuntimeError)` in `sage/sandbox/errors.py` + lazy `__init__.py` refactor (errors always-importable even when isolated_executor unavailable). `_execute_code_node` gates on BOTH ImportError AND `BWRAP_AVAILABLE=False`. Default → raise `SandboxUnavailable`. `SAGE_UNSAFE_RAW_EXEC=1` → fall back to raw subprocess + WARNING "DO NOT USE IN PRODUCTION". Disjoint from `SAGE_UNSAFE_TOOLFORGE_SUBPROCESS` (ToolForge Gate 2 escape hatch). NEW `docs/security/sandbox-policy.md` documents the three sandbox surfaces.

## Methodology validation

Pattern: cgpro DESIGN (locked spec) → codex IMPLEMENT (gpt-5.5 xhigh full-auto) → Claude verify-local (TDD via `git stash --keep-index` for pre-fix evidence) → cgpro VERIFY → SHIP.

This was the first time the methodology ran end-to-end across 4 sequential P0 tickets. Validated:
- Schema-first DESIGN catches scope drift before codex writes code.
- Codex never commits (correctly per spec); Claude-side verify+commit is the right pattern.
- cgpro VERIFY round-trips catch contract leaks even when verify-local tests are green (R3 caught 2 leaks).
- TDD via `git stash` of source-only changes (keeping new test files staged) gives clean pre-fix failure evidence.

## Consequences

- 21 new regression tests pinning the 4 invariants.
- All 4 are LATENT bugs: tests passed, mypy clean, ruff clean — but production behavior was wrong on Windows + on parallel topologies + on multi-action controller decisions + on capability-mismatched fallback.
- 5 cgpro VERIFY round-trip findings became commit-message-documented contract evidence.
- Pattern reused in cycles 2-5 (R5 RuntimeEventLog, R6 StateCore, R7 RunFrame, R9 OracleStack).
- All four merge as separate commits per "no two P0 in same patch" discipline.

## Related

- [[ADR-015-RuntimeEventLog-Cycle2]] — R5, builds on R2 `_run_core` as event injection point
- [[ADR-016-StateCore-Cycle3]] — R6, builds on R5 trace + R2 unification
- [[ADR-017-RunFrame-Cycle4]] — R7, typed per-run object spine
- [[ADR-018-OracleStack-Cycle5]] — R9, training gate using R7 evidence surface
- [[ADR-013-Wasm-Sandbox-Default]] — prior sandbox decision; R4 builds on this
- `~/.claude/plans/whimsical-launching-seahorse.md` — cycle 1 plan
- `.tmp/archive/cycle1/` — locked specs + verify rounds for R1-R4
