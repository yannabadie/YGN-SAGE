# SOTA 5-Axes Upgrade — Design Spec

**Date:** 2026-03-11
**Status:** Approved
**Scope:** 5 strategic improvements + 3 quick wins, all Rust-first

## Context

Post-Critique1 audit identified 5 SOTA gaps from ExoCortex research. All implementations target `sage-core/` (Rust) with minimal Python wiring. Goal: reduce Python surface, maximize Rust coverage.

## Axes

### Axe 1: CMA-ME for MAP-Elites
- **Module:** `sage-core/src/topology/cma_me.rs`
- **Why:** Random mutation (7 operators) lacks directional search. CMA-ME adapts covariance on continuous params (max_cost_usd, max_wall_time_s, edge weight).
- **Design:** `CmaEmitter` struct with mean vector + 3x3 covariance. 50% random / 50% CMA-sampled mutations in `evolve()`. Surrogate fitness from S-MMU similarity * quality.
- **Constraint:** No BLAS dependency. Pure Rust linear algebra (3x3 matrix ops inlined).

### Axe 2: Invariant Synthesis Loop (S3 CEGAR)
- **Module:** `sage-core/src/verification/mod.rs` (extend SmtVerifier)
- **Why:** S3 verification is one-shot. SOTA uses iterative generate-and-check with counterexample feedback.
- **Design:** `verify_invariant_with_feedback() -> SmtVerificationResult` returns structured diagnostics. `synthesize_invariant()` weakens/strengthens post-conditions over max 5 rounds. Python `kg_rlvr.py` wires feedback into S3 escalation prompt.
- **Constraint:** OxiZ has no model() extraction. Counterexamples are structural (which clause of post fails when pre holds).

### Axe 3: Learned Routing (KnowSelf/ALAMA-style)
- **Module:** `sage-core/src/routing/router.rs` (extend AdaptiveRouter)
- **Why:** Regex heuristic + shadow gate blocked at 1000 traces. Need to accelerate evidence collection and adapt thresholds online.
- **Design:** (a) Rust-native JSONL shadow trace writer, (b) 2-tier gate (soft at 500/10%, hard at 1000/5%), (c) `retrain_thresholds()` logistic regression on feedback buffer.
- **Constraint:** No Python dependency for trace collection.

### Axe 4: LTL Model Checking for Agent Plans
- **Module:** `sage-core/src/verification/ltl.rs` (new)
- **Why:** Zero temporal property verification. Policy.py only checks static constraints.
- **Design:** 4 checks on TopologyGraph via petgraph: reachability (BFS), safety (no HIGH→LOW paths), liveness (all entries reach exits), bounded liveness (depth ≤ K). `LtlVerifier` PyO3 class. Wired into `HybridVerifier`.
- **Constraint:** Finite DAG only (no infinite traces). All O(V+E).

### Axe 5: MCTS Topology Search
- **Module:** `sage-core/src/topology/mcts.rs` (new)
- **Why:** No tree search path. DynamicTopologyEngine has 5 paths, none systematic.
- **Design:** UCB1 tree search. Expansion via random mutation. Rollout via surrogate fitness (S-MMU). Budget: 50 simulations or 100ms. 6th path in `generate()`, activated when archive has ≥5 cells.
- **Constraint:** Must not block hot path (budget-limited). Pure Rust, no LLM calls.

### Quick Wins (integrated)
- **SQLite WAL:** PRAGMA in episodic.py, semantic.py, causal.py
- **Tracing:** `#[instrument]` on SmtVerifier + LtlVerifier methods
- **PyO3 opaque ID:** `generate()` returns topology_id + confidence, lazy-load via `get_topology()`

## Implementation Order

Axe 4 (LTL) → Axe 2 (invariants) → Axe 1 (CMA-ME) → Axe 5 (MCTS) → Axe 3 (routing) + quick wins throughout.

## Success Criteria

- All new Rust tests pass (`cargo test --features smt`)
- Python 1143+ tests still pass
- No new Python modules (all new logic in Rust)
- Each axe independently testable and committable
