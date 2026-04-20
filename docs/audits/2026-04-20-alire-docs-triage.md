# ALIRE.md / ALIRE2.md — Empirical Triage

**Date:** 2026-04-20
**Source docs:** `ALIRE.md` (Path 6c proposal, model-free embedding-guided evolution) and `ALIRE2.md` (Phase 1 stabilization: sync refactor + path-6 regex).
**Trigger:** User asked for advisor + Codex + ExoCortex + Context7 review before writing a plan.

This file documents what was verified by reading the actual code **before** running Codex, so the plan rests on real code not paper claims.

---

## Part 1 — ALIRE.md (Path 6c): **HEAVILY HALLUCINATED**

### Invented APIs (none of these exist)

| ALIRE.md claim | Actual code | Evidence |
|---|---|---|
| `SmmuContext::retrieve_similar_topologies(embedding, top_k:, min_similarity:)` | `TopologySmmuBridge::retrieve_similar(&mut MultiViewMMU, query_chunk_id: &str, max_results: usize) -> Vec<TopologySuggestion>` | `sage-core/src/topology/smmu_bridge.rs:180` |
| `MapElitesArchive::get_elite(id) -> Option<Elite>` | `MapElitesArchive::get(&BehaviorDescriptor) -> Option<&EliteEntry>` | `sage-core/src/topology/map_elites.rs:513` |
| `struct Elite { composite_score, embedding_similarity, quality, topology }` | `pub struct EliteEntry { graph: TopologyGraph, quality: f32, cost: f32, latency_ms: f32, evaluation_count: u32 }` | `sage-core/src/topology/map_elites.rs:268` |
| `HybridVerifier::light_verify(topology)` | `HybridVerifier::verify(&TopologyGraph) -> VerificationResult` (only method — 8 internal checks, no "light" variant) | `sage-core/src/topology/verifier.rs:84-107` |
| `TopologySchema` type | `TopologyGraph` (the real type) | `sage-core/src/topology/topology_graph.rs:444` |
| `mutations::merge_node(topology)` | `mutations::merge_nodes(...)` (plural, 4 params) | `sage-core/src/topology/mutations.rs:566` |

### Syntax errors (not legal Rust)

- `smmu.retrieve_similar_topologies(task_embedding, top_k: 10, min_similarity: 0.65)` — Rust has no named arguments. This is Swift/Python syntax.
- `elite.composite_score = elite.quality * 0.65 + elite.embedding_similarity * 0.35` — assignment to a field that doesn't exist on `EliteEntry`.

### Architectural contradictions

1. **"100% model-free" claim, then uses an ONNX model.** ALIRE.md §1 says "Zéro dépendance à un modèle externe". Later it calls `quality_estimator.estimate(&current)` — the real `QualityEstimator` in `sage-core/src/routing/quality.rs:72` is **DistilBERT ONNX + Z3**. Path 6c cannot be model-free while depending on QualityEstimator.

2. **"data-driven (pas heuristique pure)" claim, then hardcodes heuristics.** ALIRE.md line 250-258:
   ```rust
   if structural.omega > 0.65 { RewireEdge }
   else if structural.delta > 5 { SplitNode }
   else { MutatePrompt }
   ```
   This is a pure if/else on magic thresholds (0.65, 5) with no calibration plan. Violates Critical Directive #2 (minimal heuristics).

3. **Path 6 is not a single LLM model being replaced.** ALIRE.md §2 frames Path 6 as "a model parked on the training branch" that needs replacement. Reality:
   - Path 6 (learned policy) is opt-in via `SAGE_ENABLE_PATH6=1` — off by default (CLAUDE.md).
   - Path 6 was **superseded functionally by `sage_recurse` tool** (Sprint 4, commit `13463fb`) — emergent subtasks are now handled via tool invocation, not topology regeneration.
   - The LLM checkpoint on HF is `yannabadie/sage-topology-policy-local` — it still works, but it's not the bottleneck ALIRE.md implies.

4. **Redundancy with Path 1 (template match) and the archive priors.** The archive already injects priors via `TopologySmmuBridge::inject_priors` (`smmu_bridge.rs:233`). ALIRE.md's "retrieve similar + pick best elite + mutate" is close to what Path 1-2 already do when the archive is warm. The actual gap is not a new path, it's calibration of existing paths.

### Invented literature claims

ALIRE.md §3 claims peer-reviewed backing:
- "EVOLVEpro (Science, 2025)" — not verified against arXiv/Science index.
- "Machine Learning-Guided Directed Evolution (PNAS, 2019)" — exists as a general concept, but applied to protein engineering, not topology search.
- "MAP-Elites in Latent Space" — real research line (Fontaine & Nikolaidis et al.) but doesn't validate ALIRE.md's *specific* pipeline (10-NN embeddings → MAP-Elites get + 2-step mutation + Z3 verify).

### Verdict on ALIRE.md

**Do not implement as specified.** If we want a model-free Path 6 alternative, start from what already exists:
- `TopologySmmuBridge::retrieve_similar` already does embedding-based retrieval.
- `MapElitesArchive::get` + `best_by_quality` already return elites.
- Mutations already exist with correct signatures.

The right move is not to write `policy.rs` from ALIRE.md's hallucinated code, but to:
- Check if the **current** Path-1/Path-2 archive priors pipeline can be tuned for the model-free use case.
- If a new path is really needed, design it against the **actual** API surface, with real constants sourced from calibration, not 0.65/0.68/0.03 magic numbers.

---

## Part 2 — ALIRE2.md (Phase 1 Stabilization): **MOSTLY SOUND, ONE PART OBSOLETE**

### §1.1 — Replace `__setattr__` sync with explicit `sync_from_python` / `sync_to_python`

**Status: VALID and ALIGNS with existing plan.**

The `__setattr__` mirror in `sage-python/src/sage/topology_controller.py` (added in commit `e26cd7b` during the 2026-04-20 Rust-First plan) is intentionally a transitional hack. ADR-012 acknowledged this. Moving to explicit `sync_from_python(reroute_count, spawn_count, node_retries)` is the right next step.

What ALIRE2.md proposes that checks out:
- Add explicit methods on `RustTopologyController` — consistent with the existing pyclass pattern (setters set_reroute_count, set_spawn_count, set_node_retries already exist, this would aggregate them).
- Add `validate_state_consistency()` — useful invariant check (reroute_count ≥ 0, node_retries keys match graph).
- Deprecate `__setattr__` magic — YES, that was the point of the hack-being-transitional.

Scope caveat: ALIRE2.md proposes `sync_to_python() -> PythonStateSnapshot` with `abstain_count` and `gate_loops`. Those fields do exist in the Rust controller, but they are **already readable** via pyclass getters. The snapshot struct is optional polish, not a correctness fix.

### §1.2 — Make path-6 regex configurable via `config/emergent_patterns.toml` + optional ONNX semantic fallback

**Status: OBSOLETE APPROACH.**

The user already flagged this in the prior session: *"Tu sais que je n'aime pas les regex, a quoi servent t'ils?"* Directive #2 (minimal heuristics) points the same way.

Reality of path 6 in this codebase:
- The regex-based `detect_emergent_subtask` in `RustTopologyController` was ported from Python during 2.5 (commit `b1d75c9`) — it scans LLM output for "need to also", "additionally", "TODO:" patterns.
- Since `sage_recurse` tool exists (Sprint 4, `13463fb`), an agent can *call* `sage_recurse(subtask)` explicitly instead of the orchestrator string-matching for emergent subtasks.
- Making the regex "smarter" (TOML weights, ONNX fallback) adds infrastructure without solving the real problem: regex path 6 is a heuristic that duplicates what an explicit tool call can do cleanly.

Right answer: **Remove** path-6 regex detection. Keep `sage_recurse` as the path. If LLM output needs to trigger a subtask, the agent should invoke the tool, not have the orchestrator regex-scan its output.

This is the H12 direction the prior session was interrupted on. The deletion should be retried cleanly.

### §1.2 salvage: If we keep path 6 in some form

One piece worth keeping from ALIRE2.md §1.2 — the `confidence` field on `EmergentTask`. If we expose `sage_recurse` with structured output, the caller's confidence can feed downstream budgeting. But this belongs in the tool schema, not in a regex detector.

### Verdict on ALIRE2.md

- §1.1 → go. Write as a clean PyO3 method, no `__setattr__` magic.
- §1.2 → pivot. Don't make the regex smarter — remove it, lean on `sage_recurse`.

---

## Part 3 — What this implies for the plan

1. **Skip ALIRE.md Path 6c entirely** — written against APIs that don't exist. If we want model-free retrieval-biased topology generation, it's a separate design exercise with the real API.
2. **Take ALIRE2.md §1.1** — state sync refactor, land as a small PR.
3. **Pivot ALIRE2.md §1.2** — finish the path-6 regex removal (resume H12) rather than add TOML config.
4. **Pending:** OxiZ v0.2.0 upgrade (tracked separately in `project_oxiz_v020_deferred.md`), real SWE-bench smoke (from superseded plan), Sprint 5 ablation execution.

This triage was done empirically by grep-ing the actual code. Codex background agent `aa6b570ffd3032207` is expected to return an independent read; if it converges, no more validation needed. If it diverges, reconcile before planning.
