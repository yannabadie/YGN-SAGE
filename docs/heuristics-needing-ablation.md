# Heuristics-Needing-Ablation Audit — 2026-04-18

**Scope**: every magic number that affects routing / topology / scoring
decisions in the shipping agent path. Excludes pure engineering limits
(retry counts, cache sizes, timeouts) — those are acceptable under
`.claude/rules/critical-directives.md` §2 as "engineering safety
limits".

**Feeds P1.4 of the 2026-04-18 mega-plan**. Companion: §6 of
`docs/superpowers/plans/2026-04-18-mega-plan.md`.

**Rule**: per critical-directives.md §2:
- **Acceptable**: research-backed initial values, documented as subject
  to ablation
- **Acceptable**: engineering safety limits (MAX_RETRIES, cache bounds)
- **Banned**: arbitrary magic numbers without justification or
  calibration plan

This doc is the single calibration plan.

## Table of contents

1. [Routing / ModelAssigner (Rust)](#routing--modelassigner-rust)
2. [Topology controller (Python)](#topology-controller-python)
3. [Pipeline stages / DAG features](#pipeline-stages--dag-features)
4. [Quality signals (constants.py)](#quality-signals-constantspy)
5. [Meta-Harness val_score](#meta-harness-val_score)
6. [Engineering safety limits (§2 acceptable)](#engineering-safety-limits-§2-acceptable)
7. [Ablation priority queue](#ablation-priority-queue)

## Routing / ModelAssigner (Rust)

| # | Symbol | Value | File:line | Source | Blast radius | Ablation |
|---|---|---|---|---|---|---|
| R1 | diversity_penalty per provider | 0.08 | `sage-core/src/routing/model_assigner.rs:319` | **Empirical** — v5f MiniMax saturation; added Apr 18 to spread providers on coder nodes | High (every routing decision with ≥2 providers on same role) | Sweep {0, 0.04, 0.08, 0.12, 0.20} on 50-task BCB Hard + 10-task SWE-Pro |
| R2 | diversity_penalty cap | 0.20 | `sage-core/src/routing/model_assigner.rs:319` | Engineering guard (prevents runaway penalty) | Low | Sweep {0.10, 0.20, 0.30} |
| R3 | role-tier score defaults (code, reasoning, tool_use, math) | 0.30 / 0.70 / 0.75 | `sage-core/src/routing/model_assigner.rs:542–717` | Research-backed tier banding per ModelCard, calibrated from MMLU + HumanEval — BUT values themselves are intuition | Medium | Joint ablation against kNN-learned alternative (planned post-P0.1) |
| R4 | THETA_COMPLEXITY (heuristic router) | 0.5 | (if present) | Priority-3 emergency fallback only, NOT dead code (AUDIT2 2026-04-24 corrected the prior framing) — ComplexityRouter historic ~34% — non-autoritative; `evidence_pending` in `docs/CLAIMS.yaml` | Zero (emergency fallback only) | Do not ablate — replace or delete |

## Topology controller (Python)

| # | Symbol | Value | File:line | Source | Blast radius | Ablation |
|---|---|---|---|---|---|---|
| T1 | THETA_GOOD | 0.7 | `sage-python/src/sage/topology_controller.py:46` | Research-backed (AdaptOrch 2602.16873 §4.2 cross-validation) | High (gates accept/retry) | Sweep {0.5, 0.6, 0.7, 0.8} on 50-task MASBENCH breadth |
| T2 | THETA_CRITICAL | 0.3 | `topology_controller.py:47` | Research-backed (AdaptOrch §4.2) | High (gates escalate/abort) | Sweep {0.2, 0.3, 0.4} joint with T1 |
| T3 | THETA_CONSISTENCY | 0.5 | `topology_controller.py:48` | **Empirical** — parallel-output variance threshold. No paper backing. | Medium | Sweep {0.3, 0.5, 0.7} |
| T4 | THETA_PRUNE | 0.2 | `topology_controller.py:49` | **Empirical** — aggressive node pruning threshold | Medium | Sweep {0.1, 0.2, 0.3} + compare vs MCTS-tuned pruning |

## Pipeline stages / DAG features

| # | Symbol | Value | File:line | Source | Blast radius | Ablation |
|---|---|---|---|---|---|---|
| P1 | _THETA_OMEGA | 0.5 | `sage-python/src/sage/pipeline_stages.py:154` | **Empirical** — parallelism threshold relative to node count. Origin unclear. | High (chooses parallel vs sequential topology) | Sweep {0.3, 0.5, 0.7} |
| P2 | _THETA_GAMMA | 0.6 | `pipeline_stages.py:155` | **Empirical** — coupling threshold | High (triggers horizon_pipeline vs parallel_fanout) | Sweep {0.4, 0.6, 0.8} |
| P3 | _THETA_DELTA | 5 | `pipeline_stages.py:156` | **Empirical** — depth threshold, integer node count | High (horizon_pipeline gate) | Sweep {3, 5, 7, 10} |
| P4 | RELEVANCE_GATE_THRESHOLD | 0.30 | `sage-python/src/sage/constants.py:102` | CRAG paper, Sprint 3 evidence (2026-03). Partial research backing — value itself is tuned. | Medium (memory injection gate) | Sweep {0.2, 0.3, 0.4} |

## Quality signals (constants.py)

| # | Symbol | Value | File:line | Source | Blast radius | Ablation |
|---|---|---|---|---|---|---|
| Q1 | QUALITY_BASELINE | — | `constants.py:42` (deleted 2026-04-18) | Banned §2 — dead heuristic | None | ✅ DELETED along with 8 sibling `QUALITY_*` constants + `test_quality_weights_sum_to_one` |
| Q2 | QUALITY_LENGTH_WEIGHT | — | `constants.py:42` (deleted 2026-04-18) | Banned §2 — dead heuristic | None | ✅ DELETED |
| Q3 | similarity_threshold | 0.7 | `sage-core/src/engine.rs` (kNN decision boundary) | Empirical, calibrated initial | Medium | Sweep {0.6, 0.7, 0.8, 0.85} |
| Q4 | quality_gate_threshold | 0.5 | `sage-core/src/engine.rs` | **Heuristic** — should be replaced with Z3 formal labeler output | Medium | Compare vs Z3 QualityLabeler only |

## Meta-Harness val_score

| # | Symbol | Value | File:line | Source | Blast radius | Ablation |
|---|---|---|---|---|---|---|
| M1 | sentinel_bonus | 0.25 | `external/meta-harness/reference_examples/ygn_sage/benchmark.py:73` (`score_predictions`) | **Empirical reward shaping** — "gradient when no candidate crosses real-patch threshold yet". NOT research-backed. | High (every Meta-Harness candidate ranking) | Sweep {0.0, 0.1, 0.25, 0.5} vs real-patch-only |

## Engineering safety limits (§2 acceptable)

These are deliberately simple retry/cache bounds. §2 calls them
acceptable because they guard stability without gating correctness
decisions. Listed here so they're not confused with routing/topology
heuristics.

| # | Symbol | Value | File:line | Purpose |
|---|---|---|---|---|
| S1 | S2_MAX_RETRIES_BEFORE_ESCALATION | 2 | `constants.py:68` | S2 AVR retry budget |
| S2 | S3_MAX_RETRIES | 2 | `constants.py:69` | S3 CEGAR repair attempts |
| S3 | DEFAULT_EXCLUSION_TTL_SEC | 300 | `ProviderPool` | Dead-provider re-probe window |
| S4 | DEFAULT_COST_PER_1K | 0.001 | `constants.py:213` | Fallback only — primary path uses LiteLLM response_cost (post-Apr 18) |

## Ablation priority queue

Ordered by (blast radius × "banned under §2 unless calibrated" weight):

1. **Q1 / Q2** — DELETE outright, not ablate. Use `git grep QUALITY_BASELINE` to find holdouts.
2. **R1** — diversity_penalty 0.08. Directly affects every multi-agent routing decision today, empirical justification only.
3. **M1** — sentinel_bonus 0.25. Drives every Meta-Harness candidate ranking we're about to generate in P0.1; getting this wrong wastes iteration cycles.
4. **T3 / T4** — THETA_CONSISTENCY / THETA_PRUNE. Empirical, affects topology-control decisions across runs.
5. **P1 / P2 / P3** — pipeline_stages thresholds. Empirical, affects topology selection.
6. **T1 / T2** — research-backed but still need Z3-labelled ablation to confirm carrying from AdaptOrch's domain.
7. **Q3 / Q4** — similarity_threshold / quality_gate. Compare against Z3 formal.
8. **P4** — RELEVANCE_GATE_THRESHOLD. Lower priority — already has Sprint 3 evidence.
9. **R3** — role-tier score defaults. Joint ablation against a kNN-learned alternative after P0.1 stabilises.
10. **R2** — diversity_penalty cap. Lowest priority of the active R-group; safety wraps around R1.

All sweeps write to `docs/benchmarks/ablation/<YYYY-MM-DD>-<symbol>.json`
with: value, benchmark, n_tasks, pass_rate, 95% CI.

## Re-audit triggers

Rerun this doc when:
- A new empirical constant is introduced (grep PR comments for "// calibrated" / "# empirical")
- Any Q1 / Q2 caller is found in production code (they should already be dead)
- A Meta-Harness iteration completes (M1 may move based on observed signal-to-noise)
