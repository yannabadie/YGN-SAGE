---
name: Vision vs Reality Gap (Updated April 3, 2026)
description: Gaps between SAGE's research ambition and current implementation. Training convergence remains THE blocker. Updated after V2 sprint.
type: project
---

## Core Finding (updated 2026-04-03)
YGN-SAGE's 5 pillars are ~90% architecturally complete. The "Self-Adaptive" promise is partially real but the trained model has never beaten template fallback in an end-to-end benchmark.

## Gaps Closed (V2 Sprint, 2026-03-31 to 2026-04-02)
1. **Path 6 runtime loader** — FIXED (was silently dead — TopologyEdge wrong constructor)
2. **V2 SFT training** — DONE (8633 entries, loss converged)
3. **V2 GRPO training** — DONE but BROKEN (environment_factory destroyed format)
4. **MASBENCH V2 evaluation** — DONE (20% depth, regression detected)
5. **5 codebase bugs** fixed (TopologyEdge, ProviderPool kimi, codex_max, BigCodeBench CLI, tokenizer)

## Gaps Remaining
1. **No converged GRPO checkpoint that preserves format** — THE critical gap. Phase C SFT (0.922 structural) is best, but MASBENCH shows only 40% depth. GRPO is needed to push quality beyond structural ceiling.
2. **Self-programming** — agents don't create tools at runtime (OpenSAGE does)
3. **Recursive self-invocation** — The Conductor does this for test-time scaling
4. **ONNX QualityEstimator** — not trained, returns None
5. **Memory mock on most installs** — Rust S-MMU only with compiled sage_core

## SAGE's Unique Advantages (no competitor has)
1. Formal verification (OxiZ SMT + LTL + CEGAR) — sub-0.1ms
2. Checkpoint micro-decisions (upgrade/continue/reroute)
3. Edge-level credit (Graph-GRPO)
4. 5-signal reward (vs flat pass/fail everywhere else)
5. Rust core performance engine
6. Open-source MIT
7. kNN pre-routing 92%

**Why:** Every training attempt so far has either hit a reward ceiling (structural only) or destroyed the output format (environment_factory). The next attempt (V2.1 with plain reward_funcs from Phase C) is the most promising path.

**How to apply:** Don't assume "training is close to done" — it has failed 3+ times. Plan for iteration.
