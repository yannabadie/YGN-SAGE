---
name: SOTA Landscape March 2026
description: Complete competitive analysis — OpenSAGE, The Conductor, AgentConductor, CARD, HyEvo vs YGN-SAGE. Updated March 28, 2026.
type: research
---

## YGN-SAGE Position (March 28, 2026)
- BigCodeBench Hard: 37.8% (budget model) vs SOTA 40.0% (The Conductor recursive)
- SWE-bench: not evaluated. OpenSAGE at 59% Pro.
- Training: SFT OK, GRPO V5 running on 2xH100 NVL (step 39, reward ~0.06)

## Primary Competitors

### The Conductor (2512.04388, ICLR'26, Sakana AI)
- BigCodeBench Hard: **40.0%** (recursive) — SOTA
- Qwen2.5-7B trained with GRPO, 200 iters, NO KL, binary reward
- 7 workers. Recursive self-invocation for test-time scaling.
- Code NOT public.

### OpenSAGE (2602.16891, ICML'26, UC Berkeley)
- SWE-bench Pro 59%, CyberGym 60.2% (#1), Terminal-Bench 78.4% (#1)
- Self-programming ADK. Dynamic tool synthesis. Hierarchical memory (Neo4j).
- Code NOT public. THE system SAGE is named after and aims to surpass.

### AgentConductor (2602.17100)
- 97.5% HumanEval with 3B model. RL topology evolution.
- Topological density function. GRPO via veRL on Qwen2.5-3B.

### CARD (2603.01089, ICLR'26)
- Conditional GCN graph encoder. Environment-aware topology.
- Adapts to model upgrades/API changes at runtime.

### HyEvo (2603.19639, March 2026)
- Hybrid LLM+code nodes: 19x cost, 16x latency reduction on MBPP.
- Applied to SAGE: code nodes, cascaded eval, reflect-then-generate.

## SAGE Unique Advantages (no competitor has these)
1. Formal verification (OxiZ SMT + LTL + CEGAR)
2. Checkpoint micro-decisions (upgrade/continue/reroute)
3. Edge-level credit (Graph-GRPO)
4. 5-signal reward (vs flat pass/fail)
5. Rust core performance engine
6. Open-source MIT
7. kNN pre-routing 92%

## What to Steal (Priority)
1. Dynamic tool synthesis (OpenSAGE) — agents create tools at runtime
2. Recursive self-invocation (The Conductor) — test-time scaling
3. Dr. MAS per-agent normalization (2602.08847) — fixes GRPO instability
4. No KL regularization (The Conductor) — simpler, 200 iters sufficient
5. ICRL: RL-only training (2603.08068) — may eliminate SFT warmup
