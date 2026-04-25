---
name: Frontier Research March 2026 (updated end of session)
description: Verified SOTA papers for SAGE — RewardFlow, MASPRM, MAPPA, TopoCurate, confirmed gaps in learned adaptive topologies
type: reference
---

## Confirmed Research Gaps (SAGE can be FIRST)

1. **Learned topology + runtime adaptation** — NO system trains a model to generate DAGs with embedded adaptation policies
2. **Conditional/gated edges in learned DAGs** — NO system generates edges with quality-based gates
3. **FrugalGPT per-node in topology** — NO system embeds per-node cascading policies (cheap-first, escalate-if-poor)
4. **Combined learned + adaptive + multi-provider** — confirmed by ExoCortex: NONE exist

## Key Papers to Integrate

- **RewardFlow** (OpenReview 5oGJbM5u86) — Graph-based reward propagation via BFS/PageRank. States=nodes, actions=edges. Propagates terminal rewards to intermediate states. Per-node credit WITHOUT model acting at each step. Compatible with GRPO.
- **MASPRM** (2510.24803) — Multi-Agent System PRM. MAS-MCTS rollouts, Bradley-Terry per-agent credit. Code: github.com/milad1378yz/MASPRM
- **MAPPA** (2601.23228) — Per-action per-agent process rewards via AI feedback coach. +5-17.5pp AIME.
- **TopoCurate** (2603.01714, March 2026) — 3 topological metrics for data curation: Reflective Recovery, Semantic Efficiency, Distributional Diversity
- **Budget-Aware Agentic Routing** (2602.21227) — FrugalGPT per-step selection between cheap/expensive models
- **Graph-GRPO** (2603.02701) — Edge-level credit via per-edge success rate. +1.82%. Already implemented in SAGE.
- **GiGPO** (2505.10978) — Step-level advantage. Collapses for single-action BUT works for multi-turn revision.
- **AgentPRM** (2502.10325) — Process reward for LLM agents via Monte Carlo. Code: github.com/sanjibanc/agent_prm
- **Explicit Credit via Dependence Graphs** (2601.21523) — Agent interaction graph for credit assignment
- **MetaGen** (2601.19290) — Test-time topology adaptation (heuristic, not learned)
- **OFA-MAS** (2601.12996, WWW 2026) — MoE graph generative model. Code: github.com/Shiy-Li/OFA-MAS
- **ARG-Designer** (2507.18224, AAAI 2026 Oral) — Autoregressive graph generation. Code available.

## AdaptOrch Details (2602.16873)
- Entirely rule-based: threshold logic on ω, δ, γ metrics
- Adaptive Synthesis Protocol: CS (cosine similarity) scoring, re-route with γ+0.2
- Termination: ceil((1-γ₀)/0.2) ≤ 5 iterations
- Code: github.com/dmae97/adaptorch
- NOT learned — purely heuristic

## AgentConductor 2nd-Turn Structure (2602.17100)
- 2nd-turn = revised YAML only, NO adaptation metadata (no fallback_tier, no checkpoints)
- Error types: NO_YAML(-2), YAML_PARSE(-1.5), WRONG_ANSWER(1.0), RUNTIME_ERROR(0.7)
- Max 2 turns, GRPO via veRL, Qwen2.5-3B
- No per-edge or per-node credit
- No code/data released

**How to apply:** Use RewardFlow for per-node reward propagation in Run 1. Use MASPRM/MAPPA concepts for per-agent PRM in Run 2. TopoCurate metrics for data curation. GiGPO only with multi-turn revision loop.

## New Findings — March 25, 2026 Session

### Apply when Phase A converges
- **Dr. MAS** (2602.08847) — Per-agent advantage normalization. Fixes gradient instability when planner/coder/reviewer have divergent reward distributions. +5.6% avg@16. Apply in compute_gigpo_advantages() by normalizing per role prefix. CRITICAL for Phase C.
- **Dynamic Reward Weighting** (2509.11452) — Gradient-based learned weights eliminate fixed lambdas. w_i(t)=w_i(t-1)*exp(eta*I_i/mu). Works with GRPO on Qwen3-8B. Apply AFTER ablation of _W_* constants in reward.py.
- **The Conductor simplification** (2512.04388, ICLR 2026) — Binary reward (1.0/0.5/0.0) sufficient. No KL, no cost term. Efficiency emerges naturally in 200 GRPO iters. Consider if V5 reward sparsity persists.

### Already applied this session
- **HyEvo code nodes** (2603.19639) — node_type="code" in TopologySchema, runner dispatch, cascaded eval, reward hybrid bonus. 13-19x cost on MBPP.
- **Cascaded eval** (Trust or Escalate ICLR'25 + Model Cascading 2405.15842) — 4-stage cascaded_eval.py. 26-87% eval cost savings.
- **GigaEvo** (2511.17592) — Multi-island MAP-Elites shows NO benefit at <1000 entries. Removed multi_island.py, kept Rust single archive.

### Monitor only
- **PAMA** (2508.07768) — Closed-form Pareto for >3 objectives. Not needed yet.
- **CMA-MAE** (Fontaine 2022) — Better than CMA-ME at low resolution. Apply if evolution stalls.
- **C3PO** (2511.07396, NeurIPS'25) — Budget-adaptive cascade staging. Apply if fixed gamma insufficient.
