---
name: V2 Adaptive Topology Design (March 20, 2026)
description: Complete design spec for SAGE V2 — multi-turn GiGPO, episodic memory, RewardFlow, local Qwen3.5-4B validation then pod Qwen3.5-9B
type: project
---

## Design Spec
`docs/superpowers/specs/2026-03-20-v2-adaptive-topology-design.md` — committed on VeRLGIGPO branch.

## Key Decisions
1. **Multi-turn GiGPO** — model generates YAML, sees checkpoint results, decides upgrade/continue. GiGPO step-level advantage applies properly (not single-action collapse).
2. **Episodic memory during training** — SQLite stores past episodes, top-3 similar injected in observation. Memory offline (between epochs) locally, online (per-reset) on pod.
3. **RewardFlow per-node credit** (2603.18859) — PageRank propagation from terminal states to intermediate nodes. Uses same K=4 GRPO rollouts, no extra overhead.
4. **5-signal reward** — structural(0.20) + execution(0.35) + rewardflow(0.20) + resilience(0.15) + cost_efficiency(0.10). Weights subject to ablation.
5. **CARD price penalty** (2603.01089) — R_cost = 1.0 - tanh(cost / budget_ref[difficulty])
6. **Gate semantics** — `conditional` in YAML → `gate: "open"` + `condition: "quality_check"` in Rust. Uses existing TopologyEdge.condition field.
7. **V2 actions = 2 of 5** — only `continue` and `upgrade_model` exposed to training. prune/reroute/spawn deferred to V3.

## Local Validation
- **Model:** Qwen3.5-4B with Unsloth QLoRA (~5GB VRAM on RTX 3500 Ada 12GB)
- **Framework:** Unsloth + TRL GRPOTrainer (not veRL — needs multi-GPU)
- **Phase A:** 2225 prompts, structural reward, $0 API, ~2-4h
- **Phase B:** 600 curated, 5-signal reward, 8 providers, ~4-8h, ~$30-50 API

## Pod Training (after validation)
- **Model:** Qwen3.5-9B on H100 80GB
- **Framework:** verl-agent with GiGPO multi-turn
- **True incremental execution** — model decides at each checkpoint

## Data (220 new + existing)
- gpt54_adaptive_topologies.jsonl: 120 entries ✓
- gpt54_static_to_adaptive.jsonl: 60 entries ✓
- gpt54_recovery_scenarios.jsonl: 40 entries (80 effective — initial + recovered) ✓

## Files to Create/Modify
- MODIFY: sage-core/src/topology/topology_graph.rs (+3 TopologyNode, +3 TopologyGraph)
- MODIFY: sage-core/src/topology/reward.rs (+2 RewardScore, compute_full())
- MODIFY: sage-python/src/sage/verl/topology_env.py (multi-turn, memory, ProviderPool)
- MODIFY: sage-python/src/sage/verl/reward.py (+resilience, +cost, RewardFlow)
- MODIFY: sage-python/scripts/verl/convert_sft_to_verl.py (+3 data sources)
- CREATE: sage-python/src/sage/verl/rewardflow.py
- CREATE: sage-python/src/sage/verl/training_memory.py
- CREATE: sage-python/scripts/train_local_grpo.py

## Review Issues Fixed
- C1: gate:conditional → gate:open + condition field (no Rust enum change)
- C2: reward weights are initial values with ablation plan (not hardcoded)
- C3: TopologyNode new fields use keyword args with defaults (backward compat)
- C4: V2 exposes 2/5 controller actions (explicitly deferred 3)
- Recovery data: 2 entries per scenario (initial + recovered = 80 total)
- Embeddings: precomputed offline with arctic-embed-m, stored in SQLite
- Rate limit recovery: exponential backoff + structural fallback

**How to apply:** This is the master design doc. Implementation plan follows via writing-plans skill. Local validation runs autonomously (weekend), pod after results confirmed.
