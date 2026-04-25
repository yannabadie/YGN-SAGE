---
name: V2.1 Priorities (April 3, 2026)
description: Next steps after V2 training sprint — GRPO V2.1 with plain reward_funcs, data rebalancing, inter-node truncation fix
type: project
---

## Priority 1: GRPO V2.1 — Plain reward_funcs (HIGH)

Use `train_local_qwen3_4b.py` with TRL GRPOTrainer + `reward_funcs` only. No `environment_factory`, no `tools`.
- **Start from Phase C checkpoint** (0.922 reward, 40% MASBENCH) — not V2 SFT (regressed)
- `chat_template_kwargs={"enable_thinking": False}` to suppress Qwen3's `<think>` mode
- beta=0.0, temperature=1.0, K=4 generations
- Validated by AgentConductor (2602.17100) + The Conductor (Sakana AI, ICLR 2026)

**Why:** V2 GRPO with environment_factory destroyed the `<tool_call>` format. Plain reward_funcs let the model generate freely; rewards parse and evaluate.

## Priority 2: Rebalance V2 Data (MEDIUM)

Current: 22% create_topology, 60% adapt_topology → V2 SFT generated small 3-node topologies
Target: 50%+ create_topology, 30% adapt_topology, 20% multi-turn
- Duplicate create_topology entries 2-3x
- Remove simulated episodic memory templates (replace with real or none)

**Why:** Data imbalance is the likely cause of V2 SFT regression vs Phase C (-20pp MASBENCH).

## Priority 3: Fix Inter-Node Truncation (MEDIUM)

Change 500 → 4000 chars in:
- `sage-python/src/sage/topology/runner.py` ~line 240
- `sage-python/src/sage/verl/topology_env.py` line 605

**Why:** Reviewers see only first 10 lines of coder output. Multi-node topologies lose their value.

## Priority 4: Verify cards.toml Models (LOW)

Unverified models added by pod Claude:
- gemini-3.1-flash-live, grok-code-fast-1, kimi-k2-thinking
- MiniMax-M2.5 / MiniMax-M2.5-highspeed (casing inconsistent)

## Priority 5: MASBENCH on Pod (LOW)

Local machine has network timeouts. Run the same 10-task bench on pod for accurate comparison.

**How to apply:** Start with Priority 1 (GRPO V2.1). If reward converges and format is preserved, run MASBENCH immediately.
