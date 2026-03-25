# V2 Adaptive Plan — Pointer

Full plan: see the file delivered by Claude (YGN-SAGE_V2_Adaptive_Plan.md).

## Quick reference for Claude Code

### What's new in V2
The model learns to generate topologies with ADAPTATION metadata:
- `fallback_tier` on nodes (upgrade model if quality low)
- `adaptation.checkpoints` (where controller evaluates quality)
- `adaptation.max_upgrades` / `max_reroutes` (budget for runtime changes)
- `gate: conditional` on edges (closeable by controller)

### Implementation order
1. Rust: TopologyNode fields + TopologyGraph fields + get_predecessors() + reward resilience
2. Data: GPT-5.4 Pro prompts (120 adaptive + 60 static→adaptive + 40 recovery)
3. Python: env + controller integration + reward resilience signal + run_traced upgrade tracking
4. Python: ModelAssigner + ProviderPool in topology_env.py
5. Tests: unit + integration + EXEC=1 end-to-end
6. Pod: training

### Files to modify (Rust)
- `sage-core/src/topology/topology_graph.rs` — TopologyNode + TopologyGraph + get_predecessors()
- `sage-core/src/topology/reward.rs` — resilience field + compute_with_resilience()
- `sage-core/src/topology/pyo3_wrappers.rs` — if needed for new PyO3 exports

### Files to modify (Python)
- `sage-python/src/sage/verl/topology_env.py` — controller integration, ModelAssigner, ProviderPool
- `sage-python/src/sage/verl/reward.py` — resilience signal in compute_score()
- `sage-python/src/sage/topology/runner.py` — run_traced() upgrade tracking
- `sage-python/src/sage/grpo/execution_reward.py` — _build_topology_graph() adaptive fields
- `sage-python/scripts/verl/convert_sft_to_verl.py` — new data sources

### New files
- `sage-python/src/sage/verl/boot_training.py` — lightweight ModelAssigner bootstrap
- `sage-python/data/gpt54_adaptive_topologies.jsonl` — 120 entries
- `sage-python/data/gpt54_static_to_adaptive.jsonl` — 60 entries  
- `sage-python/data/gpt54_recovery_scenarios.jsonl` — 40 entries
