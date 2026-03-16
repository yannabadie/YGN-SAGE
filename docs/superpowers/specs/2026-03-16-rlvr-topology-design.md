# RLVR-Topology: Online Multi-Agent Architecture Learning with Verified Dense Rewards

## Problem

AgentConductor (arXiv 2602.17100) achieves +29pp on APPS via RL-trained topology generation with a 3B model. Graph-GRPO (arXiv 2603.02701) achieves 92.45% average via edge-level credit assignment. Both train offline with execution-only rewards (binary pass/fail).

YGN-SAGE has a unique advantage: **formal verification infrastructure** (OxiZ SMT, HybridVerifier, LTL) that can provide dense, formally grounded rewards — not binary pass/fail, but clause-level diagnostic feedback. Combined with online learning (bandit + MAP-Elites already wired), this creates a system that:
1. Learns topology policy online (not frozen at inference)
2. Gets dense verified rewards (not binary execution results)
3. Optimizes jointly over topology + model assignment (not topology alone)

No published system combines these three properties.

## Architecture

### Phase 1: S_complex Density Function + N_max Bounds (Rust, ~80 LOC)

Prerequisite infrastructure. Implements AgentConductor's proven topology cost metric.

**S_complex** (from AgentConductor Theorem 1):
```
S_node = exp(-|V| / N_max(system))
S_edge = exp(-|E| / (|V| * (|V| - 1) / 2))
S_depth = 1 - max_sequential_depth / |V|
S_complex = exp(S_node + 2 * S_edge + S_depth)
```

**N_max bounds** (difficulty-aware):
- S1 (simple): N_max = 4
- S2 (moderate): N_max = 7
- S3 (complex): N_max = 10

File: `sage-core/src/topology/density.rs`
PyO3 exports: `TopologyDensity.compute(graph, system) -> DensityScore`

### Phase 2: Verified Dense Reward Function (Rust, ~120 LOC)

The core innovation. Multi-signal reward combining execution + formal verification.

**Reward signals** (all from existing SAGE infrastructure):
1. **Execution score** (0.0-1.0): task pass@1 from sandbox
2. **Structural score** (0.0-1.0): HybridVerifier (6 structural + 4 semantic checks)
3. **Temporal score** (0.0-1.0): LtlVerifier (reachability, safety, liveness, bounded liveness)
4. **Density score** (0.0-1.0): S_complex from Phase 1
5. **CEGAR bonus** (0.0-0.2): counterexample count from `verify_invariant_with_feedback()`

**Composite reward** (not heuristic — each component is formal):
```
R = w_exec * execution_score
  + w_struct * structural_score
  + w_temporal * temporal_score
  + w_density * density_score
  + cegar_bonus
```

Weights learned via bandit (not hardcoded). ContextualBandit already supports per-arm exploration — extend arms to include reward weight configurations.

File: `sage-core/src/topology/reward.rs`
PyO3 exports: `TopologyReward.compute(graph, execution_result, system) -> RewardScore`

### Phase 3: APPS + LiveCodeBench Benchmark Adapters (Python, ~200 LOC each)

Required for comparison with AgentConductor and Graph-GRPO.

**APPS adapter** (`sage-python/src/sage/bench/apps_bench.py`):
- Dataset: APPS (10,000 problems, 3 difficulty levels)
- Evaluation: subprocess execution with test cases
- Pattern: same as BigCodeBench adapter

**LiveCodeBench adapter** (`sage-python/src/sage/bench/livecodebench_bench.py`):
- Dataset: LiveCodeBench (contamination-free, rolling)
- Evaluation: standard competitive programming evaluation
- Pattern: same as BigCodeBench adapter

CLI: `python -m sage.bench --type apps --limit 20` and `python -m sage.bench --type livecodebench --limit 20`

### Phase 4: Topology SFT Data Collection (Python, ~150 LOC)

Generate training data for the topology policy model.

**Script**: `sage-python/scripts/collect_topology_sft.py`

For each benchmark task:
1. Run task through SAGE with each of the 8 templates
2. Record (task, topology_yaml, execution_result, reward_score)
3. Keep only topologies with reward > threshold
4. Export as JSONL for SFT training

Target: 5,000+ validated topology examples (AgentConductor used 4,500).

### Phase 5: GRPO Training Loop (Python, ~300 LOC)

Train a small model (Qwen-2.5-3B or phi-3-mini) to generate topologies.

**Script**: `sage-python/scripts/train_topology_grpo.py`

Uses TRL's `GRPOTrainer` with custom reward function:
```python
from trl import GRPOTrainer, GRPOConfig

def reward_fn(completions, prompts):
    """RLVR reward: verified dense rewards from SAGE infrastructure."""
    rewards = []
    for completion, prompt in zip(completions, prompts):
        topology = parse_yaml_topology(completion)
        graph = build_topology_graph(topology)

        # Dense verified reward (not binary)
        reward = TopologyReward.compute(graph, execute(graph, prompt), system)
        rewards.append(reward)
    return rewards

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_fn],
    config=GRPOConfig(
        num_generations=16,  # K=16 per Graph-GRPO
        beta=0.04,
    ),
)
```

### Integration: Wire Trained Policy into TopologyEngine

After training, the model is exported to ONNX and loaded by Rust `TopologyEngine` as a new 7th path:
```
Path 0: S-MMU retrieval (existing)
Path 1: MAP-Elites archive (existing)
Path 2: LLM synthesis (existing)
Path 3: Mutation (existing)
Path 4: MCTS (existing)
Path 5: Template fallback (existing)
Path 6: RLVR-trained policy (NEW — highest priority when model available)
```

The trained policy becomes the primary topology generator, with existing paths as fallback.

## Success Criteria

1. S_complex reduces avg topology cost by >30% vs current templates
2. N_max prevents over-engineering (no S1 tasks get >4 nodes)
3. APPS pass@1 > 40% (AgentConductor: 58.8% with 3B, but they use dedicated SFT)
4. LiveCodeBench pass@1 > 30% (competitive with Graph-GRPO)
5. Trained topology policy generates valid DAGs >95% of the time
6. Online learning shows improvement over 100+ tasks (bandit regret decreases)

## Files

| File | Action | LOC | Lang |
|------|--------|-----|------|
| `sage-core/src/topology/density.rs` | CREATE | ~80 | Rust |
| `sage-core/src/topology/reward.rs` | CREATE | ~120 | Rust |
| `sage-core/src/topology/mod.rs` | MODIFY | +5 | Rust |
| `sage-core/src/lib.rs` | MODIFY | +4 | Rust |
| `sage-python/src/sage/bench/apps_bench.py` | CREATE | ~200 | Python |
| `sage-python/src/sage/bench/livecodebench_bench.py` | CREATE | ~200 | Python |
| `sage-python/src/sage/bench/__main__.py` | MODIFY | +20 | Python |
| `sage-python/scripts/collect_topology_sft.py` | CREATE | ~150 | Python |
| `sage-python/scripts/train_topology_grpo.py` | CREATE | ~300 | Python |

Total: ~1080 LOC (200 Rust + 880 Python). Phases 1-3 are independent and can be parallelized.

## Research References

- AgentConductor (2602.17100): SFT+GRPO topology, +29pp on APPS
- Graph-GRPO (2603.02701): Edge-level credit assignment, 92.45% avg
- VeRPO (2601.03525): Dense verifiable rewards for code
- OFA-MAS (2601.12996): MoE autoregressive graph generation
- MaAS (2502.04180): Agentic Supernet, joint topology+model (ICML 2025)
- AlgoForge: Collaborative GRPO for Planner+Coder agents
