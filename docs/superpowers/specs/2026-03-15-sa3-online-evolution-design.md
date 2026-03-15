# SA-3: Online Evolution — Design Spec

## Problem

YGN-SAGE has a complete offline evolution system (MAP-Elites + CMA-ME + MCTS + LLM synthesis) but `_auto_evolve=False` everywhere. The system never evolves during task execution. The previous benchmark (-1pp on HumanEval+) was done with single-model offline evolution on a saturated benchmark — wrong conditions.

Research (AlphaEvolve, Live-SWE-agent, EvoAgent) shows that online evolution — evolving during task execution — is what produces breakthroughs. Live-SWE-agent achieved 77.4% on SWE-bench by self-evolving its approach during execution.

## Principle: Rust first, no heuristics

The evolution engine is already Rust (`topology/engine.rs`, `topology/map_elites.rs`, `topology/cma_me.rs`, `topology/mcts.rs`). Online evolution extends this to runtime, not replaces it.

## Architecture

```
Task arrives
    │
    ▼
Pipeline Stage 2 (SELECT_TOPOLOGY)
    │── DynamicTopologyEngine.generate() [existing]
    │   returns topology based on S-MMU, archive, LLM synthesis
    │
    ▼
Pipeline Stage 4 (EXECUTE)
    │── TopologyRunner executes per-node
    │
    ▼
Pipeline Stage 5 (LEARN)
    │── QualityEstimator scores result (Z3 labeler)
    │── ContextualBandit records outcome [existing]
    │── NEW: EvolutionFeedback
    │   ├── If quality < threshold → mutate topology + retry
    │   ├── Record (topology, quality) in MAP-Elites archive
    │   └── CMA-ME updates covariance from outcome
    │
    ▼
Next task: archive has been updated → better topology selection
```

## What changes

### 1. Rust: `topology/engine.rs` — add `record_outcome()` method

The engine already has `evolve()` for offline optimization. Add `record_outcome(topology_id, quality, latency_ms)` that:
- Updates MAP-Elites archive cell for this topology
- Feeds CMA-ME emitter with the outcome
- Updates S-MMU bridge with topology performance data

This is already partially wired in `topology/smmu_bridge.rs` but the quality signal was always 0.5 (heuristic). Now it gets real Z3-verified quality.

### 2. Python: `pipeline.py` Stage 5 — wire evolution feedback

After bandit recording, add:
```python
if self.engine and quality is not None:
    self.engine.record_outcome(ctx.topology_id, quality, ctx.latency_ms)
```

### 3. Python: `boot.py` — enable `_auto_evolve`

Set `_auto_evolve=True` when TopologyEngine is available. The agent_loop checks this flag and triggers evolution after each task.

### 4. Rust: `topology/engine.rs` — add `should_evolve()` method

Returns true when the archive has enough diversity data (>10 observations) and the last N outcomes show low quality. This prevents unnecessary evolution on already-good topologies.

## What does NOT change

- MAP-Elites, CMA-ME, MCTS algorithms (already Rust, already correct)
- TopologyGraph, TopologyNode, TopologyEdge (unchanged)
- HybridVerifier (unchanged)
- Offline evolution CLI (`python -m sage.evolution`) (unchanged)

## Success Criteria

- Evolution fires automatically when quality is low (not manually triggered)
- Archive grows over time with task outcomes
- Subsequent tasks benefit from archive-based topology selection
- No performance regression on BigCodeBench (evolution should help, not hurt)
- Zero heuristic thresholds — `should_evolve()` uses Z3 quality scores, not magic numbers

## Files

| File | Action | LOC estimate |
|------|--------|-------------|
| `sage-core/src/topology/engine.rs` | MODIFY | ~50 (record_outcome, should_evolve) |
| `sage-core/src/topology/pyo3_wrappers.rs` | MODIFY | ~20 (PyO3 expose) |
| `sage-python/src/sage/pipeline.py` | MODIFY | ~10 (wire in Stage 5) |
| `sage-python/src/sage/boot.py` | MODIFY | ~5 (_auto_evolve=True) |
| `sage-python/src/sage/agent_loop.py` | MODIFY | ~10 (trigger evolution) |

Total: ~95 LOC, mostly Rust.
