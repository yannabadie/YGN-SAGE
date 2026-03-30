# Evolution

Evolutionary self-improvement engine using MAP-Elites and LLM-driven code mutation. One of the 5 cognitive pillars of YGN-SAGE.

## Modules

### `engine.py` -- EvolutionEngine

Main evolution loop implementing MAP-Elites with SAMPO (Strategic Action for Meta-Parameter Optimization). Chooses 1 of 5 SAMPO actions per generation: explore, exploit, refine hyperparameters, prune, or diversify. Manages generation lifecycle and trajectory tracking.

### `mutator.py` -- Code Mutation

Base mutation logic. Generates candidate code modifications (parameter tweaks, structural changes) for the evolution pipeline.

### `llm_mutator.py` -- LLM-Driven Mutation

Uses LLM providers to propose intelligent code mutations. Injects SAMPO context (current action, fitness landscape) into the mutation prompt so the LLM understands the strategic direction.

### `evaluator.py` -- Fitness Assessment

Evaluates candidate individuals against fitness criteria. Produces scores used by the MAP-Elites grid to determine elite placement.

### `population.py` -- Population Grid

Manages the MAP-Elites population: `Individual` (genotype + fitness + metadata) and the elite grid. Handles insertion, replacement, and bounded history tracking.

### `self_improve.py` -- Self-Improvement Orchestration

High-level orchestrator that wires the evolution loop into the agent system. Coordinates mutation, evaluation, and selection across generations.

### `ebpf_evaluator.py` -- eBPF Evaluation (Experimental)

Experimental evaluator using eBPF sandbox (via `sage-core` SnapBPF). Provides CoW memory snapshots for safe mutation rollback.

## Online Evolution (SA-3)

The evolution engine runs **online** during task execution, not just offline. The pipeline wiring:

1. **Stage 5 (LEARN)**: `pipeline.py` calls `engine.record_outcome()` with real quality scores
2. **Agent loop**: When `should_evolve()` returns true (Rust, gated on outcome count + archive coverage), calls `engine.evolve(pop_size=5, generations=2)` — a lightweight pass (~10ms in Rust)
3. **Archive persists**: MAP-Elites archive saved/loaded via SQLite at boot/shutdown

`should_evolve()` (Rust) gates on: min_outcomes >= 5, cooldown >= 3 new outcomes, coverage < 80%.

## AdaptiveMutator (ShinkaEvolve, arXiv 2509.19349)

Thompson sampling bandit over LLM tiers for mutation selection. Each tier (budget, fast, mutator, reasoner) has a Beta posterior updated by mutation success/failure. The bandit converges toward the tier that produces the most improving mutations.

## Architecture

```
EvolutionEngine
  |-- SAMPO action selection (5 actions)
  |-- LLMMutator (propose candidates)
  |-- AdaptiveMutator (Thompson sampling tier selection)
  |-- Evaluator (score fitness + Wilcoxon validation)
  |-- Population (MAP-Elites grid)
  |-- Rust TopologyEngine.should_evolve() → evolve()
  \-- SnapBPF (rollback on failure)
```
