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

## Architecture

```
EvolutionEngine
  |-- SAMPO action selection (5 actions)
  |-- LLMMutator (propose candidates)
  |-- Evaluator (score fitness)
  |-- Population (MAP-Elites grid)
  \-- SnapBPF (rollback on failure)
```
