# Topology

Topology pillar: evolutionary search over agent graph structures and formal knowledge representation.

## Modules

### `evo_topology.py` -- TopologyEvolver

MAP-Elites evolutionary search over agent topologies. Explores the space of agent graph configurations (node count, connectivity, specialization) to find high-performing structures. Operates within the verified envelope established by the contracts system.

### `kg_rlvr.py` -- FormalKnowledgeGraph

Process Reward Model using Z3 DSL for formal reasoning. Parses `<think>` blocks from LLM output and scores each reasoning step via a safe AST evaluator (no `eval()`). All SMT paths (verify_invariant, verify_arithmetic, prove_memory_safety, check_loop_bound, score_with_z3) use Rust OxiZ first with z3-solver fallback. `verify_invariant` uses Rust `verify_invariant_with_feedback()` for clause-level diagnostic feedback.

### `llm_caller.py` -- LLM Topology Synthesis (Path 3)

LLM-driven topology generation: role prompt → structure prompt → Rust TopologySynthesizer. Completes the 6-path strategy in DynamicTopologyEngine.

### `engine.py` -- Topology Engine

Core topology management: creates, evaluates, and evolves agent graph structures.

### `patterns.py` -- Topology Patterns

Library of known-good topology patterns (chain, star, mesh, hierarchical) used as seeds for evolutionary search.

### `planner.py` -- Topology Planner

Plans topology transformations based on task requirements and performance feedback.

### `topology_archive.py` -- Topology Archive

Persistent archive of evaluated topologies with fitness metadata. Enables cross-run topology reuse.

### `topology_verifier.py` -- Topology Verification

Graph-analysis verification of topology properties (DAG validation, depth, deadlock detection) using Kahn's algorithm. Ensures evolved topologies satisfy structural invariants before deployment.

## Rust Topology (sage_core)

The Rust core provides high-performance topology components:

- **DynamicTopologyEngine** -- 6-path generate strategy: S-MMU retrieval → MAP-Elites archive → LLM synthesis → mutation → MCTS → template fallback. Evolution via MAP-Elites + CMA-ME refinement.
- **TopologyGraph** -- Unified IR wrapping petgraph::DiGraph with typed nodes (roles, capabilities, budgets) and three-flow edges (Control, Message, State).
- **8 Templates** -- Sequential, Parallel, AVR, SelfMoA, Hierarchical, Hub, Debate, Brainstorming.
- **HybridVerifier** -- 6 structural + 4 semantic checks + LTL integration (safety→errors, liveness→warnings).
- **MapElitesArchive** -- Quality-diversity archive with 4-dim BehaviorDescriptor (108 cells), Pareto dominance.
- **CMA-ME Emitter** -- Covariance Matrix Adaptation for directional topology parameter optimization.
- **MCTS Searcher** -- Monte Carlo Tree Search with UCB1 selection, 50 simulations or 100ms budget.
- **TopologyExecutor** -- Dual-mode scheduling: Static (Kahn's toposort) for acyclic, Dynamic (gate-based readiness) for cyclic.

## Design Note

Topology search runs INSIDE the verified contract envelope (see `contracts/`). Z3 targets contracts and policies, not topology shape directly. This separation was a key V2 design decision.
