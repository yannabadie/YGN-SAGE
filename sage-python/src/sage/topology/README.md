# Topology

Topology pillar: evolutionary search over agent graph structures and formal knowledge representation.

## Modules

### `evo_topology.py` -- TopologyEvolver

MAP-Elites evolutionary search over agent topologies. Explores the space of agent graph configurations (node count, connectivity, specialization) to find high-performing structures. Operates within the verified envelope established by the contracts system.

### `kg_rlvr.py` -- FormalKnowledgeGraph

Process Reward Model using Z3 DSL for formal reasoning. Parses `<think>` blocks from LLM output and scores each reasoning step via a safe AST evaluator (no `eval()`). Supports Z3 assertions: `bounds`, `loop`, `arithmetic`, `invariant`. Used in S3 (formal verification) tier routing.

### `engine.py` -- Topology Engine

Core topology management: creates, evaluates, and evolves agent graph structures.

### `patterns.py` -- Topology Patterns

Library of known-good topology patterns (chain, star, mesh, hierarchical) used as seeds for evolutionary search.

### `planner.py` -- Topology Planner

Plans topology transformations based on task requirements and performance feedback.

### `topology_archive.py` -- Topology Archive

Persistent archive of evaluated topologies with fitness metadata. Enables cross-run topology reuse.

### `z3_topology.py` -- Z3 Topology Verification

Z3-based verification of topology properties (connectivity, capability coverage, resource bounds). Ensures evolved topologies satisfy structural invariants before deployment.

## Design Note

Topology search runs INSIDE the verified contract envelope (see `contracts/`). Z3 targets contracts and policies, not topology shape directly. This separation was a key V2 design decision.
