# ASI Convergence Plan (2026+)

Based on deep audits of NotebookLM research synthesis, this track represents the leap from SOTA to an ASI baseline by refactoring the core pillars of YGN-SAGE.

## Phase 1: Cognitive Memory Refactoring (S-MMU)
- [x] **Task 1: TierMem Architecture**
  - Implement a two-tier memory hierarchy in `sage-core`.
  - Retain `WorkingMemory` (Arrow) for high-density execution tensors.
  - Integrate a Rust-based DAG (e.g., `petgraph`) for contextual memory.
- [x] **Task 2: Context-Aware MCP (CA-MCP)**
  - Implement the A-MEM / MEM1 standard.
  - Agents must update a unified semantic node instead of linearly appending context (Active Forgetting).

## Phase 2: Neuro-Symbolic Wasm/eBPF Execution
- [ ] **Task 3: Deprecate Docker**
  - Replace legacy Docker fallback with Firecracker micro-VMs or extended `wasmtime`.
  - Implement SnapBPF memory restoration hooking into OS page cache.
- [x] **Task 4: SMT Firewall (Z3 Validator)**
  - Integrate Z3 (Rust bindings) in `EbpfSandbox`.
  - Mechanize the validation of code AST before JIT compilation to eliminate "Implementation Drift".
  - Implemented `z3_validator.rs` in `sage-core`.

## Phase 3: Dynamic Topology via MCTS
- [ ] **Task 5: AFlow Implementation**
  - Replace static agent workflows with a `TopologyPlanner`.
  - Use Stochastic Differentiable Tree Search (S-DTS) to explore agent DAG configurations.
  - Use the newly built H96 Zero-Copy router to rapidly evaluate node probabilities.

## Phase 4: Open-Ended Evolution (DGM & SAMPO)
- [ ] **Task 6: Whole-System Evolution**
  - Upgrade the MAP-Elites `EvolutionEngine` to mutate hyperparameters, memory structures, and tools, not just prompts/code.
  - Implement SAMPO (Stable Agentic Multi-turn Policy Optimization) using sequence-level clipping instead of token-level clipping.
- [ ] **Task 7: System 3 AI & KG-RLVR**
  - Introduce Process Reward Models based on Knowledge Graphs ($R_{path}$) to force rigorous compositional logic in agent reasoning paths (`<think>`).
