# src/

Source code for the `sage_core` Rust crate. Each module maps to a PyO3-exported class or function accessible from Python.

## Module Layout

### `lib.rs` -- PyModule Entry Point

Defines the `#[pymodule] fn sage_core(...)` function. Registers all PyClasses and PyFunctions into the Python module. Conditionally includes `sandbox` (behind `sandbox` or `tool-executor` features), `embedder` (behind `onnx` feature), and `routing`. When `tool-executor` is enabled, registers `ValidationResult`, `ExecResult`, and `ToolExecutor` PyClasses. When `onnx` is enabled, registers `RustEmbedder` and `StructuralFeatures` PyClasses. Routing classes (`SystemRouter`, `ContextualBandit`, `ModelRegistry`, `ModelAssigner`, `RoutingDecision`, `RoutingConstraints`) are always registered.

### `types.rs` -- Core Data Types

PyO3-exported enums and structs for agent configuration:

- **`MemoryScope`** -- `Isolated` (default), `Shared`, `Inherited`. Controls how an agent accesses parent memory.
- **`TopologyRole`** -- `Root` (default), `Vertical`, `Horizontal`, `Mesh`. Agent's position in the topology.
- **`AgentStatus`** -- `Created`, `Running`, `Paused`, `Completed`, `Failed`, `Terminated`.
- **`AgentConfig`** -- Agent creation config with ULID-based ID, model, system prompt, tools, max_steps, parent_id. Supports builder pattern: `with_tools()`, `with_max_steps()`, `with_parent()`, `with_memory_scope()`, `with_topology_role()`.
- **`ToolSpec`** -- Tool definition with name, description, JSON parameters schema, category, and sandbox requirement flag.
- **`Message`**, **`Role`**, **`ToolCall`**, **`ToolResult`** -- Conversation types (Rust-only, not PyO3-exported).

### `agent.rs` -- AgentBridge

Rust-side runtime representation of an agent in the pool:

- **`Agent`** -- Wraps `AgentConfig` with runtime state: `status`, `step_count`, `result`, `children_ids`. Not directly exposed to Python (used internally by `AgentPool`).

### `pool.rs` -- AgentPool

Thread-safe agent registry backed by `DashMap<String, Agent>`:

- **`AgentPool`** (PyClass) -- `register(config)` returns agent ID, `search(query)` finds by name/prompt substring, `list()` returns all configs, `get_children(parent_id)` returns child configs, `terminate(id)` sets status to Terminated, `len()` / `is_empty()` for size queries.

### `hardware.rs` -- HardwareProfile

System capability detection via `sysinfo` and `raw-cpuid`:

- **`HardwareProfile`** (PyClass) -- `detect()` static method probes total/free memory, physical/logical cores, CPU brand, AVX2/AVX512/NEON flags. Properties: `is_simd_capable`, `to_json()`.

### `simd_sort.rs` -- SIMD-Accelerated Sorting

PyFunctions for high-performance sorting (uses Rust pdqsort; placeholder for vqsort when Windows support lands):

- **`h96_quicksort(arr)`** -- Sort f32 Vec, returns new sorted Vec.
- **`h96_quicksort_zerocopy(arr)`** -- In-place sort on a NumPy array (zero-copy via PyArray1).
- **`vectorized_partition_h96(arr, pivot)`** -- Partition into (left < pivot, right >= pivot).
- **`h96_argsort(arr)`** -- Returns indices that would sort the array ascending. Used for MCTS UCB node selection in TopologyPlanner.

## Submodule Directories

- **`memory/`** -- Multi-tier memory data plane (Arrow, S-MMU, RAG cache, ONNX embedder with `batch_cosine_similarity` SIMD)
- **`routing/`** -- Cognitive routing engine (see below)
- **`sandbox/`** -- ToolExecutor security pipeline (tree-sitter validator, subprocess executor, Wasm WASI sandbox). `validator.rs`, `subprocess.rs`, `tool_executor.rs` behind `tool-executor` feature. `wasm.rs` behind `sandbox` feature.

### `routing/` -- Cognitive Routing Engine

SystemRouter (PRIMARY, 88% GT accuracy) + ContextualBandit + ModelRegistry + ModelAssigner. Stage 0 structural features always compiled.

- **`features.rs`** -- `StructuralFeatures` (PyClass): word_count, has_code_block, has_question_mark, keyword_complexity, keyword_uncertainty, tool_required. `extract(task) -> Self`. 6 keyword groups. 6 unit tests.
- **`system_router.rs`** -- `SystemRouter` (PyClass): cognitive system decision engine. Hard constraints → structural scoring → domain hint → bandit/budget selection. `route_integrated()`, `record_outcome()`. PRIMARY router (88% GT accuracy).
- **`model_card.rs`** -- `ModelCard` (PyClass): CognitiveSystem (S1/S2/S3) + domain_scores + safety_rating + TOML parsing.
- **`model_registry.rs`** -- `ModelRegistry` (PyClass): TOML-loaded model catalog, telemetry calibration, `select_best_for_domain()`.
- **`model_assigner.rs`** -- `ModelAssigner` (PyClass): per-node model assignment using ModelCard scoring (affinity 0.4 + domain 0.4 + cost 0.2). Filters by capabilities and budget. `assign_models()`, `assign_single_node()`.
- **`bandit.rs`** -- `ContextualBandit` (PyClass): per-arm Beta/Gamma posteriors, Thompson sampling, Pareto front. SQLite persistence behind `cognitive` feature.
- **`knn.rs`** -- kNN routing support for embedding-based classification.
- **`quality.rs`** -- Quality estimation utilities for routing feedback.

### `topology/` -- Topology Engine

Unified topology IR, templates, verification, and evolutionary search:

- **`topology_graph.rs`** -- `TopologyGraph` (PyClass): petgraph::DiGraph with typed `TopologyNode` (roles, capabilities, budgets) and three-flow `TopologyEdge` (Control, Message, State). Methods: `add_node`, `add_edge`, `node_count`, `edge_count`, `get_node`, `to_json`.
- **`templates.rs`** -- `PyTemplateStore` (PyClass): 8 topology templates (Sequential, Parallel, AVR, SelfMoA, Hierarchical, Hub, Debate, Brainstorming). `instantiate(name, model_ids) -> TopologyGraph`.
- **`verifier.rs`** -- `HybridVerifier` (PyClass): 6 structural + 4 semantic checks + LTL integration. `verify(graph) -> VerificationReport`.
- **`map_elites.rs`** -- `MapElitesArchive` (PyClass): quality-diversity archive with 4-dim BehaviorDescriptor (108 cells), Pareto dominance. SQLite persistence behind `cognitive` feature.
- **`cma_me.rs`** -- `CmaEmitter`: CMA-ME for directional topology parameter optimization. Diagonal covariance, elite-weighted updates.
- **`mcts.rs`** -- `MctsSearcher`: Monte Carlo Tree Search with UCB1 selection, random mutation expansion, HybridVerifier rollout. Budget: 50 simulations or 100ms.
- **`mutations.rs`** -- 7 mutation operators: add_node, remove_node, swap_model, rewire_edge, split_node, merge_nodes, mutate_prompt.
- **`llm_synthesis.rs`** -- `TopologySynthesizer`: 3-stage LLM pipeline (Role → Structure → Validation).
- **`executor.rs`** -- `PyTopologyExecutor` (PyClass): dual-mode scheduling — Static (Kahn's toposort) for acyclic, Dynamic (gate-based readiness) for cyclic.
- **`engine.rs`** -- `PyTopologyEngine` (PyClass): 6-path generate strategy (S-MMU → archive → LLM → mutation → MCTS → template). Evolution via MAP-Elites + CMA-ME refinement.

### `verification/` -- Formal Verification

SMT verification and temporal property checking:

- **`smt.rs`** -- `SmtVerifier` (PyClass, behind `smt` feature): OxiZ pure-Rust SMT (QF_LIA). 10 PyO3 methods: `prove_memory_safety`, `check_loop_bound`, `verify_arithmetic`, `verify_arithmetic_expr`, `verify_invariant`, `verify_invariant_with_feedback`, `synthesize_invariant`, `verify_array_bounds`, `validate_mutation`, `verify_provider_assignment`. Expression parser, CEGAR loop (max 5 rounds), clause-level diagnostics.
- **`ltl.rs`** -- `LtlVerifier` (PyClass, always compiled): temporal property verification on TopologyGraph. `check_reachability` (BFS), `check_safety` (no HIGH→LOW paths), `check_liveness` (entries→exits), `check_bounded_liveness` (depth ≤ K).
