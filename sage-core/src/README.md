# src/

Source code for the `sage_core` Rust crate. Each module maps to a PyO3-exported class or function accessible from Python.

## Module Layout

### `lib.rs` -- PyModule Entry Point

Defines the `#[pymodule] fn sage_core(...)` function. Registers all PyClasses and PyFunctions into the Python module. Conditionally includes `sandbox` (behind `sandbox` or `tool-executor` features), `embedder` (behind `onnx` feature), and `routing` (behind `onnx` feature). When `tool-executor` is enabled, registers `ValidationResult`, `ExecResult`, and `ToolExecutor` PyClasses. When `onnx` is enabled, registers `RustEmbedder`, `AdaptiveRouter`, `RoutingResult`, and `StructuralFeatures` PyClasses.

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

- **`memory/`** -- Multi-tier memory data plane (Arrow, S-MMU, RAG cache, ONNX embedder)
- **`routing/`** -- Adaptive Router (see below)
- **`sandbox/`** -- ToolExecutor security pipeline (tree-sitter validator, subprocess executor, Wasm WASI sandbox). `validator.rs`, `subprocess.rs`, `tool_executor.rs` behind `tool-executor` feature. `wasm.rs` behind `sandbox` feature. `ebpf.rs` disabled.

### `routing/` -- Adaptive Router (feature: `onnx`)

Learned S1/S2/S3 routing pipeline. Stage 0 (structural features, always compiled) + Stage 1 (BERT ONNX classifier, behind `onnx` feature).

- **`features.rs`** -- `StructuralFeatures` (PyClass): word_count, has_code_block, has_question_mark, keyword_complexity, keyword_uncertainty, tool_required. `extract(task) -> Self`. 6 keyword groups. 6 unit tests.
- **`router.rs`** -- `AdaptiveRouter` (PyClass): `route(task) -> RoutingResult`, `route_stage0(task) -> RoutingResult`, `record_feedback(...)`, `has_classifier() -> bool`. Dynamic ONNX input discovery (supports BERT and RoBERTa models). 512-token truncation. 10 unit tests.
