# sage-core

Rust data-plane crate for YGN-SAGE, providing high-performance memory management, sandboxed execution, and SIMD utilities. Exposed to the Python SDK via PyO3 bindings as the `sage_core` module.

## Crate Type

Built as both `cdylib` (for PyO3/maturin) and `rlib` (for Rust consumers).

## Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `tool-executor` | Enables `ToolExecutor`, `ValidationResult`, `ExecResult` via tree-sitter 0.26 + process-wrap 9 | off |
| `sandbox` | Enables `WasmSandbox` + WASI sandbox via wasmtime v36 LTS (runtime + component-model + wasmtime-wasi) | off |
| `cranelift` | Adds JIT compilation to `sandbox` (Linux only; causes MSVC stack overflow on Windows) | off |
| `onnx` | Enables `RustEmbedder` via ort 2.0 + tokenizers 0.21 (all-MiniLM-L6-v2, 384-dim) | off |

## Build Commands

```bash
# Default build (no optional features)
cargo build

# Build with ONNX embedder support
cargo build --features onnx

# Build with ToolExecutor (tree-sitter + subprocess)
cargo build --features tool-executor

# Build with Wasm sandbox (pre-compiled modules only on Windows)
cargo build --features sandbox

# Build with ToolExecutor + Wasm WASI sandbox (full security pipeline)
cargo build --features sandbox,tool-executor

# Build with sandbox + JIT compilation (Linux CI only)
cargo build --features sandbox,cranelift

# Build + install Python bindings
maturin develop
maturin develop --features onnx
maturin develop --features tool-executor
maturin develop --features sandbox,tool-executor

# Run tests
cargo test --workspace                            # 7 tests (default features)
cargo test --features tool-executor               # 36 tests (validator + subprocess + ToolExecutor)
cargo test --features sandbox,tool-executor       # 63 tests (full sandbox + ToolExecutor)
cargo test --features onnx                        # +5 ONNX embedder tests (requires model download + onnxruntime DLL)
cargo clippy                                      # Lint
```

## Module Overview

| Module | Description |
|--------|-------------|
| `memory/` | Arrow-backed working memory, S-MMU multi-view graph, FIFO+TTL RAG cache, ONNX embedder |
| `sandbox/` | ToolExecutor (tree-sitter validator + subprocess + Wasm WASI sandbox), eBPF executor (disabled) |
| `types.rs` | Core data types: AgentConfig, ToolSpec, MemoryScope, AgentStatus, TopologyRole |
| `agent.rs` | AgentBridge -- Rust-side agent runtime representation |
| `pool.rs` | AgentPool -- thread-safe DashMap-backed sub-agent registry |
| `hardware.rs` | HardwareProfile -- CPU/memory/SIMD capability detection |
| `simd_sort.rs` | SIMD-accelerated sorting functions for topology planning |
| `lib.rs` | PyModule entry point, registers all PyClasses and PyFunctions |

## Key Dependencies

- **pyo3 0.25** -- Python bindings (extension-module)
- **arrow 55.0 + pyo3-arrow** -- Zero-copy columnar memory
- **petgraph 0.6** -- S-MMU multi-view directed graph
- **dashmap 6** -- Lock-free concurrent maps (AgentPool, RagCache, SnapBPF)
- **wasmtime 36.0 + wasmtime-wasi 36.0** -- Wasm Component Model + WASI p2 sandbox (optional)
- **tree-sitter 0.26 + tree-sitter-python 0.25** -- AST-based code validation (optional)
- **process-wrap 9** -- Subprocess execution with tokio timeout (optional)
- **ort 2.0.0-rc.12** -- ONNX Runtime inference, `load-dynamic` (optional, needs `pip install onnxruntime`)
- **tokenizers 0.21** -- HuggingFace tokenizer (optional)
- **sysinfo + raw-cpuid** -- Hardware detection
- **ulid + chrono** -- Event IDs and timestamps

## Python Exports

All PyClasses registered in `lib.rs`:

- `AgentConfig`, `ToolSpec`, `MemoryScope`, `AgentStatus`, `TopologyRole`
- `AgentPool`, `WorkingMemory`, `MemoryEvent`, `HardwareProfile`
- `RagCache`
- `WasmSandbox` (behind `sandbox` feature)
- `ToolExecutor`, `ValidationResult`, `ExecResult` (behind `tool-executor` feature)
- `RustEmbedder` (behind `onnx` feature)
- `h96_quicksort`, `h96_quicksort_zerocopy`, `h96_argsort`, `vectorized_partition_h96` (functions)
