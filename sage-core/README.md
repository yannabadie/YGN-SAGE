# sage-core

Rust data-plane crate for YGN-SAGE, providing high-performance memory management, sandboxed execution, and SIMD utilities. Exposed to the Python SDK via PyO3 bindings as the `sage_core` module.

## Crate Type

Built as both `cdylib` (for PyO3/maturin) and `rlib` (for Rust consumers).

## Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `sandbox` | Enables `WasmSandbox` via wasmtime v36 LTS (runtime + component-model) | off |
| `cranelift` | Adds JIT compilation to `sandbox` (Linux only; causes MSVC stack overflow on Windows) | off |
| `onnx` | Enables `RustEmbedder` via ort 2.0 + tokenizers 0.21 (all-MiniLM-L6-v2, 384-dim) | off |

## Build Commands

```bash
# Default build (no optional features)
cargo build

# Build with ONNX embedder support
cargo build --features onnx

# Build with Wasm sandbox (pre-compiled modules only on Windows)
cargo build --features sandbox

# Build with sandbox + JIT compilation (Linux CI only)
cargo build --features sandbox,cranelift

# Build + install Python bindings
maturin develop
maturin develop --features onnx

# Run tests
cargo test --workspace                  # 36 tests (default features)
cargo test --features onnx              # +5 ONNX embedder tests (requires model download)
cargo clippy                            # Lint
```

## Module Overview

| Module | Description |
|--------|-------------|
| `memory/` | Arrow-backed working memory, S-MMU multi-view graph, FIFO+TTL RAG cache, ONNX embedder |
| `sandbox/` | Wasm Component Model sandbox (wasmtime v36), eBPF executor (disabled), Z3 validator |
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
- **wasmtime 36.0** -- Wasm Component Model sandbox (optional)
- **ort 2.0.0-rc.12** -- ONNX Runtime inference (optional)
- **tokenizers 0.21** -- HuggingFace tokenizer (optional)
- **sysinfo + raw-cpuid** -- Hardware detection
- **ulid + chrono** -- Event IDs and timestamps

## Python Exports

All PyClasses registered in `lib.rs`:

- `AgentConfig`, `ToolSpec`, `MemoryScope`, `AgentStatus`, `TopologyRole`
- `AgentPool`, `WorkingMemory`, `MemoryEvent`, `HardwareProfile`
- `RagCache`
- `WasmSandbox` (behind `sandbox` feature)
- `RustEmbedder` (behind `onnx` feature)
- `h96_quicksort`, `h96_quicksort_zerocopy`, `h96_argsort`, `vectorized_partition_h96` (functions)
