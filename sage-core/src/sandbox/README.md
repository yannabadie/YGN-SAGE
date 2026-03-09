# sandbox/

Sandboxed code execution for YGN-SAGE agents. Provides isolation via Wasm Component Model. The entire module is behind the `sandbox` Cargo feature flag.

## Module Layout

### `mod.rs` -- Module Root

Conditionally exports submodules. Currently active: `wasm`. The `ebpf` module is disabled (solana_rbpf removed from dependencies due to CI cross-compilation issues).

### `wasm.rs` -- WasmSandbox

Component-based Wasm sandbox using wasmtime v36 LTS with the Component Model and WIT-defined `tool-env` world (see `interface.wit` at crate root).

- **`WasmSandbox`** (PyClass) -- Creates a wasmtime Engine with component-model support. Caches compiled components in a `DashMap` for reuse.

Key methods:
- `execute_precompiled(name, compiled_bytes, tool_name, args_json, env)` -- Run a pre-compiled Wasm component (works without cranelift). Uses `Component::deserialize()` (unsafe: bytes must match the same wasmtime version and config).
- `execute(name, wasm_bytes, tool_name, args_json, env)` -- Compile and run raw `.wasm` bytes. Requires the `cranelift` feature; returns an error on Windows without it.
- Returns a Python dict: `{stdout, stderr, exit_code, result_json}`.

WIT interface (`interface.wit`):
```wit
world tool-env {
    record tool-input { name, args, env }
    record tool-output { stdout, stderr, exit-code, result-json }
    export run: func(input: tool-input) -> tool-output;
}
```

### `ebpf.rs` -- eBPF Executor (disabled)

Not compiled into the crate. The `solana_rbpf` dependency was removed because its build script cross-compiles for the BPF target, breaking CI on Ubuntu. The source remains for reference.

Contains two PyClasses (not exported):
- `EbpfSandbox` -- Loads ELF or raw eBPF instructions, executes via solana_rbpf VM.
- `SnapBPF` -- Userspace CoW memory snapshotting via `DashMap<String, Arc<Vec<u8>>>`. Supports `snapshot()`, `restore()` (returns clone), `delete()`, `count()`.

### `snap_bpf.c` -- Kernel eBPF Stub

**STUB -- NOT FUNCTIONAL.** Placeholder for a future kernel-level SnapBPF agent. Compiles but contains no real logic. The actual CoW implementation is the Rust `SnapBPF` struct in `ebpf.rs`.

### `z3_validator.rs` -- REMOVED

Deleted as dead code per Z3-01 audit finding. Z3 verification is in Python: `sage.sandbox.z3_validator` and `sage.topology.kg_rlvr`.

## Platform Notes

- **Windows**: Only `execute_precompiled()` works. The `cranelift` feature causes `STATUS_STACK_BUFFER_OVERRUN` during MSVC compilation.
- **Linux CI**: Use `--features sandbox,cranelift` for full JIT compilation support.
