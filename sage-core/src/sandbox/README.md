# sandbox/

Sandboxed code execution and security validation for YGN-SAGE agents. Since ADR-013 §5 (2026-04-22), the default pipeline for `validate_and_execute` is `tree-sitter AST validation → embedded RustPython wasm32-wasip1 sandbox (deny-by-default WASI-p1)`; there is NO subprocess fallback on this path. A subprocess fall-through still exists on `execute_raw`, which is a separately gated bypass (`SAGE_UNSAFE_RAW_EXEC=1`) that also skips AST validation.

## Feature Flags

| Feature | Modules enabled |
|---------|-----------------|
| `tool-executor` | `validator.rs`, `subprocess.rs`, `tool_executor.rs` |
| `sandbox` | `wasm.rs` + `wasm_python.rs` (embedded RustPython, opt-in via `cranelift`) (+ Wasm paths in `tool_executor.rs`) |
| `cranelift` | JIT compilation for `wasm.rs` and `wasm_python.rs` (Linux default; Windows build supported since ADR-013) |

Since ADR-013 §5 (2026-04-22), `sandbox`, `cranelift`, `tool-executor`, and `cognitive` are **Cargo default features**. A default `cargo build` ships a safe-by-default binary. Use `--no-default-features` for leaner builds.

### Build-time sandbox artifact gate

`sage-core/build.rs` looks for a pre-compiled `rustpython.wasm` at
`external/rustpython-wasm-target/wasm32-wasip1/release/rustpython.wasm`
and embeds it into the library via `include_bytes!`. If it's missing
the default behaviour is to emit an empty placeholder + a cargo warning
so fresh clones still compile — the runtime short-circuits with
`WasmPythonInitError::BytesMissing` on first call.

Release and CI builds that need the sandbox to be present should set
`SAGE_REQUIRE_WASM=1` in the build environment. With that flag set, a
missing `rustpython.wasm` becomes a compile-time `panic!` so the build
fails loudly before producing a placeholder-backed binary. This is the
minimum-invasive realisation of the ALIRE audit's "assert non-empty
sandbox artifact" recommendation — it doesn't break fresh-clone
developer workflows, but it gives CI / release pipelines an explicit
gate.

### Wasm-python JIT cache

`WasmPythonExecutor::new()` caches the compiled `Module` under
`$SAGE_WASM_CACHE_DIR` (or `$HOME/.sage/wasm_python_cache/` by default)
to avoid re-JITing the 37 MB wasm on every fresh process. Cache key
binds `Engine::precompile_compatibility_hash` + a SHA-256 of the
embedded wasm; stale / corrupt cache files self-heal on deserialize
failure. Opt-out with `SAGE_WASM_CACHE_DISABLE=1`.

## Module Layout

### `mod.rs` -- Module Root

Conditionally exports submodules based on feature flags. `wasm` requires `sandbox`, `validator`/`subprocess`/`tool_executor` require `tool-executor`.

### `validator.rs` -- tree-sitter Python Code Validator (feature: `tool-executor`)

AST-based security validation using tree-sitter 0.26 with tree-sitter-python 0.25. Error-tolerant: validates partial parse trees even when code has syntax errors.

- **`validate_python_code(code: &str) -> ValidationResult`** -- Scans the CST for blocked imports and calls.
- **`ValidationResult`** (PyClass) -- `valid: bool`, `errors: Vec<String>`.

**23 blocked modules:** `os`, `sys`, `subprocess`, `shutil`, `ctypes`, `importlib`, `socket`, `http`, `ftplib`, `smtplib`, `xmlrpc`, `multiprocessing`, `threading`, `signal`, `resource`, `code`, `codeop`, `pathlib`, `glob`, `tempfile`, `pickle`, `shelve`, `builtins`.

**11 blocked calls:** `exec`, `eval`, `compile`, `__import__`, `breakpoint`, `open`, `getattr`, `setattr`, `delattr`, `globals`, `locals`.

Uses S-expression queries on `import_statement`, `import_from_statement`, and `call` nodes. Note: tree-sitter 0.26 `QueryMatches` implements `StreamingIterator`, not `Iterator` — use `while let Some(m) = matches.next()`.

9 unit tests.

### `subprocess.rs` -- Subprocess Executor (feature: `tool-executor`)

Sandboxed Python subprocess execution with timeout and kill-on-drop:

- **`execute_python_subprocess(python_exe, code, args_json, timeout_secs) -> ExecResult`** -- Writes code to a temp file, passes args via stdin, runs with `tokio::process::Command` and `kill_on_drop(true)`.
- **`ExecResult`** (PyClass) -- `stdout`, `stderr`, `exit_code` (i32), `timed_out` (bool), `duration_ms` (u64).

Uses `tokio::runtime::Builder::new_current_thread()` for the async runtime. Temp files use PID + nanosecond timestamp for uniqueness.

5 unit tests.

### `tool_executor.rs` -- ToolExecutor PyO3 Class (feature: `tool-executor`)

Combined validator + sandboxed executor. Primary entry point for dynamic tool creation from Python.

- **`ToolExecutor`** (PyClass) -- `new(python_exe=None, timeout_secs=30)`
- **`validate(code) -> ValidationResult`** -- tree-sitter validation only
- **`validate_and_execute(py, code, args_json) -> ExecResult`** -- Validate via tree-sitter, then execute via the sandbox chain below. Since ADR-013 §5 (2026-04-22), the chain **fails closed** without Wasm — no silent subprocess fallback. Releases GIL via `py.allow_threads()`.
- **`execute_raw(py, code, args_json) -> ExecResult`** -- Gated by `SAGE_UNSAFE_RAW_EXEC=1`. Bypasses AST validation. Prefers embedded RustPython when bundled; falls through to subprocess otherwise (the explicit opt-in authorised the bypass).
- **`load_precompiled_component(compiled_bytes, wasi=false)`** -- Load pre-compiled Wasm component (works without cranelift). Behind `sandbox` feature.
- **`load_component(wasm_bytes, wasi=false)`** -- Compile and load Wasm component (requires `cranelift` feature).
- **`has_wasm() -> bool`** -- Check if an operator-loaded Wasm component is present (does NOT report on the embedded RustPython path — use `embedded_wasm_available()` at the crate root for that).
- **`has_wasi() -> bool`** -- Check if the loaded component uses WASI.

**Execution priority for `validate_and_execute` (post ADR-013 §5):**
1. Operator-loaded Wasm Component-Model (if a component was registered via `load_component` / `load_precompiled_component`).
2. Embedded RustPython wasm_python sandbox (default path, bundled under `sandbox + cranelift` with a non-empty `rustpython.wasm` at build time).
3. **Hard fail** with an operator-facing `stderr`. No silent subprocess fallback — the old `SAGE_UNSAFE_UNSANDBOXED=1` gate was removed when the 40-attack red-team corpus validated the Wasm path.

`execute_raw` keeps its own ordering (embedded RustPython → subprocess), which is legal only because the `SAGE_UNSAFE_RAW_EXEC=1` gate already authorised the bypass.

8 unit tests.

### `wasm.rs` -- Wasm Sandbox (feature: `sandbox`)

Component-based Wasm sandbox using wasmtime v36 LTS with the Component Model and WIT-defined `tool-env` world.

- **`WasmSandbox`** (PyClass) -- Creates a wasmtime Engine with component-model support. Caches compiled components in `DashMap`.
- **`WasiState`** -- WASI host state with deny-by-default capabilities. `new_restrictive()` creates a context with only `inherit_stdout()` + `inherit_stderr()`. Implements `WasiView` trait.

Key methods:
- `execute_precompiled(name, compiled_bytes, tool_name, args_json, env)` -- Run pre-compiled component (no cranelift needed)
- `execute_precompiled_wasi(name, compiled_bytes, tool_name, args_json, env)` -- Run pre-compiled WASI component with deny-by-default capabilities
- `execute(name, wasm_bytes, tool_name, args_json, env)` -- Compile and run (requires `cranelift`)

Standalone functions for ToolExecutor:
- `execute_wasi_component(engine, component, code, args_json)` -- WASI-aware execution
- `execute_bare_component(engine, component, code, args_json)` -- Non-WASI execution

WIT interface (`interface.wit`):
```wit
world tool-env {
    record tool-input { name, args, env }
    record tool-output { stdout, stderr, exit-code, result-json }
    export run: func(input: tool-input) -> tool-output;
}
```

### `ebpf.rs` -- eBPF Executor (disabled)

Not compiled into the crate. Source remains for reference. See main README for details.

### `snap_bpf.c` -- Kernel eBPF Stub

**STUB -- NOT FUNCTIONAL.** Real CoW implementation is the Rust `SnapBPF` struct in `ebpf.rs`.

## Wasm Components

Pre-built components in `sage-core/plugins/`:
- **`python-runner/`** -- Phase 2 expression evaluator (pure-compute, no WASI). Implements `tool-env` world with eval mode (`{"eval": "a + b", "a": 3}`) and echo mode.
- **`cpython-runner/`** -- CPython WASI component via componentize-py. Full Python execution in WASI sandbox (~42MB).

## Platform Notes

- **Windows**: Only `execute_precompiled()` / `load_precompiled_component()` work. The `cranelift` feature causes `STATUS_STACK_BUFFER_OVERRUN` during MSVC compilation.
- **Linux CI**: Use `--features sandbox,cranelift` for full JIT compilation support.
- **PyO3 tests**: Must call `pyo3::prepare_freethreaded_python()` before `Python::with_gil()`.
