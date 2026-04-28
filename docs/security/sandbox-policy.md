# Sandbox policy

YGN-SAGE has TWO distinct sandbox surfaces. They are independent and have
their own escape hatches that DO NOT cross-unlock.

## 1. Tool execution path

`ToolExecutor.validate_and_execute` (Rust, `sage-core/src/sandbox/`):
- AST validator (tree-sitter)
- Embedded RustPython `wasm32-wasip1` runtime (`wasm_python.rs`)
- Deny-by-default: no filesystem, no network, no subprocess, no env

Default since 2026-04-22 ADR-013. `SAGE_UNSAFE_RAW_EXEC=1` allows
`ToolExecutor.execute_raw` which bypasses BOTH layers - audited dev escape
hatch only.

Build-time gate: `SAGE_REQUIRE_WASM=1` turns a missing `rustpython.wasm`
artifact into a `panic!` instead of placeholder. Belongs on `maturin build`
step in CI release profiles, not on pytest runtime.

## 2. Topology code-node path

`TopologyRunner._execute_code_node`
(`sage-python/src/sage/topology/runner.py`):
- Calls `sage.sandbox.isolated_executor.execute_isolated`
- Linux + bwrap available: bubblewrap-isolated subprocess
- Linux + no bwrap: fail-closed (R4)
- non-Linux: fail-closed (R4)

R4 contract (since 2026-04-28):
- `ImportError` on `from sage.sandbox.isolated_executor import execute_isolated`:
  `SandboxUnavailable` raised
- `BWRAP_AVAILABLE=False` (no bwrap binary detected): `SandboxUnavailable`
  raised
- `SAGE_UNSAFE_RAW_EXEC=1` (env, explicit): falls back to raw
  `subprocess.run`, logs WARNING with "DO NOT USE IN PRODUCTION"

## 3. ToolForge Gate 2

`sage.tools.forge` Gate 2 (`_run_tests`):
- Default: fail-closed without subprocess
- `SAGE_UNSAFE_TOOLFORGE_SUBPROCESS=1` (env, explicit): allows subprocess
  for the test runner

Disjoint escape hatch: `SAGE_UNSAFE_RAW_EXEC` does NOT unlock ToolForge Gate 2.

## 4. Operator decision matrix

| Scenario | Setting | Result |
|---|---|---|
| Production | All env unset | All 3 surfaces fail-closed |
| Local dev (no bwrap) | `SAGE_UNSAFE_RAW_EXEC=1` | Tool exec sandboxed (Rust); code-node raw subprocess (warning); ToolForge Gate 2 still fail-closed |
| Local dev (need ToolForge tests) | `SAGE_UNSAFE_TOOLFORGE_SUBPROCESS=1` | Tool exec sandboxed; code-node fail-closed; ToolForge Gate 2 unlocked |
| CI release build | `SAGE_REQUIRE_WASM=1` (maturin step) | Build aborts if `rustpython.wasm` missing |

## References

- ADR-013 - wasm sandbox default (2026-04-22)
- R4 - code-node fail-closed (2026-04-28)
- `sage-python/src/sage/sandbox/errors.py` - `SandboxUnavailable` typed exception
- `sage-python/src/sage/tools/forge.py` - ToolForge Gate 2 fail-closed
