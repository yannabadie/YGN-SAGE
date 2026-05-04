# Cycle-10 P5 (B4 wheels) — local Windows smoke evidence

**Date:** 2026-05-04
**Repo HEAD at smoke:** `a4fd39db` (cycle-10 P0-P9 + P4 closure landed)
**Operator:** Claude Opus 4.7 (1M context) per cgpro_kimi_audit_response_20260504 cycle-10 P5 first-salve.

## Goal

Validate the wheel build path on at least one of the three target platforms
(Linux x86_64, Windows MSVC, macOS aarch64) before committing the
`workflow_dispatch`-only `.github/workflows/wheels.yml` matrix. Per
advisor 2026-05-04: "smoke-test on at least one platform locally before
committing the multi-OS workflow — confirms one of three OS targets works."

## Setup

- Host: Windows 11, MSVC build chain present
- Python: 3.13 (CPython)
- Rust: stable, cargo + maturin available
- Working directory: `sage-core/`
- Build command:
  ```
  python -m maturin build --release --out target/wheels --strip
  ```
  Default features pulled from `sage-core/pyproject.toml` `[tool.maturin]`:
  `sandbox + cranelift + tool-executor + cognitive`. `smt` and `onnx` are
  intentionally omitted from the default wheel to keep size under PyPI's
  100 MB hard cap.

## Build result

```
Compiling cranelift-native v0.131.0
Compiling cranelift-frontend v0.131.0
Compiling wasmtime-internal-unwinder v44.0.0
Compiling wasmtime-internal-cranelift v44.0.0
Compiling wasmtime v44.0.0
Compiling wiggle v44.0.0
Compiling wasmtime-wasi-io v44.0.0
Compiling wasmtime-wasi v44.0.0
Finished `release` profile [optimized] target(s) in 1m 34s
📦 Built wheel for CPython 3.13 to target/wheels\sage_core-0.1.0-cp313-cp313-win_amd64.whl
```

- **Wheel name:** `sage_core-0.1.0-cp313-cp313-win_amd64.whl`
- **Size:** **31.8 MB** (well under PyPI 100 MB cap; embedded
  `rustpython.wasm` ~37 MB upstream + cranelift ~15 MB + sage-core code,
  all stripped)
- **Build time (warm):** ~1m 34s on Windows host with cargo cache
- **Build time (cold):** ~8-10 min including embedded RustPython wasm
  compile (one-time per submodule SHA)

## Fresh venv install + import smoke

```bash
python -m venv /tmp/p5-smoke
/tmp/p5-smoke/Scripts/python.exe -m pip install --upgrade pip
/tmp/p5-smoke/Scripts/python.exe -m pip install \
  C:/Code/YGN-SAGE/sage-core/target/wheels/sage_core-0.1.0-cp313-cp313-win_amd64.whl
```

Both succeeded with **no Rust toolchain available in the venv**. This is
the critical proof: the wheel is self-contained.

## Symbol surface assertion (default features wheel)

```python
import sage_core

default_required = [
    'TopologyEngine', 'SystemRouter', 'ContextualBandit',
    'ModelAssigner', 'ModelRegistry', 'MultiViewMMU',
    'WorkingMemory', 'ToolExecutor',
]
# All 8 present. ✓

# smt-gated symbols (oxiz dep) MUST be absent in default wheel
opt_in_smt = ['HybridVerifier', 'LtlVerifier']
# Both absent. ✓
```

Output:
```
SMOKE OK: 8/8 default symbols present
smt gate: 0/2 smt symbols (expected 0 for default wheel)
default attrs sample: ['AgentConfig', 'AgentPool', 'AgentStatus',
                       'BanditDecision', 'CognitiveSystem',
                       'ContextualBandit', 'DensityScore', 'ExecResult']
```

## What this proves

- ✅ `sage-core/pyproject.toml` is correctly wired for `maturin build`
- ✅ Default-features wheel is functional (TopologyEngine, SystemRouter,
  ContextualBandit, etc. all importable) on Windows MSVC + Python 3.13
- ✅ Wheel is self-contained (installs in fresh venv with no Rust
  toolchain)
- ✅ Wheel size sustains PyPI 100 MB hard cap (31.8 MB)
- ✅ `smt` gating works correctly (opt-in symbols absent from default
  wheel — important for B4 size budget AND for the "planned, not shipped"
  framing of `RustLearnedQualityEstimator` per cycle-10 P7)

## What this does NOT prove (cycle-11 follow-up)

- ❌ Linux x86_64 wheel build (no local Linux runner)
- ❌ macOS aarch64 wheel build (no local Mac)
- ❌ `pip install ygn-sage` end-to-end (requires `sage-core` on PyPI;
  the dep declaration in `sage-python/pyproject.toml` already exists
  per cycle-9 cgpro round-1 fix, but the actual PyPI publish is
  out of P5 first-salve scope)
- ❌ smt+onnx variant smoke (different wheel artifact, not built locally)
- ❌ Wheel under cibuildwheel/manylinux2014 environment (the local
  build is a "naked" Windows MSVC wheel)

These gaps are tracked in `.github/workflows/wheels.yml`'s acceptance
gate comment block. The first `workflow_dispatch` run on the multi-OS
matrix produces evidence for the missing platforms.

## Path to PyPI release (cycle-11+)

Per the workflow file's trailing comment block:

1. Successful `workflow_dispatch` run on all 6 matrix entries
2. Smoke passing on at least Linux + Windows
3. Operator review of artifact sizes (PyPI 100 MB cap; smt+onnx variant
   may exceed)
4. TestPyPI dry-run via separate `release-test` workflow
5. Real PyPI publish via tag-triggered release workflow

Each step is a separate cycle-11 commit, not a P5 first-salve concern.

## Related

- `sage-core/pyproject.toml` (created cycle-10 P5)
- `.github/workflows/wheels.yml` (workflow_dispatch draft, cycle-10 P5)
- Cycle-10 plan: `.claude/plans/2026-05-04-cycle-10-verified-runtime-release-preview.md`
- cgpro_kimi_audit_response_20260504 (cycle-10 plan + advisor framing)
