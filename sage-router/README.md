# sage-router

**Standalone Cost-Aware S1/S2/S3 LLM Routing**

`sage-router` is a **standalone Python package** extracted from the YGN-SAGE
runtime, packaged so it can be installed and used without pulling in the
full SAGE Rust core. It provides cost-aware S1/S2/S3 cognitive routing
based on structural features and (optionally) kNN over task embeddings.

> **Status (2026-05-04, cycle-10):** This package is **not used by the
> canonical `sage-python` runtime**. The runtime imports its routing from
> `sage-python/src/sage/strategy/` (which is the active version, with Rust
> acceleration via `sage_core` when available). `sage-router` is a
> "lift-and-shift" subset of those modules, kept here for users who want
> just the routing logic without the rest of YGN-SAGE.
>
> Decision pending (cycle-10/11): whether to publish this as a separate
> PyPI package (`sage-router`) under the B4 wheels initiative, or to fold
> back into `sage-python` as the canonical home and retire this directory.

## Relationship to `sage-python`

| File | `sage-router/src/sage_router/` | `sage-python/src/sage/strategy/` |
|---|---|---|
| `adaptive_router.py` | ✓ standalone copy (Python only) | ✓ runtime copy + Rust fallback via `sage_core::AdaptiveRouter` |
| `knn_router.py` | ✓ standalone copy (numpy optional) | ✓ runtime copy + Rust kNN via `sage_core::RustKnnRouter` (92% GT) |
| `metacognition.py` | ✓ standalone copy (`ComplexityRouter`, heuristic 34% GT) | ✓ runtime copy (Priority-3 emergency fallback only — see CLAUDE.md directive #4) |
| `quality_estimator.py` | ✓ minimal heuristic version | ✗ runtime version is in `sage/quality_estimator.py` (Z3-formal or ONNX-learned, abstain on absence) |
| `structural_features.py` | ✓ standalone copy | ✓ runtime copy |

The two trees are **near-duplicates by design**: `sage-router` is the
"works without Rust core, zero deps" path; `sage-python/src/sage/strategy/`
is the "best-effort with Rust acceleration when available" path that the
full YGN-SAGE pipeline actually uses.

## Install (standalone use only)

```bash
# Zero-dep core
pip install -e sage-router/

# With kNN routing (numpy)
pip install -e "sage-router/[knn]"

# Dev tools
pip install -e "sage-router/[dev]"
```

For the full YGN-SAGE runtime — including Rust kNN, Rust SystemRouter,
ContextualBandit, TopologyEngine, sandboxed execution, etc. — install the
parent package per the root `README.md`:

```bash
cd sage-core && maturin develop --features smt,onnx
cd sage-python && pip install -e ".[all,dev]"
```

## Test

```bash
cd sage-router && pytest tests/
```

(216 LOC of standalone tests. Independent from the YGN-SAGE main test
suite — running `pytest` from repo root will not include them.)

## Why a separate package?

The YGN-SAGE main install requires:
- Rust toolchain (`maturin`, `rustc` 1.94+, `cargo`)
- Multi-feature build (`--features smt,onnx`)
- Embedded `rustpython.wasm` artifact for the sandbox

For users who only want **cognitive routing** (classify a task → which
LLM tier? which model? what's the budget?) and don't need the full
multi-agent runtime, this overhead is too much. `sage-router` exposes
just the routing surface, in pure Python, with optional numpy.

## Status disambiguation

This package is:
- ✅ **Real code, not zombie**: 1374 LOC across 6 modules + tests, with
  its own `pyproject.toml` (`name = "sage-router"`, version `0.1.0`).
- ❌ **NOT imported by `sage-python` or `sage-core`**: `grep -r "sage_router"`
  in those trees returns 0 matches. The two trees evolve independently.
- 🟡 **Not yet published to PyPI**: install is `pip install -e
  sage-router/` from the monorepo. PyPI publication is part of cycle-10
  P5 (B4 wheels) decision.
- 🟡 **Not tied to `sage-discover/`**: that's a separate adjunct package
  (`name = "sage-discover"`, depends on `ygn-sage>=0.1.0`).

If you read this file and the parent README.md, you should be able to
answer "is `sage-router/` part of the canonical YGN-SAGE runtime?" with
**no** — without grep — and "is it dead/zombie code?" with **also no**.

## Cycle-10 disposition

Per `cgpro_kimi_audit_response_20260504` and the cycle-10 plan at
`.claude/plans/2026-05-04-cycle-10-verified-runtime-release-preview.md`,
the decision tree for this package is:

1. If a downstream consumer (other than `pip install -e sage-router/`
   for testing) emerges before cycle-11: keep, add CI, publish to PyPI
   alongside the B4 ygn-sage wheels.
2. Otherwise: fold the modules back into `sage-python/src/sage/strategy/`
   as the canonical home and remove this directory in cycle-11.

This README itself is the cycle-10 P2 deliverable: making the
disposition explicit so an external auditor cannot mistake `sage-router/`
for either zombie code or canonical runtime.
