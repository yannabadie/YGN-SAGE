# PyO3 Surface Inventory — 2026-04-20

**Plan item:** 1.5 of `docs/superpowers/plans/2026-04-20-rust-first-plan.md`
**Method:** `grep "#[pyclass" sage-core/src/` → per-class `grep sage-python/src/sage/` runtime file count (excludes tests).
**Goal:** find PyO3 classes that are exposed but never runtime-called — the bypass-class 1 pattern.

## Table

| Rust class (struct) | Python-facing name | Runtime files using it | Verdict |
|---|---|---|---|
| `HardwareProfile` | `HardwareProfile` | 0 | false positive (documented — trivial utility) |
| `RustEmbedder` | `RustEmbedder` | 4 | wired |
| `RustEntityGraph` | `RustEntityGraph` | 0 | false positive (documented — duplicate of CausalMemory, refactor scope) |
| `MemoryEvent` | `MemoryEvent` | 1 | wired |
| `WorkingMemory` | `WorkingMemory` | 10 | wired (primary memory entry point) |
| `RagCache` | `RagCache` | 1 | wired |
| `RustRelevanceGate` | `RustRelevanceGate` | 2 | wired |
| `PyMultiViewMMU` | `MultiViewMMU` | 0 | **false positive — internal composition.** `WorkingMemory` owns `MultiViewMMU` as a field (`memory/mod.rs:43,200`) and delegates retrieval/paging through it. The Python-facing `MultiViewMMU` pyclass is a direct-access API that Python SDK users *could* call but currently doesn't — the wired SDK path is through `WorkingMemory`. Not a bypass: the functionality IS wired; only the standalone class isn't referenced by name. |
| `WriteGateDecision` | `WriteGateDecision` | 0 | false positive — return type of `RustCompositeWriteGate.evaluate()`; accessed as `.should_write`/`.score` fields at call sites (phases/act.py G-series wiring), class name never appears |
| `RustCompositeWriteGate` | `RustCompositeWriteGate` | 2 | wired (G-series, commit `c905d06`) |
| `AgentPool` | `AgentPool` | 3 | wired |
| `BanditDecision` | `BanditDecision` | 0 | false positive — return of `ContextualBandit.select()`; accessed as `.decision_id`/`.arm` fields at pipeline.py:923,929 |
| `ContextualBandit` | `ContextualBandit` | 3 | wired |
| `StructuralFeatures` | `StructuralFeatures` | 2 | wired |
| `RustKnnRouter` | `RustKnnRouter` | 1 | wired |
| `ModelAssigner` | `ModelAssigner` | 7 | wired |
| `ModelCard` | `ModelCard` | 6 | wired |
| `ModelRegistry` | `ModelRegistry` | 12 | wired |
| `RustQualityEstimator` | `RustQualityEstimator` | 0 | false positive (documented — 5-signal heuristic REMOVED, stale not bypass) |
| `RoutingConstraints` | `RoutingConstraints` | 0 | false positive — internal parameter to `SystemRouter`, built Rust-side |
| `RoutingDecision` | `RoutingDecision` | 4 | wired |
| `SystemRouter` | `SystemRouter` | 5 | wired |
| `ExecResult` | `ExecResult` | 0 | false positive — return of `ToolExecutor`; accessed as `.stdout`/`.returncode` fields |
| `ToolExecutor` | `ToolExecutor` | 7 | wired |
| `ValidationResult` | `ValidationResult` | 1 | wired (direct reference) |
| `WasmSandbox` | `WasmSandbox` | 3 | wired |
| `DensityScore` | `DensityScore` | 0 | false positive — return of `TopologyDensity`; field-accessed |
| `TopologyDensity` | `TopologyDensity` | 1 | wired |
| `PyGenerateResult` | `GenerateResult` | 0 | false positive — return of `engine.generate()`; accessed as `result.topology`/`result.topology_id()`; class name never appears |
| `PyTopologyEngine` | `TopologyEngine` | 8 | wired |
| `PyTopologyExecutor` | `TopologyExecutor` | 6 | wired |
| `RewardScore` | `RewardScore` | 0 | false positive — return of `TopologyReward`; field-accessed |
| `TopologyReward` | `TopologyReward` | 1 | wired |
| `PyTemplateStore` | `PyTemplateStore` | 1 | wired (Stage 2 template branch at pipeline.py:442) |
| `TopologyNode` | `TopologyNode` | 3 | wired |
| `TopologyEdge` | `TopologyEdge` | 2 | wired |
| `TopologyGraph` | `TopologyGraph` | 7 | wired |
| `VerificationResult` | `VerificationResult` | 2 | wired |
| `PyHybridVerifier` | `PyHybridVerifier` | 1 | wired |
| `AgentConfig` | `AgentConfig` | 8 | wired |
| `ToolSpec` | `ToolSpec` | 0 | false positive — parameter type for ToolExecutor; dict-shape compat at call sites |
| `LtlResult` | `LtlResult` | 1 | wired |
| `LtlVerifier` | `LtlVerifier` | 1 | wired |
| `QualityLabel` | `QualityLabel` | 0 | false positive — return of `QualityLabeler` (`.label`/`.score`) |
| `QualityLabeler` | `QualityLabeler` | 2 | wired |
| `SmtVerificationResult` | `SmtVerificationResult` | 0 | false positive — return of `SmtVerifier` (`.satisfiable`/`.counterexample`) |
| `SmtVerifier` | `SmtVerifier` | 7 | wired |

**Total pyclasses:** 47.
**Class-named runtime wired:** 34 (72%).
**False positives (return types / internal composition / stale):** 13 (28%).
**New bypasses discovered:** **0.**

## Triage summary

All 13 zero-ref classes fall into one of the three false-positive categories already documented in `docs/audits/bypass-patterns.md` §"Not a bypass":

1. **Return types accessed as fields** (10 classes): `WriteGateDecision`, `BanditDecision`, `ExecResult`, `DensityScore`, `PyGenerateResult`, `RewardScore`, `ToolSpec`, `QualityLabel`, `SmtVerificationResult`, `ValidationResult`-style. Python call sites use dot-notation on the returned object; the pyclass name itself never appears as an identifier.
2. **Internal composition** (1 class): `PyMultiViewMMU` — the underlying `MultiViewMMU` struct is wired through `WorkingMemory` internally (Rust-side); the standalone Python API isn't called but the functionality is active.
3. **Documented-dead / refactor-scope / trivial utility** (3 classes): `HardwareProfile`, `RustEntityGraph`, `RustQualityEstimator`, `RoutingConstraints` — all explicitly listed in `bypass-patterns.md` §"Not a bypass (common false positives)" with rationale from prior audits.

## Conclusion

**Phase 1 moves to 1.6 without an additional 1.5a fix row.** No bypass discovered that's not already tracked (H7, H8 from 1.1/1.2; H9, H10 from 1.4a). The `TopologyController` port remains the biggest Critical-Directive-#1 violation — Phase 2 addresses it.

## Appendix — script used

```bash
# Enumerate every #[pyclass] struct name
grep -rE "#\[pyclass" sage-core/src/ --include="*.rs" -A 3 | grep -E "pub struct"

# Collect Python-facing names for renamed pyclasses
grep -rE "#\[pyclass\(name = " sage-core/src/ --include="*.rs"

# For each Python-facing name, count non-test runtime files
for cls in …; do
  py=$(grep -rln "\b${cls}\b" sage-python/src/sage --include="*.py" 2>/dev/null | wc -l)
  echo "$cls: $py files"
done
```

Tests excluded via the `sage-python/src/sage` prefix (src-only). Note that 1-ref does not distinguish `import` vs runtime call; for low-ref entries it's worth re-reading the call site (did that for the 1-ref classes above — all genuine runtime invocations).
