# ALIRE + ALIRE2 — audit verification against main @ 4704b51

**Date:** 2026-04-23
**Audit docs verified:**
- `ALIRE.md` — inspected repo state at commit `44a157c` (2026-04-22)
- `ALIRE2.md` — inspected repo state at commit `ef2100b` (2026-04-23, earlier)
**Main commit at verification time:** `4704b51`
**Post-fix main commit after this artefact lands:** `2bd966c` (includes A0a/A0b/A0c/A0d)
**Protocol:** `PROMPT.md` — Phases 1-4 executed in-session with plan-mode
gating; Context7 used for OpenTelemetry semantic-conventions only
(already cited in `roadmap.md` B1); no external-oracle consultation
because the verified gaps had clear code-level fixes.

## Synthèse

Total assertions verified: **9** (A…I) across the two audit tables.

| Category | Count |
|---|---|
| ✅ Confirmed (live on today's main) | 4 (A, C, D, H) |
| ⚠️ Partially true / nuanced | 2 (B, I) |
| ✅ Accurate description but **orphaned** post ADR-013 §5 flip | 3 (E, F, G) |
| ❌ Refuted | 0 (but 1 claim inside I refuted) |
| 🔍 Not verifiable in-session | 0 |

**Headline:** of the 4 live high-severity gaps, 3 (A0a, A0b, A0c) are
addressed by shipped commits in this same session; the 4th (C — Rust
controller scaffolded top-level) is filed as B8 in `roadmap.md` for
the next cycle (populate vs de-scope is an open scoping question).

## Matrice des problèmes

### Live on main @ 4704b51 — high severity

| ID | Claim | Severity | Evidence | Status after this session | Remediation commit |
|---|---|---|---|---|---|
| **A** | Shared mutable `AgentLoop` state in `pipeline.py` bypass — 10 fields mutated, only 3 restored. | High | `pipeline.py:1167–1296` mutations; `finally` at 1292–1296 restores 3/10. No concurrency isolation. | ✅ **Fixed (targeted)** | `9067be5` (A0a). Full refactor to per-run immutable context = B9 in roadmap. |
| **C** | `RustTopologyController.evaluate_and_decide` returns `None` — "scaffold stub". | High | `sage-core/src/topology/controller.rs:189–197` body is `let _ = (node_idx, result, task); None`. Per-path methods (check_empty_error_reroute, check_quality_cascade, etc.) at lines 246+ ARE populated and ARE called from Python. | ✅ **De-scoped + deleted 2026-04-23 (B8 closure)** | Stub deleted; ADR-012 amended; regression guard test added. Advisor + codex gpt-5.4 high-reasoning converged on de-scope. |
| **D** | Fail-open write gate + non-blocking verification (no strict mode). | High | `pipeline.py:172` ungated-on-init-failure log; `pipeline.py:1025–1058` SAT fail sets `ctx.verification_passed=False` then continues; `pipeline.py:1156` emits `EXECUTE_UNVERIFIED`. | ✅ **Fixed** | `2bd966c` (A0b — `SAGE_STRICT_GOVERNANCE=1`). |
| **H** | `Tool.execute()` leaks full traceback to model-visible output. | Medium-High | `sage-python/src/sage/tools/base.py:24–33` returns `f"Error: {type(e).__name__}: {e}\n{traceback.format_exc()}"`. | ✅ **Fixed** | `684bb17` (A0c — `log.exception` + type:message only). |

### Partial / nuanced

| ID | Claim | Verdict | Notes |
|---|---|---|---|
| **B** | `contracts/policy.py` is a stub "after training removal". | ⚠️ Partially true | File has working `PolicyVerifier` with `check_info_flow`, `check_fan_limits`, `verify_all` over DAG nodes — not a `pass`-stub. Referenced by `boot_pipeline.py` + `executor.py`. The "stub" framing is about the removed training-time policy verification, not the file being empty. Not a high-severity remediation. Keep as-is; the ToolPolicy capability manifest (B3) is the larger direction. |
| **I** | QualityEstimator contains "heuristic extraction" + ONNX not shipped. | ⚠️ Half-right, half-refuted | **ONNX not shipped: confirmed.** `quality_estimator.py:44–58` explicit comment. No `.onnx` artefact in repo. **Heuristic extraction: refuted.** No regex / length / keyword signals in the file; path is pure formal (Z3) + learned (when present) + None abstention. Doc sync: 6 docs still implied the ONNX path was active — fixed in `bf220e0` (A0d). |

### Accurate but orphaned post ADR-013 §5 flip

The 2026-04-22 sandbox §5 flip (`c2113d8`) moved `validate_and_execute`
off the subprocess path. ALIRE2's findings about `sandbox/manager.py`,
`isolated_executor.py`, and `sage-core/src/sandbox/subprocess.rs`
describe code that is accurate-as-described but no longer on the
default call path.

| ID | Claim | Orphaned? | Notes |
|---|---|---|---|
| **E** | `sandbox/manager.py` uses `create_subprocess_shell`. | Yes | Lines 129/150/174 still use shell invocation. `allow_local=False` default + Wasm-first priority means it's dead code on the happy path. Not a live vulnerability; ticket for deletion separately. |
| **F** | `isolated_executor.py` falls back to plain subprocess on non-Linux. | Yes | Only reachable via `manager.py._execute_local(allow_local=True)`. Not called by the Rust ToolExecutor path. |
| **G** | `sage-core/src/sandbox/subprocess.rs` provides timeout isolation only. | Gated | Docstring is honest. Reachable only via `execute_raw + SAGE_UNSAFE_RAW_EXEC=1`. Working as designed. |

## Plan d'action (pointer into roadmap.md)

Implementation items from ALIRE2 triage (this session):

| Item | Roadmap horizon | Status |
|---|---|---|
| A0a — AgentLoop bypass restore all 10 fields | A0a | ✅ shipped `9067be5` |
| A0b — SAGE_STRICT_GOVERNANCE fail-closed mode | A0b | ✅ shipped `2bd966c` |
| A0c — Tool.execute() traceback redaction | A0c | ✅ shipped `684bb17` |
| A0d — DistilBERT-ONNX-not-shipped docs sweep | A0d | ✅ shipped `bf220e0` |
| B8 — RustTopologyController de-scope (delete stub + ADR-012 amendment) | Horizon B | ✅ closed 2026-04-23; advisor + codex gpt-5.4-high converged |
| B9 — Per-run immutable context (AgentLoop refactor) | Horizon B | Pending; A0a is the targeted interim |
| E/F/G cleanup (delete orphan sandbox/manager + isolated_executor) | Horizon B / tidy | Pending; usage search needed first |

## Post-advisor blind-spot verifications (2026-04-23)

Advisor flagged three things to verify before calling A0 done. Verdicts:

1. **A0c completeness — NO additional leak sites.** Grepped
   `traceback\.format_exc|format_exc\(` across `sage-python/src`. Two
   hits: `eval_protocol.py:195` in `ErrorCapture.record` stores the
   traceback in a dataclass field consumed by
   `_log.warning`/`print` (operator paths, stdout); `swebench_bench.py:1381`
   logs the traceback via `log.error` (operator path). Neither reaches
   the model-visible `ToolResult.output`. A0c is complete.
2. **A0a mutation test added.** The original `finally`-restoration
   tests would have passed even if the bypass block accidentally
   dropped all its mutations. New test
   `test_bypass_path_actually_mutates_during_run` captures
   agent_loop state from *inside* `agent_loop.run` via a `side_effect`
   and asserts `max_steps`/`validation_level`/`stall_after_tool_steps`
   /`_skip_routing`/`_current_topology`/`write_gate`/`gate_current_task`
   take their bypass-path values during the run — not their pre-bypass
   sentinels. Closes the "restore → no-op if mutation is gone" gap.
3. **A0b emit-vs-raise ordering — sync, emit-before-raise.**
   `Pipeline._emit` is a sync method that calls
   `event_bus.emit(AgentEvent(...))` synchronously. The strict-mode
   branch at `pipeline.py:1194-1201` calls `self._emit("EXECUTE_HALTED_UNVERIFIED", …)`
   then `raise RuntimeError(...)`. The event is guaranteed to land
   before the exception propagates. No async/await concern.

## Divergences with the original audits

Three points where this verification disagrees with the source audits:

1. **ALIRE2 §6 "heuristic extraction in QualityEstimator" (claim I):
   refuted.** The file has NO in-tree heuristic signals. The architecture
   is Z3 + (optional ONNX) + None abstention. The audit may have been
   looking at a pre-2026-Q1 version or at a different file (e.g. routing
   heuristics) and mis-attributed.

2. **ALIRE2 §6 sandbox claims E/F/G re-classified as "orphaned post
   ADR-013", not "live risks".** The audit inspected commit `ef2100b`
   which is AFTER the §5 flip (`c2113d8` on 2026-04-22). The subprocess
   paths are still in the tree but not reachable on the default call
   path. The audit reads them as risks because the file-level comments
   describe dangerous behaviour; the audit missed that `ToolExecutor`'s
   default path now hard-fails rather than falling through to
   subprocess. Their remediations "replace shell-based execution" /
   "fail-closed on non-Linux" were already DONE, just in a different
   shape than the audit expected. The three orphan files should be
   deleted to remove the maintenance wart.

3. **ALIRE2 §5 "Rust-first runtime overstated" (claim C): scope
   narrowed.** The top-level orchestration method
   `evaluate_and_decide` IS a scaffold returning `None`. But the
   per-path methods (empty-error-reroute, quality-cascade, parallel-
   inconsistency, etc.) ARE populated and ARE called from Python's
   `TopologyController`. So "Rust-first" is true for the individual
   decision primitives; what's missing is the top-level orchestration
   that would compose them in Rust rather than Python. ADR-012's
   claim "Rust-primary since 2026-04-20" is factually correct for
   each decision path but misleading if read as "complete Rust
   control plane". B8 will either finish the orchestration or
   update the ADR to match reality.

## Notes on unexplored audit items

ALIRE (not ALIRE2) has additional items I did not re-verify this
session because they were already addressed in earlier sessions
(dangerous_tools flip, ExoCortex default-store leak, sandbox
three-layer wording, etc.) or are already tracked in `roadmap.md`
Horizons B/C (OpenTelemetry GenAI spans, durable trace/replay,
ToolPolicy capability manifest, platform wheels). This artefact is
scoped to ALIRE2's new-since-ALIRE claims.

## References

- ALIRE.md — audit at commit `44a157c` (2026-04-22)
- ALIRE2.md — audit at commit `ef2100b` (2026-04-23)
- `roadmap.md` — horizon-pacing backlog with A0/B/C items
- PROMPT.md — audit protocol that drove this pass
- `docs/benchmarks/2026-04-23-track3-closeout.md` — complementary Track 3 close-out from the same day
- Commits shipped this session in response to ALIRE2:
  - `684bb17` fix(tools): redact Tool.execute traceback (A0c)
  - `bf220e0` docs: caveat DistilBERT ONNX as not-shipped (A0d)
  - `9067be5` fix(pipeline): restore all 10 mutated AgentLoop fields (A0a)
  - `2bd966c` feat(pipeline): SAGE_STRICT_GOVERNANCE fail-closed (A0b)
