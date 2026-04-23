# YGN-SAGE roadmap

**Last updated:** 2026-04-23
**Scope:** forward-looking work surfaced by the 2026-04-23 session (Track 2+3
close-out, wasm JIT cache, pre-emission diff-context verifier, ALIRE audit
triage). Not a long-term strategy doc — a living backlog grouped by
expected time horizon. Priorities inside each horizon ordered by
impact-over-effort.

Reference frames:
- `ALIRE.md` — external audit of commit `44a157c` (2026-04-22). Several of
  its critical items landed this week (dangerous_tools flip, subprocess-
  fallback docs sweep, ADR-013); the remaining items are in this roadmap.
- `docs/benchmarks/2026-04-23-track3-closeout.md` — the Track 3 breadcrumbs
  pinned there map to items below.
- `docs/superpowers/specs/2026-04-23-diff-context-verifier-design.md` —
  spec that drove the 2026-04-23 observe-mode ship; repair-mode
  implementation is in the short-term section below.

---

## Horizon A — short-term (next 1-2 weeks)

### A0 — ALIRE2 triage remediations (2026-04-23)

The 2026-04-23 ALIRE2 verification (`docs/audits/2026-04-23-alire-verification.md`)
confirmed 4 high-severity live gaps on main. All 4 were shipped the
same session as small, reviewable commits — kept here so the audit
trail is explicit.

| Item | Shipped commit | Note |
|---|---|---|
| **A0a** Restore all 10 mutated AgentLoop fields in the pipeline bypass `finally`. Targeted fix; full refactor = B9. | `9067be5` | `pipeline.py` snapshots `_orig_bypass_state` dict pre-mutation; finally restores all 10 (happy + exception). +2 regression tests. |
| **A0b** `SAGE_STRICT_GOVERNANCE=1` turns the write-gate-init-failure and verification-failed paths into hard aborts (emit `EXECUTE_HALTED_UNVERIFIED`). Default off; byte-identical. | `2bd966c` | `pipeline.py` + `_is_strict_governance()` helper. +16 regression tests. |
| **A0c** Redact raw traceback from `Tool.execute()` model-visible output (info-leak fix, ALIRE2 §6). | `684bb17` | `tools/base.py` returns type:message; `log.exception` keeps operator-side visibility. +3 regression tests. |
| **A0d** Caveat the "DistilBERT ONNX QualityEstimator" claim in 6 docs — active backend is Z3 QualityLabeler + abstention; ONNX artefact NOT shipped. | `bf220e0` | README.md, pillars.md, results.md, methodology.md, Obsidian Papers/CascadeRouting.md + Architecture/Pipeline.md. |

### A1. Accumulate observe-mode data across opportunistic SWE-bench smokes
**Why:** the 2026-04-23 observe smoke (N=10) gave us 2 flagged patches.
Repair-mode flip needs ≥10 flagged + ≥10 clean observations to discriminate
signal from false positives at a meaningful SE. Dedicated smokes cost
$5-10 apiece; instead, **every** future SWE-bench run should opt into
observe mode passively.

**Concrete action:** update smoke invocations in docs + scripts (already
done in `.claude/rules/development.md` + `CLAUDE.md`) so new runs include
`SAGE_DIFF_VERIFIER_MODE=observe`. Wait for 3-4 smokes to accumulate the
sample.

**Done when:** 20+ observe-annotated tasks sit in `docs/benchmarks/`
with at least 10 PATCH entries among them.

### A2. Investigate the 20%+ fast-abort rate on SWE-bench generation
**Why:** 2/10 tasks in the observe smoke aborted in < 60 s with 0 tool
calls (astropy-14182, astropy-7746). Earlier smokes showed similar
fast-abort patterns. If 20% of budget silently evaporates into early
fails, the observe/repair data is small-N for everything else.

**Concrete action:** pull the gen log for a fast-abort case. Is it
provider circuit-breaker? SSL? Classification cold-start? Once root-
caused, decide whether it needs a fix or is acceptable noise.

**Cost:** ~2 h of log-reading + 1 targeted smoke. Essentially zero $.

### A3. Repair-mode implementation (conditional on A1 data)
**Why:** spec § "Validation plan" — repair mode feeds the diff-verifier
mismatch diagnostic to an LLM one-shot repair. The observe smoke confirmed
the mismatch signal is clean (zero false positives on two patches);
waiting on more sample before flipping.

**Concrete action when gate passes:** implement the
`_repair_with_verifier_feedback` stub spec'd in the design doc; extend
`test_swebench_emission_wiring.py` with a repair-mode wire test; run
paired N=50 smoke (observe-only vs observe+repair).

### A4. Public-claims reconciliation (README ↔ PyPI)
**Why:** ALIRE flagged a three-way divergence (README/commits/PyPI) on
test counts. README has been refreshed (commit `be2d3fc`), but the
published PyPI `ygn-sage 0.1.0` still carries the March description.
Publishing a new PyPI patch release (`v0.1.1`) would sync, at the cost
of a minor release cycle.

**Concrete action:** bump patch version, refresh classifiers, push to
PyPI via the existing release workflow if one exists (grep for it)
or add a brief `docs/RELEASE.md`.

### A5. Fix the `ss.F.` anomaly captured in `tasks/bsc5mma67.output`
**Why:** the test-ignore ceiling drifted below tests and landed (commit
`5efdd42`). The original failing run wasn't labelled and ceased to exist
in the task runtime — attribute the output to a session context for
next time, or wrap long-running pytest invocations in a `TaskCreate` so
the task ID survives.

**Cost:** trivial, mostly a workflow hygiene task.

---

## Horizon B — medium-term (next 1-2 months)

### B1. OpenTelemetry GenAI span integration
**Why:** EventBus is in-process and lossy under backpressure (ALIRE
finding). OpenTelemetry published a GenAI semantic-conventions spec
(Development stability as of 2026-Q1) covering LLM request/response
attributes (`gen_ai.provider.name`, `gen_ai.operation.name`,
`gen_ai.request.model`, `gen_ai.usage.{input,output}_tokens`,
`gen_ai.response.finish_reasons`). That's the standard contract we'd
emit.

**Concrete action:** pick an OTel Python SDK (opentelemetry-sdk +
opentelemetry-exporter-otlp). Wire provider-side emission in
`sage.providers.pydantic_ai_provider` (and deprecated openai_compat
while it lives). Initial target: every LLM call emits a span with the
minimum-required attributes above. Deferred: tool-call spans, MCP
conventions. Docker harness span path is separate.

**Dependencies:** none blocking. Would benefit from B2 (durable store)
to land first, but they can proceed in parallel.

### B2. Durable trace + deterministic replay harness
**Why:** ALIRE high-severity item, and it's the prerequisite for any
serious formal-verification work (ALIRE's "runtime assurance" direction).
Every LLM call + tool call + topology decision + memory write gets a
trace ID and payload hashed; replay reconstructs the exact decision
sequence under mocked providers/tools.

**Concrete action:** start with an in-memory JSONL trace schema (one
line per event, typed). Ship the writer first; deserialisation + replay
later. Target: SWE-bench bench can reproduce a prior run's decisions
when replayed on mocks.

**Cost:** significant (~2-3 weeks). Defer if A1-A3 uncover a smaller
lever.

### B3. ToolPolicy capability manifest
**Why:** ALIRE critical item. Current `ToolRegistry` exposes
register/list/describe but no capability labels, side-effect contracts,
or approval requirements. That's a prompt-injection exposure.

**Concrete action:** new `ToolSpec` field `capabilities:
list[Capability]` + `data_access: DataScope` + `approval_required: bool`.
Policy check inserted in `execute_tool_call` before the tool runs.
Migrate all builtin tools (`execute_bash`, `read_file`, `search_repo`,
`apply_patch`, etc.) to declare capabilities. Default-deny on new tools
without a manifest.

**Dependencies:** none. Can ship incrementally (typed tools first, then
generated ToolForge tools).

### B4. Platform wheels with Rust core on PyPI
**Why:** ALIRE high severity. Current PyPI wheel is pure-Python;
`pip install ygn-sage` doesn't get the Rust extension. Either fix the
install story or clearly scope the PyPI package as "Python bindings —
build Rust from source".

**Concrete action:** extend the release workflow with a `maturin build
--release` matrix (Linux/macOS/Windows) and upload via `twine`. Include
the `rustpython.wasm` artefact (or build it in CI) so `SAGE_REQUIRE_WASM=1`
passes. Decide whether the wasm artefact goes in the wheel (37 MB bloat)
or a separate package (`ygn-sage-sandbox`).

**Cost:** 1 week. Mostly CI work.

### B5. CI job that builds wasm + enforces `SAGE_REQUIRE_WASM=1`
**Why:** the 2026-04-23 `SAGE_REQUIRE_WASM` gate (commit `cf188df`)
turns missing `rustpython.wasm` into a build error, but no CI job
exercises that path. A release pipeline that doesn't verify the wasm
artefact is the same failure mode in a nicer suit.

**Concrete action:** add a `.github/workflows/sandbox-build.yml` job
that clones RustPython, builds `wasm32-wasip1`, caches it, then runs
`SAGE_REQUIRE_WASM=1 cargo build --features sandbox`. ~20 min added to
CI (cacheable via `actions/cache`).

### B6. Fast-abort root cause fix (depends on A2 findings)
**Why:** whatever A2 surfaces, fixing the 20% fast-abort rate has
proportional payoff on every SWE-bench smoke.

### B8. Finish or de-scope the Rust controller orchestration entry
**Why:** ALIRE2 verification (A0a companion, 2026-04-23) confirmed
`RustTopologyController::evaluate_and_decide` returns `None` (scaffold
stub) in `sage-core/src/topology/controller.rs:189–197`. Per-path
methods (`check_empty_error_reroute`, `check_quality_cascade`,
`check_parallel_inconsistency`, etc.) ARE populated and ARE called
from Python's `TopologyController`. So "Rust-primary since 2026-04-20"
(ADR-012) is factually correct for each decision path, but misleading
if read as "complete Rust control plane".

Two options:

1. **Populate** the orchestration body so Rust composes the per-path
   methods itself — Python just consumes the `Option<Decision>`
   return. Frees Python from the adapter layer.
2. **De-scope** — remove or rename `evaluate_and_decide` to make the
   scaffold obvious, and update ADR-012 to say "Rust per-path checks
   with Python-side orchestration" instead of implying a single
   Rust entry point.

Gated on: do we have a real concurrency use case where the per-path
methods need top-level Rust orchestration? If not, de-scope is cheaper
and more honest.

**Estimated effort:** 1-2 days if populate; 2-3 hours if de-scope.

### B9. Per-run immutable execution context (AgentLoop refactor)
**Why:** A0a (shipped `9067be5`) fixes incomplete restoration of the
10 mutated fields in the bypass path's `finally`. That closes
state-bleed under serial nested calls, but NOT under concurrent ones —
two `run()` coroutines on the same singleton AgentLoop would still
race on the shared fields between the snapshot and the `finally`.

The architectural fix is stop mutating a shared object entirely: build
a fresh, immutable execution context per run. References ALIRE2 §4.

**Estimated effort:** 1-2 weeks. Don't start until A1/A3 (observe +
repair mode) stabilise; this is a big refactor and interleaving it
with Track 3 work multiplies risk.

### B7. Test count drift gate automation
**Why:** the `test_mypy_count.py` ceiling drifted by 5 between commits
over weeks (caught in this session). The ceiling-bump flow is manual
and error-prone.

**Concrete action:** add a pre-commit hook or CI step that, on ceiling
drift, auto-generates a commit that lists the new ignores and their
source commits for review. Or: replace the ceiling test with a
per-file baseline (count per module is allowed; new modules must
declare).

---

## Horizon C — longer-term (3-6 months)

### C1. Semantic LLM verifier (option C from the diff-context spec)
**Gated on:** diff-verifier repair mode shipping (A3) and NOT moving
resolved rate ≥ 4 pp at N=50.

**Why:** the 2026-04-23 Track 3.1b analysis found 3 orthogonal
semantic-miss failure modes (context hallucination, over-aggressive
scope, wrong-layer coercion). The content-verifier addresses mode 1.
Modes 2 and 3 would need an LLM pass asking "does this diff actually
fix the problem". Expensive (+ $0.02-0.10/task) but the only
prompt-level remediation for mode 3.

### C2. Domain-knowledge injection (Track 3 breadcrumb #2)
**Why:** django-10924's `FilePathField.formfield` wrong-layer fix
requires knowing about Django's migration autodetector semantics. Not
a library-API contract (so `lookup_library_docs` doesn't help),
not a research-paper question (so `search_exocortex` doesn't help).
Could be:
- domain-specific knowledge bases seeded at boot (Django, Flask,
  numpy, etc.)
- or a pre-emission "is there framework-specific convention I'm
  missing?" LLM meta-question

Open research question. Keep a backlog note; implement only if observed
more than once in tracers.

### C3. External benchmark harness (GAIA, AgentBench, τ-bench, SWE-bench-Live)
**Why:** ALIRE high severity. Internal benchmark docs are candid but
don't substitute for external baselines. GAIA, AgentBench, and
τ-bench are the credible frames for agent evaluation; SWE-bench-Live
is the contamination-resistant next generation of SWE-bench-Verified.

**Cost:** multi-week. Likely need a dedicated harness module + artefact
store.

### C4. Runtime assurance layer (ALIRE "highest-leverage change" tier)
**Why:** pre/post-conditions on every tool call, data-flow constraints,
policy enforcement before side effects. Builds on B3 (ToolPolicy) and
B2 (durable trace).

### C5. Model-checked orchestration specs (TLA+/Alloy)
**Why:** ALIRE "world-class" tier. Multi-agent coordination,
cancellation, retry semantics captured in a formal spec. Tests verify
the runtime traces match the spec.

**Cost:** very high. Consider only once B2 + B3 land; speculative
otherwise.

---

## Open questions (breadcrumbs)

These are design ambiguities without a decisive experiment yet. Kept
here so they don't get lost.

1. **Is 0.95 really the right fuzzy-threshold for the diff-context
   verifier?** Spec correction `3c3fc27` narrowed the fuzzy branch to
   whitespace-only equivalence, with 0.95 retained as an observability
   hint. If the observe bucket accumulates fuzzy-below-threshold cases
   that correlate with whitespace drift we didn't anticipate, the
   threshold may need tuning or the branch may need another narrowing.

2. **Is `search_exocortex` actually useful in SWE-bench?** Prompt
   hygiene commit `29987bc` reframed it from anti-affordance to
   positive use case, but a follow-up audit hasn't measured whether
   usage went from 0 calls to non-zero. Worth a one-line grep on the
   next SWE-bench gen log: does `tool.call name=search_exocortex`
   appear anywhere?

3. **Module::serialize cache vs wasmtime built-in cache.** The
   2026-04-23 JIT cache (commit `50b4ee8`) hand-rolls a single-file
   .cwasm artefact. wasmtime's `Config::cache_config_load_default()`
   is the official built-in alternative. If we find ourselves adding
   cache-management complexity (invalidation rules, concurrent-writer
   safety beyond the current tempfile+rename, multi-module caches),
   consider migrating to the built-in.

4. **Why 8/10 EMPTY is the norm on SWE-bench Lite smokes.** Covered
   by A2 above but worth repeating as a meta-question: is this a
   model/capability ceiling, an infra issue (circuit breaker), or a
   prompt issue? The answer shapes which roadmap items are
   prioritised next.

5. **Do we need a dedicated prompt-injection red-team corpus?** The
   2026-04-22 wasm sandbox red-team (40/40 passing) validated the
   capability layer; nothing equivalent exists for prompt injection
   via the agent's tool-call path. ALIRE flagged this as high; hasn't
   surfaced as a concrete incident yet.

---

## Dropped / superseded directions

* **SWE-bench parity smoke at N=50** — the `±2 pp` statistical gate was
  below the N=50 SE ceiling (noise floor ≈ 10 pp/task; combined arm-gap
  SE ≈ 2 pp at N=50). Replaced by the functional criterion (typed-only
  produces patches) which landed 2026-04-22.
* **Track 3.2 "read test files first" prompt addition** — invalidated
  by the Track 3.1 finding that agents DO read test files on all three
  tracers.
* **Track 3.5 dedicated N=50 paired smoke** — deferred at the 2026-04-23
  close-out; no prompt-level lever had a lift hypothesis worth the $30-50
  spend.
* **Noise-floor calibration via paired identical-config runs** —
  advisor correctly flagged this as "measurement theater"; noise floor
  can be computed post-hoc from any future N=50+ smoke via resampling.

* **ALIRE2 §6 sandbox claims E/F/G (subprocess-based execution paths)**
  — accurate-as-described in code but **orphaned** post ADR-013 §5
  flip (`c2113d8`, 2026-04-22). `sandbox/manager.py` +
  `sandbox/isolated_executor.py` remain in the tree as legacy fallback
  that is unreachable on the default path (`allow_local=False`, Wasm
  first, hard-fail if Wasm absent). `sage-core/src/sandbox/subprocess.rs`
  gated behind `SAGE_UNSAFE_RAW_EXEC=1`. Deletion of the Python-side
  orphan files is tidy-but-not-urgent — ticketed as a separate B-tier
  cleanup; see `docs/audits/2026-04-23-alire-verification.md` for the
  divergence note.

* **ALIRE2 §3 "heuristic extraction in QualityEstimator" (claim I)**
  — refuted by code inspection. No regex / length / keyword signals in
  `quality_estimator.py`. Path is Z3 + (optional ONNX) + None
  abstention. The ONNX-not-shipped half of the claim IS true (handled
  by A0d docs sweep, `bf220e0`).

---

## Horizon pacing

Roughly: two Horizon A items per week; one Horizon B item per month;
Horizon C items tracked but not worked on until a B item forces the
issue. Do not lump A and B in the same commit; ship them separately.
