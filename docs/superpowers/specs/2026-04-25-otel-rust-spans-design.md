# roadmap-B1.b: Rust spans bridge to OpenTelemetry — Design Spec

**Status:** Draft (2026-04-25)
**Sub-item of:** roadmap-B1 (OpenTelemetry GenAI spans)
**Predecessor commit:** `131de8aa` (B1 Python-side shipped)
**Author:** Claude Opus 4.7 + Yann Abadie
**Reviewers:** advisor (pre-approved autonomous chain), prior B1 spec at `2026-04-25-otel-genai-spans-design.md`

---

## 1. Goal

Make `sage-core` Rust hot paths visible in the **same OTel trace tree** as the
Python `sage.pipeline.run`. Today the 22 Rust files using `tracing::info_span!`
or `#[instrument]` are dark to OTel — `info_span!` macros register call sites
but no subscriber is installed in production builds, so spans are dropped on
the floor.

**Diagnostic class this unblocks:** roadmap-A2 needed a Stage-4 quality-cascade
timing breakdown to attribute the Kimi-400 → fallback-empty cascade. The cascade
runs in Rust (`RustQualityLabeler::label`, `ContextualBandit::sample`), and
cost 2 hours of log-mining on 2026-04-23. With B1.b, the same investigation
collapses to span-tree inspection (~5 minutes).

**Non-goals (deferred to later sub-items / sessions):**
- B1.c — `sage-discover` MCP server retrieval spans (independent code path)
- B1.d — `ui/FastAPI` HTTP spans (separate subsystem)
- B1.e — sampler tuning (gated on production volume data)
- ALIRE2 §B9 — per-run immutable AgentLoop context (separate refactor)

---

## 2. Constraints

| ID | Constraint | Rationale |
|----|------------|-----------|
| C1 | Rust-first directive (CLAUDE.md #1) | Performance-critical paths in Rust must own their telemetry; no Python-side workaround |
| C2 | Zero overhead when `SAGE_OTEL_EXPORTER=none` | Default config; unused subscribers must not allocate or emit |
| C3 | Same A16 redaction guarantees as B1 Python | Even though Rust spans currently emit only counts/IDs, audit + enforce on any future payload-bearing fields |
| C4 | No PyO3 upgrade (currently `pyo3 = "0.25"`) | PyO3 0.27 bump is a multi-day cross-cutting change unrelated to OTel |
| C5 | No new top-level deps without feature gate | Wheel size + Rust compile time matter; add via `otel` Cargo feature |
| C6 | Tests must run without `RUST_LOG`/`SAGE_OTEL_EXPORTER` set | Default test env is bare; instrumentation must remain off-by-default in tests |
| C7 | Rust ≤ Python in span-name semantics | Rust uses `sage.<crate>.<op>` (e.g. `sage.routing.system_router.route`); never overrides parent semantic conventions |
| C8 | OTel rust crate ≥ 0.27 supported (W3C TraceContext) | Pick a version that pairs with `tracing-opentelemetry` 0.28+ |

---

## 3. Architecture

### 3.1 Component diagram

```
┌──────────────── Python ────────────────┐    ┌────────── Rust (sage-core) ──────────┐
│                                        │    │                                       │
│  SAGE_OTEL_EXPORTER=otlp_http          │    │  Cargo feature: otel (opt-in)         │
│  ↓                                     │    │                                       │
│  observability/__init__.py             │    │  observability/mod.rs                 │
│    _init_tracer()                      │    │    init_otel(exporter, endpoint)      │
│    Python TracerProvider               │    │    Rust TracerProvider               │
│    OTLP HTTP exporter                  │    │    OTLP HTTP exporter                 │
│                                        │    │    tracing-opentelemetry Layer        │
│  observability/spans.py                │    │    Registry::default().with(layer)    │
│    sage_span(name, op, ...)            │    │                                       │
│    yields parent_traceparent_str       │    │  bridge_python_span(traceparent)      │
│                                        │    │    parses W3C traceparent string      │
│         ──── PyO3 boundary ─────       │    │    creates tracing span               │
│                                        │    │    span.set_parent(rust_cx)           │
│   sage.pipeline.run (Python span)      │    │    returns RustSpanHandle             │
│     │                                  │    │                                       │
│     └─ sage_span("sage.assign", ...)   │    │  Existing info_span!()/instrument!()  │
│        │  ──── traceparent injected ──→│    │  blocks become CHILDREN of bridged    │
│        │                               │    │  span automatically (tracing parent   │
│        └─ rust_assigner.assign(task)   │    │  inheritance)                         │
│           │                            │    │                                       │
│           sage_core.bridge_python_span─┼────┤  ┌─ "sage.routing.system_router.route"│
│           ↓                            │    │  ├─ "sage.assigner.score"            │
│           rust_method() runs           │    │  └─ "sage.bandit.sample"             │
│                                        │    │     OTLP → same collector as Python  │
│                                        │    │     (correlated by trace_id)         │
│                                        │    │                                       │
└────────────────────────────────────────┘    └───────────────────────────────────────┘
```

### 3.2 Boot lifecycle

1. Python `_init_tracer()` runs (existing — `sage-python/src/sage/observability/__init__.py`).
2. After Python TracerProvider is set, if `SAGE_OTEL_EXPORTER ∈ {console, otlp_http}`,
   call `sage_core.init_otel(exporter, endpoint)` (new). Rust mirrors the same
   exporter setup with its own TracerProvider. **Idempotent** — second call no-ops.
3. For `SAGE_OTEL_EXPORTER=logfire`: **Rust spans are NOT exported in MVP**.
   Python continues to use the logfire SDK; Rust init is skipped with a
   one-time WARN (`"logfire mode active; Rust spans will not be exported until
   B1.b.7 lands"`). Reasoning: logfire's auth header + endpoint contract is
   underspecified in the public docs, and the Python path already covers the
   primary user-facing observability surface. Tracking as B1.b.7 in §9.
4. For `SAGE_OTEL_EXPORTER=none`: `init_otel` is never called. Rust tracing
   remains a no-op (no subscriber installed).
5. When `sage-core` is built **without** the `otel` feature, the PyO3 module
   exposes `init_otel` as a stub that returns `None`, and `bridge_python_span`
   returns a no-op handle. Python falls back to "Python-only spans" mode. WARN
   once if `SAGE_OTEL_EXPORTER ≠ none` but the Rust feature is absent.

### 3.3 Per-call propagation

The Python `sage_span` context manager (existing) is enhanced:

```python
@contextmanager
def sage_span(name: str, op: str, *, record_exception: bool = True, **attrs):
    if not _otel_enabled():
        yield None
        return
    ...
    with tracer.start_as_current_span(name, ...) as span:
        # NEW: bridge into Rust if the otel-rust feature is active.
        # The handle is a Python ContextManager-like object that owns
        # the Rust-side span lifecycle. End-of-scope drops it.
        rust_handle = _maybe_bridge_to_rust(name, span)
        try:
            yield span
        finally:
            if rust_handle is not None:
                rust_handle.close()
```

`_maybe_bridge_to_rust` extracts the Python span's `SpanContext`, formats as
W3C traceparent (`{version:02x}-{trace_id:032x}-{span_id:016x}-{flags:02x}`),
and calls `sage_core.bridge_python_span(traceparent, name)`. Rust returns
an opaque `RustSpanHandle` that PyO3-wraps a `tracing::span::EnteredSpan`
(returned by `Span::entered()`, **not** a bare `Span`). The handle's `Drop`
runs the guard's `Drop`, exiting the span and restoring the previous current
span on the calling thread. This is critical: a bare `Span` registers a span
metadata entry but does NOT make it the thread-local current span — without
the entered guard, existing `info_span!` calls inside Rust would attach to
no parent (orphan-rooted), defeating the bridge.

**Thread-pinning note:** `EnteredSpan` is `!Send`. PyO3 callbacks hold the
GIL, which already pins the calling thread, so this is fine in practice.
Implementation MUST NOT spawn the handle into a thread pool or `tokio::spawn`.

**Why a handle, not a thread-local?** Rust async work spawned inside one Python
span needs the parent bound at span creation, not at thread-of-execution time.
A handle gives the caller explicit ownership — close-when-drop semantics match
both sync and async PyO3 calls. Thread-locals also break across PyO3 → tokio
work-stealing.

**Sync-only confirmation (audit-time):** the §4.1 audit verified that `sage-core`
has exactly 1 `async fn` (`sandbox/subprocess.rs:62 execute_async`) and it does
NOT contain any `info_span!` or `#[instrument]`. All 27 audited span call sites
are inside synchronous functions. The `.entered()` guard pattern is therefore
safe — no `.await` points break the parent-inheritance chain. If a future call
site introduces an async-spanned function, it must use `#[instrument]` or
`.instrument(future)` instead of `.entered()` to preserve cross-await parent
linkage.

### 3.4 Existing `info_span!` calls (no change required)

22 files already use `tracing::info_span!` / `#[instrument]`. With the
`tracing-opentelemetry` Layer installed and the bridged span as the active
context, these spans **automatically nest** as children:

```rust
// existing code (system_router.rs:201) — no change
let _span = info_span!("system_router.route", task_len = task.len(), budget = budget).entered();
```

When the bridged Rust span is the current context, `info_span!` reads it as
parent via tracing's standard inheritance, and `tracing-opentelemetry` Layer
emits an OTel span with the right parent_span_id. Zero changes to existing
call sites for the MVP.

**Rename pass deferred:** spans currently named `system_router.route` should
eventually become `sage.routing.system_router.route` to align with the Python
`sage.<stage>.<op>` convention. **Out of scope for B1.b** — does not affect
correctness of the trace tree; only cosmetic in tooling. Filed as B1.b.1
follow-up.

---

## 4. Payload safety in Rust

### 4.1 Audit of existing `info_span!` / `#[instrument]` call sites

Audited at spec-time (2026-04-25). All sites verified:

| File:line | Span name | Attrs | Payload risk |
|-----------|-----------|-------|--------------|
| `topology/engine.rs:235` | `topology_engine.generate` | `system`, `exploration_budget`, `task_len` | Counts/IDs only ✅ |
| `topology/engine.rs:362` | `topology_engine.smmu_path` | (none) | ✅ |
| `topology/engine.rs:435` | `topology_engine.archive_path` | (none) | ✅ |
| `topology/engine.rs:477` | `topology_engine.mutation_path` | (none) | ✅ |
| `topology/engine.rs:520` | `topology_engine.mcts_path` | (none) | ✅ |
| `topology/engine.rs:548` | `topology_engine.template_fallback` | `system` | ✅ |
| `topology/engine.rs:594` | `topology_engine.record_outcome` | `topology_id`, `quality`, `cost` | ✅ |
| `topology/engine.rs:711` | `topology_engine.evolve` | `pop_size`, `generations` | ✅ |
| `topology/engine.rs:735` | `evolve_generation` | `gen` | ✅ |
| `topology/llm_synthesis.rs:172` | `synthesis_stage` | `stage` (enum const) | ✅ |
| `topology/llm_synthesis.rs:191` | `synthesis_stage` | `stage` (enum const) | ✅ |
| `topology/llm_synthesis.rs:217` | `synthesis_stage` | `stage` (enum const) | ✅ |
| `topology/llm_synthesis.rs:348` | `topology_synthesis` | (none) | ✅ |
| `routing/system_router.rs:201` | `system_router.route` | `task_len`, `budget` | ✅ |
| `routing/system_router.rs:255` | `system_router.route_constrained` | `task_len`, `max_cost`, `max_latency`, `min_quality`, `explore` | ✅ |
| `routing/system_router.rs:370` | `system_router.route_integrated` | `task_len`, `topology_id` | ✅ |
| `routing/system_router.rs:492` | `system_router.record_outcome` | `decision_id`, `quality`, `cost`, `latency_ms` | ✅ |
| `routing/bandit.rs:378` | `bandit.select` | `arms`, `exploration` | ✅ |
| `routing/bandit.rs:483` | `bandit.select_contextual` | `arms`, `exploration`, `context_dim` | ✅ |
| `routing/bandit.rs:595` | `bandit.record` | `decision_id`, `quality`, `cost`, `latency_ms` | ✅ |
| `topology/map_elites.rs:438` | `map_elites.insert` | 6 bucket ints + `quality`, `cost` | ✅ |
| `topology/mcts.rs:127` | `mcts.search` | `max_simulations`, `max_time_ms` | ✅ |
| `verification/quality_labeler.rs:357` | `label` | `#[instrument(skip(self, task, response))]` | ✅ |
| `verification/smt.rs:602–873` | 10× `#[instrument(skip(self))]` | (none) | ✅ |
| `topology/density.rs:95` | `#[instrument(skip(self, graph))]` | (none) | ✅ |
| `memory/entity_graph.rs:74,98,122,153,233` | 5× `#[instrument(skip(self))]` | (none) | ✅ |
| `memory/relevance_gate.rs:45,62` | 2× `#[instrument(skip(self))]` | (none) | ✅ |

**Audit verdict:** **zero call sites carry raw user content.** Every `info_span!`
records counts, numeric values, IDs (decision_id, topology_id), or stage
enum constants. Every `#[instrument]` uses explicit `skip(...)` for any payload
parameters. **No Rust-side redaction helper required for MVP.**

If any future call site needs payload attrs (e.g. an `error_message`), it must
go through the redaction helper specified in §4.2 — not into a span attribute
directly. This is a code-review checkpoint, not a runtime gate.

### 4.2 Rust-side redaction helper (forward-looking; not wired in MVP)

Per §4.1 audit, no current Rust span attrs need redaction. The helper is
specified but **not implemented in B1.b MVP** — it ships when (and if) the
first payload-carrying span attribute lands.

Forward-looking spec for `sage-core/src/observability/redaction.rs`:

```rust
/// Redact secrets matching the same patterns as Python A16 RedactionFilter.
pub fn redact(s: &str) -> String { /* TBD when first caller appears */ }

/// Truncate to N UTF-8 bytes with the same `…[truncated]` suffix as Python.
pub fn truncate_utf8(s: &str, max_bytes: usize) -> String { /* TBD */ }
```

This avoids YAGNI: writing + maintaining a Rust port of A16 patterns has a real
cost (drift risk against `sage-python/src/sage/security/redaction.py`), and the
audit confirms zero MVP callers. **If audit verdict changes during
implementation, add this module before merging.**

### 4.3 Env contract

| Env var | Effect |
|---|---|
| `SAGE_REDACT_SECRETS=0` | Disable redaction in BOTH Python and Rust (consistency with B1) |
| `SAGE_OTEL_RAW_PAYLOADS=1` | Skip redaction + truncation in BOTH (dev only, big WARN already in Python) |

Rust reads these envs at the same time Python does — no separate Rust env.

---

## 5. Cargo wiring

### 5.1 New `otel` feature (opt-in initially, default later)

```toml
[features]
default = [
    "extension-module", "cognitive", "sandbox", "cranelift", "tool-executor",
    # otel NOT in default for first ship — opt in via maturin develop --features otel
]
otel = [
    "dep:opentelemetry",
    "dep:opentelemetry_sdk",
    "dep:opentelemetry-otlp",
    "dep:tracing-opentelemetry",
    "dep:tracing-subscriber",
]

[dependencies]
opentelemetry = { version = "0.27", optional = true }
opentelemetry_sdk = { version = "0.27", features = ["rt-tokio"], optional = true }
opentelemetry-otlp = { version = "0.27", features = ["http-proto", "http-json", "reqwest-client"], optional = true }
tracing-opentelemetry = { version = "0.28", optional = true }
tracing-subscriber = { version = "0.3", features = ["registry", "env-filter"], optional = true }
```

**Version pinning rationale:** `tracing-opentelemetry 0.28` is the version that
pairs with `opentelemetry 0.27`. Both released after PyO3 0.25 was current — no
PyO3 conflicts. `opentelemetry 0.30` requires PyO3 0.27 via the rigetti crate
(not used here), so we stay on 0.27 to avoid the upgrade. Verified compatible
with `tracing = "0.1"` (current sage-core dep).

### 5.2 Build matrix

| Build | Has `otel` feature? | Behavior |
|-------|---------------------|----------|
| `maturin develop` (default) | No | `init_otel` returns `None`. `bridge_python_span` no-ops. Python WARNs if exporter ≠ none. |
| `maturin develop --features otel,smt,onnx` | Yes | Full Rust OTel bridge. Spans flow to OTLP. |
| CI (smt + tool-executor + cognitive) | No | OTel paths not exercised. Compile-time gate keeps build minimal. |
| Release wheel (future) | Yes | Once stable, flip `otel` into default features. |

### 5.3 `sage-python/src/sage/__init__.py` smoke import

`sage_core.init_otel` and `sage_core.bridge_python_span` must always exist
as attributes — Rust exports them unconditionally with stub bodies when
the `otel` feature is off, real bodies when on. Avoids the
`AttributeError` cliff at the PyO3 boundary.

---

## 6. Trace context propagation contract

### 6.1 Python → Rust handoff

Python passes a W3C traceparent string at `sage_span` enter:

```
00-{trace_id_lower_hex_32}-{span_id_lower_hex_16}-{flags_lower_hex_2}
```

Rust parses with `opentelemetry::propagation::TraceContextPropagator::extract`.
On parse failure: log WARN, create a Rust span with no parent (becomes a new
root). MVP does not retry or surface the error to Python.

### 6.2 Rust → Python (out of scope)

No Rust → Python span propagation in B1.b. Rust never spawns Python work.
If a future feature needs this (e.g. Rust calling a Python tool), it gets
its own propagator + design.

---

## 7. Testing

### 7.1 Rust unit tests (`cd sage-core && cargo test --features otel,smt --lib`)

| Test | Verifies |
|------|----------|
| `observability::tests::init_otel_idempotent` | 2nd call to `init_otel` returns same handle, doesn't double-install subscriber |
| `observability::tests::traceparent_parse_valid` | Standard W3C string parses to correct trace_id + span_id |
| `observability::tests::traceparent_parse_invalid` | Garbage input returns Err, doesn't panic |
| `observability::tests::span_inherits_parent` | InMemorySpanExporter records the bridged span with correct parent_span_id |
| `observability::tests::no_subscriber_no_panic` | Calling `bridge_python_span` without prior `init_otel` returns no-op handle, doesn't panic |

### 7.2 Python integration tests (`cd sage-python && python -m pytest tests/observability/`)

| Test | Verifies |
|------|----------|
| `test_rust_bridge.py::test_rust_span_nests_under_python_parent` | Acceptance criterion §11.A |
| `test_rust_bridge.py::test_rust_init_skipped_when_exporter_none` | Acceptance criterion §11.B |
| `test_rust_bridge.py::test_rust_init_skipped_when_feature_off` | Sage-core without `otel` feature warns once, no error |
| `test_rust_bridge.py::test_redaction_applied_to_rust_attrs` | A16 patterns redacted in Rust before reaching exporter |

Tests use `InMemorySpanExporter` on Python side + a mocked OTLP endpoint
(or a Rust-side `opentelemetry_sdk::testing::trace::InMemoryExporter`).

### 7.3 Smoke test (manual)

Run `python -m sage.bench --type routing_gt` with `SAGE_OTEL_EXPORTER=console`
and `--features otel`. Confirm console output shows:
- 1 root `sage.pipeline.run` span (Python)
- 6 child stage spans (Python)
- Each routing call emits a `system_router.route` Rust span as child of its
  Python `sage.assign` parent
- All correlated by single `trace_id`

### 7.4 Coverage check

The §4.1 audit table contains 27 audited call sites with 21 distinct span names
(some files reuse the same name across multiple stages, e.g. `synthesis_stage`).
At least 8 distinct span names should appear in a `routing_gt` smoke trace
(those on the routing → bandit → record_outcome critical path). The full set
of 21 only appears in benchmarks that exercise topology synthesis + evolution.
Implementation deliverable's smoke test asserts ≥ 8 distinct Rust span names
on the routing path; broader coverage is a manual sanity check.

---

## 8. Acceptance criteria

| ID | Criterion | Verification |
|----|-----------|--------------|
| A | Rust `system_router.route` span has correct `parent_span_id` matching the Python `sage.assign` span when `SAGE_OTEL_EXPORTER=otlp_http` and `--features otel` | `test_rust_span_nests_under_python_parent` |
| B | When `SAGE_OTEL_EXPORTER=none`, no Rust subscriber is installed and no Rust spans are emitted | `test_rust_init_skipped_when_exporter_none` |
| C | Without `otel` feature, the `sage_core.init_otel` import succeeds and returns `None`; the Python wrapper logs WARN once | `test_rust_init_skipped_when_feature_off` |
| D | ≥ 8 distinct Rust span names visible in a `routing_gt` smoke trace (routing path coverage) | smoke test in §7.3 |
| E | Zero new test failures on existing Rust + Python suites (501 + 2493 baseline) | full CI pass |

---

## 9. Out-of-scope (B1.b) — future tickets

| ID | Item | Why deferred |
|----|------|--------------|
| B1.b.1 | Rename `system_router.route` → `sage.routing.system_router.route` (cosmetic alignment) | No correctness impact; can ship later |
| B1.b.2 | OTel metrics (counters, histograms) | Different API surface, separate spec |
| B1.b.3 | Logfire-specific Rust exporter (logfire crate or direct API) | Logfire OTLP path covers the use case; Rust-native logfire crate would be optimization |
| B1.b.4 | Auto-injection of W3C traceparent into outgoing Rust HTTP calls (provider clients) | Provider HTTP is in Python today; revisit when Rust-side HTTP appears |
| B1.b.5 | Sampler tuning | Same as B1.e — gated on production data |
| B1.b.6 | Rust → Python span propagation | No call paths today |
| B1.b.7 | Logfire-mode Rust export (`SAGE_OTEL_EXPORTER=logfire` + Rust→logfire OTLP) | Auth/endpoint contract underspecified; Python logfire path already covers primary use case. WARN once on use of logfire+Rust feature combo. |
| B1.b.8 | CI matrix coverage for `--features otel` | New code is opt-in, default `cargo test --lib` won't exercise it. Add a CI job (`features otel`) **OR** explicitly accept that the otel path is dev-machine-tested only until first production user. Decided in plan, not spec. |

---

## 10. Cost / effort estimate

| Phase | Effort |
|-------|--------|
| Cargo wiring + `init_otel` skeleton | 0.5 day |
| `bridge_python_span` + traceparent parse + Rust unit tests | 0.5 day |
| Python `sage_span` integration + WARN paths | 0.25 day |
| Python integration tests (4) | 0.25 day |
| §4.1 attr-redaction audit pass on 22 files | 0.25 day |
| Smoke test + docs + roadmap update | 0.25 day |
| **Total** | **~2 days** (matches advisor's pre-pick estimate) |

**Risk multipliers:**
- 1.3× if `tracing-opentelemetry 0.28` has API drift from 0.31 docs (not verified
  against rust 0.27 yet) — adds 0.5 day to find right version pair
- 1.5× if any of the 22 `info_span!` call sites needs deep refactor (e.g.
  carries a payload that the audit reveals) — adds 1 day
- No PyO3 upgrade risk (verified C4)

Worst case: ~3 days. Best case: ~1.5 days.

---

## 11. Open questions resolved during design

| Q | Answer |
|---|--------|
| Does Rust need its own TracerProvider OR can it call Python's via PyO3? | Own SDK with OTLP to same collector. Standard polyglot pattern. PyO3 callbacks for span export would cause GIL contention on the hot path. |
| How does Rust acquire the Python parent span at PyO3 boundary? | Python serializes its current SpanContext as a W3C traceparent string and passes it to `bridge_python_span(traceparent, name)`. Rust parses + sets as parent. |
| Why not the rigetti `pyo3-opentelemetry` crate? | It requires PyO3 0.27 (we're on 0.25). Multi-day cross-cutting upgrade. Manual traceparent passing achieves the same propagation without the upgrade. |
| What about logfire mode? | Rust uses OTLP HTTP against logfire's collector if `LOGFIRE_TOKEN` resolvable; else WARN once, Rust spans skipped. |
| Async patterns? | Existing tracing crate macros (`info_span!`, `#[instrument]`) are async-aware. No new pattern needed. |
| Zero overhead when off? | tracing crate's no-op path when no subscriber installed: a few instructions per `info_span!` to check global state. Negligible (existing `info_span!` calls already pay this when no subscriber is wired). |

---

## 12. References

- W3C TraceContext: https://www.w3.org/TR/trace-context/
- tracing-opentelemetry crate: https://docs.rs/tracing-opentelemetry/latest/tracing_opentelemetry/
- opentelemetry-rust 0.27: https://docs.rs/opentelemetry/0.27/opentelemetry/
- ALIRE2 §4 — shared mutable state (B9 follow-up)
- B1 spec (Python-side, predecessor): `docs/superpowers/specs/2026-04-25-otel-genai-spans-design.md`
- roadmap-A2 cascade investigation (motivating diagnostic): `docs/audits/2026-04-23-alire-verification.md`
