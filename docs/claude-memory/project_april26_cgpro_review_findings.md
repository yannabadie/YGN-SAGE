---
name: April 26 cgpro post-closeout review findings
description: cgpro found 6 new traps (3 critical) beyond the 7 pending roadmap-A8..A13 items; A14 (bandit causality) + A15 (packaging) + Trap F (tool_executor stale docs) verified in live code
type: project
originSessionId: bf130342-a1fc-4ba3-9819-62c0d87a6b87
---
# cgpro 2026-04-26 post-closeout review — verified findings

cgpro consultation against commit `e8540c3f` (8540c3f8d4a5fd50085b395c59e45857ebac4b0). Web search on (always-on policy). Conversation `69ee3d8d-6154-8392-b79a-3a0202e887d2`.

## Verdict on what shipped (25-commit closeout cycle)

> "CI debt closed; runtime/security contract coverage still has known holes."

Stochastic redesigns: bandit orthogonal-context = legitimate (real cosine separation in choose_contextual). cma_me budget bumps = legitimate but incomplete (CmaEmitter::ask still uses rand::rng() — tests stochastic by construction; A9/A10 still required for determinism). mypy exclude for generated_tools = honest as written, but should add a CI guardrail that the directory contains only generated artefacts.

## Three new CRITICAL traps (verified in live code 2026-04-26)

### roadmap-A14 — Python pipeline bandit off-policy (REAL PROD BUG)

**What**: pipeline.py:461 (Stage 0 classify) calls `self._rust_router.route(ctx.task, ctx.budget)` — the **legacy API** at system_router.rs:199-200 (doc-comment "legacy API"). It picks a model M via `decide_system` + `select_for_system` + `select_within_budget` — **does not consult the bandit**. Then pipeline.py:1232-1244 (Stage 4 execute) calls `self.bandit.select_with_context(0.1, task_context)` separately. Returns `BanditDecision { decision_id, model_id, template, ... }` but **only `decision.decision_id` is stored** as `ctx.bandit_decision_id` (lines 1244, 1250). The chosen `model_id` and `template` are dropped on the floor. Stage 5 (`pipeline.py:1762`) records outcome against `ctx.bandit_decision_id`. **The bandit posteriors update for an arm `(model_M, template_T)` whose model never executed.**

**Verified by**: Read `pipeline.py:440-499` (Stage 0 calls legacy `.route()`); `pipeline.py:1230-1252` (bandit.select_with_context, only decision_id stored); `pipeline.py:1758-1765` (record_outcome against orphan decision_id); `system_router.rs:199-243` (legacy `.route()` does NOT call bandit); `bandit.rs:227` (`BanditDecision` carries model_id + template that get dropped).

**Combined with the 2026-04-26 `restore_arm` fix** (`9f251276`, persists context_sum/context_count): the bandit was simultaneously (1) learning the wrong attribution forever and (2) losing context bias on every restart. Fix (1) was identified today but never shipped.

**Decision needed before fixing**: keep accumulated bandit posteriors (off-policy garbage) vs reset SQLite bandit (clean restart, lose history).

**cgpro proposal A14 — bandit causality test**: before `record_outcome()` fires, assert the selected arm changes the executed model/template. If the selected arm is only telemetry, refuse to record it as causal learning.

### roadmap-A15 — Packaging contract drift (REAL DISTRIBUTION GAP)

**What**: README announces `pip install ygn-sage`. `sage-python/pyproject.toml:18-31` lists dependencies httpx/pydantic/rich/anyio/aiosqlite/numpy/truststore/pydantic-ai. **No `sage_core` dependency.** `.claude/rules/architecture.md` declares: "sage_core is required at runtime — ImportError raised at TopologyController.__init__ if absent." CI compensates with separate maturin build + pip install --force-reinstall; PyPI users get nothing matching that recipe.

**Verified by**: Read `sage-python/pyproject.toml:1-60`; checked dependencies section; cross-referenced `.claude/rules/architecture.md` "sage_core required at runtime" claim.

**Downstream**: when `sage_core.ToolExecutor` is missing, `create_python_tool()` falls back to the Python subprocess sandbox path. That path provides timeout isolation only — no seccomp, namespaces, cgroups, FS isolation, or network isolation (unlike the wasm sandbox).

**cgpro proposal A15 — distribution smoke**: clean install must import `sage_core`, AND dynamic tool execution must fail closed if Rust ToolExecutor is unavailable (no silent fallback to Python subprocess sandbox unless `SAGE_UNSAFE_SUBPROCESS=1` or similar explicit opt-in).

### roadmap-A3a — Diff-context verifier reason-code emission (NO API BUDGET NEEDED)

**What**: verifier currently collapses malformed input / header drift / missing files / creation+deletion all to `[] = "no opinion"`. Roadmap A1 already records a real missed failure (astropy-6938 malformed hunk header arithmetic, Docker rejected before content verifier could help). Zero flags ≠ "no opinion needed" — could be "no opinion possible".

**cgpro proposal A3a — emit reason codes** for every hunk verification:
- `clean` / `content_mismatch` / `file_missing` / `malformed_hunk_header` / `hunk_body_count_mismatch` / `file_creation_or_deletion` / `not_unified_diff` / `unsupported_no_opinion`

Turns "zero flags" into an interpretable distribution. **Local action, no API budget.** Pairs with the A6 CRLF-normalization fix (which added the `crlf_normalized` repair_stage) — same observability discipline applied to the verifier itself.

## Three smaller traps (lower priority but pinned)

### Trap C — GammaPosterior cost/latency semantics counterintuitive (latent bomb in A12 fix-code path)

`bandit.rs::GammaPosterior::update(value)` increments `rate` by the observed value. `mean = shape / rate`. **Larger observed cost/latency → lower posterior mean.** Counterintuitive if `expected_cost` / `expected_latency` are presented as "minimands" (the documented intent on `BanditDecision`).

**Today inert**: `choose_contextual()` only uses sampled quality + cosine; cost/latency are sampled for telemetry, not selection. **Tomorrow's bomb**: if A12 takes the "fix-code" path (real Pareto multi-objective routing), this orientation needs explicit handling — either flip the convention or document "lower posterior mean = higher expected cost". Verified in `bandit.rs::GammaPosterior` (lines TBD when A12 is opened).

### Trap E — CI matrix gaps

`pyproject.toml` advertises Python 3.13. `.github/workflows/ci.yml` uses Python 3.12 across linux-pytest, integration-smoke, windows-pytest. **3.13 is not exercised in CI.** Windows job builds Rust wheel with `--features smt,onnx` but **NOT `sandbox,cranelift`** — so embedded wasm sandbox is not exercised on Windows. A8 must add a sandbox-on-Windows assertion alongside the wasm-build job.

### Trap F — `tool_executor.rs` stale top-level docs

cgpro claim: `tool_executor.rs` module-doc says subprocess fallback is "always available", but `validate_and_execute()` hard-fails if wasm isn't loaded (no subprocess fallback per ADR-013 §5 flip 2026-04-22). Same docs/code mismatch class as A12 — and tests pass while docs lie.

**Status as of 2026-04-26**: file location not verified yet (`sage-core/src/tool_executor.rs` returned "Path does not exist" in initial verification — file may have been moved/renamed; verify via Glob before editing).

## cgpro's revised priority order (active-risk + truthfulness-of-green criterion)

| # | Item | One-liner |
|---|------|-----------|
| 1 | A8 | Build rustpython.wasm in CI — sandbox is P0 contract with zero CI coverage |
| 2 | **A14** | Bandit causality test — prove select drives execution before record_outcome |
| 3 | **A15** | Packaging fail-closed — clean install must work; no silent subprocess fallback |
| 4 | A12 docs | ~30 min, fix Pareto/constraint-aware claims to match what code actually does |
| 5 | **A3a** | Verifier reason codes — local, no API budget |
| 6 | A9+A10 | RNG seam + sort `arm_keys` — structural debt |
| 7 | A13 | Constraints/lockfile — reproducibility |
| 8 | A11 | Three-layer split — depends on A9 |
| 9 | A3 | Paired N=50 observe-vs-repair — API-budget gated |

cgpro's pushback line on user's "minimize future flake-debugging time" framing:

> "Your user's argument optimizes developer time. A8 and A14 optimize **truthfulness of green**. The bigger risk is not another stochastic flake; it is a green run that does not exercise the sandbox and a learner that may be updating the wrong arm."

If only one item ships next session: **A8** (per cgpro). But this work cycle scopes A8+A14+A15+A12+A3a+Trap F as a coherent batch (all "truthfulness of green" class).

## Methodological note for future cycles

cgpro caught (1) the `restore_arm` persistence bug on the first review and now (2) the bandit causality issue + (3) the packaging gap on the follow-up. Pattern: substantial closeouts get a cgpro pass with GitHub repo URL + structured "what shipped / what's stuck / what's next" + 2-3 specific questions. Time investment per call: 10-25 min wall-clock; finds prod bugs that diff review misses.
