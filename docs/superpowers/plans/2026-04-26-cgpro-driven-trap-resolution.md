# cgpro-Driven Autonomous Trap Resolution Protocol

**Created**: 2026-04-26
**Owner**: Claude (orchestrator) + cgpro (source of truth) + codex gpt-5.5-xhigh (workhorse implementer)
**Active conversation**: `cgpro_2026_04_26_review` (alias for `69ee3d8d-6154-8392-b79a-3a0202e887d2`) — every cgpro call uses `--resume cgpro_2026_04_26_review` for continuity.
**Scope**: Resolve the 9 items surfaced by the 2026-04-26 cgpro post-closeout review (A8, A14, A15, A12 docs, A3a, A9+A10, A13, A11, plus traps C/E/F as side-tickets) autonomously, with cgpro as the source of truth and codex as the workhorse implementer.

## Token economy contract

User instruction (2026-04-26): "économiser les tokens claude au maximum".

Allocation:
- **cgpro** (free on Claude context, pay-on-OpenAI): heavyweight strategic review, spec authoring, post-implementation verification.
- **codex gpt-5.5-xhigh** via `codex:rescue`: implementation + local tests + diff emission. Returns small structured summaries.
- **subagents** (`Explore`, `Plan`): focused targeted work that keeps the main context lean.
- **Claude (this session)**: orchestration, verification of small diffs (≤200 LOC visible at a time), final commit narratives, debate-when-disagreeing arbitration.

Concretely: every trap resolution that costs me >2k tokens of reading or >500 LOC of writing should be delegated. Claude reads diffs, not whole files; Claude composes commits, not whole modules.

## The protocol — per-trap state machine

For each trap T in priority order:

```
DESIGN ──→ IMPLEMENT ──→ VERIFY-LOCAL ──→ VERIFY-CGPRO ──→ SHIP
   ↑                          │                  │
   └──── FIX ←────────────────┘                  │
                       (test failure)            │
                                                 ↓
                                            (cgpro flags drift)
                                                 ↓
                                                FIX
                                                 │
                                                 ↓
                                          re-VERIFY-LOCAL
```

### DESIGN (cgpro authors the spec)

Send to `cgpro --resume cgpro_2026_04_26_review`:

```
Trap T — locking the spec.

Your 2026-04-26 review proposed: <one-paragraph quote of cgpro's earlier proposal for T>

I want to lock the implementation spec before we hand to codex. Please:
1. Confirm the proposal still stands or refine it.
2. Specify file paths + function signatures + acceptance test names.
3. Specify the verification command (e.g. `cd sage-core && cargo test --features smt --lib bandit::tests::T`).
4. Flag any traps in the implementation surface I should warn codex about.
5. If the proposal needs a decision from me (e.g. "keep posteriors vs reset"), name the decision and your default recommendation.
```

cgpro returns a spec. **If I disagree with any part**, surface the conflict in a follow-up `--resume` call. Don't silently switch. The user is explicit: "n'hésites pas a lui donner plus d'informations si tu n'es pas d'accord avec lui et débattre".

### IMPLEMENT (codex executes)

Hand to `codex:rescue` agent with a prompt of the form:

```
Implement trap T per the locked spec at <link to spec snippet from cgpro>.

Repo: C:\Code\YGN-SAGE (current branch: main).
Files in scope: <paths>.
Acceptance tests: <names>.
Verification: <commands>.

Constraints:
- TDD: write the failing acceptance test FIRST, then make it pass.
- No new abstractions. No comments unless WHY is non-obvious.
- Match existing style. Don't refactor surrounding code.
- Run the verification commands before declaring done. Report stdout.

Return: a structured summary with (a) files changed, (b) test names added, (c) verification command outputs, (d) anything you flagged as not-quite-matching the spec.
```

### VERIFY-LOCAL (Claude runs the tests)

Trust-but-verify the codex summary. Run the verification commands locally:

```bash
cd sage-core && cargo test --features smt --lib <test_filter>
cd sage-python && python -m pytest tests/<test_path> -v
cd sage-core && cargo clippy --no-default-features -- -D warnings
cd sage-python && ruff check src/
```

If anything fails, loop back to FIX with codex carrying the failure output.

If everything passes, run a small `git diff` to scan the changes (eyeballing for: scope creep, unrelated edits, dropped semicolons, new TODOs).

### VERIFY-CGPRO (cgpro audits the diff)

Send to `cgpro --resume cgpro_2026_04_26_review`:

```
Trap T — verification pass.

Codex shipped a fix per your locked spec. Diff attached (or summary, if diff is large).

Diff:
<unified diff or codex summary>

Verification:
<test outputs>

Three questions:
1. Does the implementation match your spec? Any drift?
2. Did codex slip in any unrelated change that should be scoped out?
3. Any second-order trap I'm missing?
```

If cgpro flags drift, loop back to FIX. If cgpro approves, proceed to SHIP.

### SHIP (Claude commits)

Compose a commit message (cgpro and codex don't commit; that's the orchestrator's job). Include:
- One-line subject: `<type>(<scope>): <verb-phrase>` (matches existing repo style — see `git log --oneline`).
- Body: 2-4 sentence "why", reference to trap ID (e.g. roadmap-A14), reference to spec doc.
- Co-Authored-By: cgpro and Codex GPT-5.5 (xhigh) so attribution survives.

Update `roadmap.md`: mark trap T as `✅ SHIPPED <date>` with commit SHA. Update `MEMORY.md` "Active direction" if state has materially changed.

## Order of execution

cgpro's priority (their words): A8 → A14 → A15 → A12 docs → A3a → A9+A10 → A13 → A11 → A3.

I'm starting from a different angle for **architecture validation**: pick the cheapest trap as the first run to debug the protocol itself. Then escalate to higher-stakes traps once the workflow is proven.

| Order | Trap | Why this order |
|---|---|---|
| 1 (test) | **A12 docs-only** | Cheapest, safest, fully reversible. Validates the cgpro+codex+verify+ship loop end-to-end. |
| 2 | **A14 bandit causality** | CRITICAL prod bug. Highest user-visible impact. |
| 3 | **A15 packaging fail-closed** | CRITICAL distribution gap. |
| 4 | **A3a verifier reason codes** | No API budget needed. Pairs with A1 observability work. |
| 5 | **A8 wasm in CI** | Largest CI-infra change. Includes Trap E (Python 3.13 + Windows sandbox matrix) + Trap F (tool_executor.rs docs sweep) since they're all "CI matrix coherence" class. |
| 6 | **A9+A10 RNG seam + sort arm_keys** | Structural debt, paired. |
| 7 | **A13 lockfile/constraints** | Reproducibility. |
| 8 | **A11 three-layer test split** | Depends on A9. |
| - | A3 paired N=50 | API-budget gated. Out of scope for this cycle. |

## Per-trap notes

### A12 docs-only (test run)

Spec preview (will lock with cgpro before implementing):
- Edit `sage-core/src/routing/bandit.rs` module-doc to remove "global Pareto front" / "constraint-aware selection" claims and describe what `choose()` / `choose_contextual()` actually do (sampled quality + cosine bonus; cost/latency are sampled for telemetry, not selection).
- Add a `#[doc = "..."]` regression hint pointing to the future A12-fix-code path that would resurrect these claims.
- (Trap F bundle): same docs-cleanup commit also fixes any `tool_executor.rs` module-doc that claims subprocess fallback is "always available" — should match validate_and_execute()'s ADR-013 hard-fail behaviour.

Acceptance test: `cd sage-core && cargo doc --no-deps 2>&1 | grep -E "warning|error"` returns no warnings tied to the edited modules.

### A14 bandit causality test

This is the prod bug. Before implementing the test, decide with cgpro (in DESIGN phase):
- (a) keep accumulated bandit posteriors and fix only forward attribution, OR
- (b) reset SQLite bandit state on next deploy (lose history), OR
- (c) audit the in-prod posteriors first (sample N decisions, check if the chosen arm correlates with the executed model — if random, reset is the only honest path).

cgpro's default rec will be in their DESIGN response. I'll either accept or debate.

The fix itself (per cgpro): assert `decision.model_id` is **what executed** before letting `record_outcome()` fire. Probably means: pipeline.py:1244 stores `ctx.bandit_decision = decision` (full struct, not just ID); Stage 3 ModelAssigner has to consult `ctx.bandit_decision.model_id` (or the bandit has to be wired into Stage 0 via `route_integrated()` instead of `route()`); Stage 5 only fires `record_outcome` when `ctx.executed_model == ctx.bandit_decision.model_id`.

There may be a simpler rewrite: replace `_rust_router.route(...)` at pipeline.py:461 with `_rust_router.route_integrated(...)` (which DOES use the bandit). Then drop the duplicate `bandit.select_with_context()` call at pipeline.py:1243-1252 — Stage 0 already did the selection. cgpro will tell us if this is the right shape.

### A15 packaging fail-closed

Two sub-fixes:
1. `sage-python/pyproject.toml` declares `sage_core` as a dep (or document that the wheel ships sage_core as an extension; check pyproject for the binary-extension story).
2. `forge.py:352-354` (per AUDIT2 §6 / ALIRE2): when Rust validator unavailable, fail closed unless `SAGE_TOOLFORGE_STRICT=0` explicitly opts into the legacy ast.parse() fallback.

Note: A18 (in roadmap.md) covers part of #2; reconcile so they don't conflict.

### A3a verifier reason codes

`sage-python/src/sage/bench/swebench_diff_verifier.py` — extend `_verify_hunks` to emit reason codes per hunk + a top-level outcome. JSON schema gets a new `_diff_verifier_reasons: list[str]` field. Smoke harness aggregates reasons across runs for the bucket-analysis report.

### A8 wasm in CI (+ Trap E + Trap F merged in)

GitHub Actions sandbox-build job:
1. Cache key on `external/rustpython` submodule SHA + `wasm32-wasip1` toolchain version.
2. `cd external/rustpython && cargo build --release --target wasm32-wasip1 --features freeze-stdlib`.
3. Upload as artifact for downstream sage-core build job to pull.
4. sage-core build with `--features smt,onnx,sandbox,cranelift,tool-executor` + `SAGE_REQUIRE_WASM=1` (so the wasm-missing case fails the build).
5. New CI step: `python -c "import sage_core; assert sage_core.embedded_wasm_available()"` runs in linux-pytest, integration-smoke, AND windows-pytest.
6. Add Python 3.13 to the matrix in linux-pytest (Trap E).

### A9+A10 RNG seam + sort arm_keys

Spec template:
- `ContextualBandit::choose()` and `choose_contextual()` accept `&mut impl Rng` instead of internally calling `rand::rng()`.
- New `*_with_rng` PyO3 wrappers (Python tests can pin `ChaCha8Rng::seed_from_u64(seed)`).
- `arm_keys = self.arms.keys().sorted_by_key(|k| (k.model_id.clone(), k.template.clone())).collect()` before Thompson sampling.
- Same for `CmaEmitter::ask()`.
- Test changes: 5 stochastic tests promoted to "seeded" tier (deterministic with fixed seed).

### A13 lockfile / constraints

Generate `sage-python/constraints.txt` from a clean `pip install -e .[all,dev]` resolution. CI installs with `pip install -c constraints.txt -e .[all,dev]`. Separate weekly-scheduled `latest-deps` CI job re-resolves without constraints to catch drift.

### A11 three-layer test split

Promote A9-seeded tests to "layer 2", extract pure-mechanics deterministic tests as "layer 1", mark "this usually converges" assertions as `#[ignore]` for nightly. Depends on A9 RNG seam for the layer 2 promotion to be meaningful.

## Failure modes & rollback

- **codex returns broken code that passes tests**: cgpro VERIFY-CGPRO step catches this (cgpro reads the diff, not just test output).
- **cgpro hallucinates a bug**: Claude verifies file:line citations before commit. Two false-positive checks already this cycle.
- **codex disagrees with cgpro**: Claude debates with cgpro using codex's argument as input. Whoever has primary-source evidence wins.
- **A trap's blast radius exceeds expectation**: Stop, document the surprise, ask user for direction. Don't power through.
- **CI breaks**: rollback the shipping commit (`git revert <sha>`); the protocol's per-trap separation makes single-commit reverts safe.

## Progress tracking

This document is updated as each trap closes. Status table:

| # | Trap | Spec locked? | Implemented? | cgpro verified? | Shipped (commit) | Notes |
|---|---|---|---|---|---|---|
| 1 | A12 docs | ⏳ | | | | Pilot run for protocol validation |
| 2 | A14 bandit causality | | | | | Decision needed: keep vs reset posteriors |
| 3 | A15 packaging | | | | | Reconcile with roadmap-A18 |
| 4 | A3a reason codes | | | | | |
| 5 | A8 wasm CI | | | | | Bundles Trap E + Trap F |
| 6 | A9+A10 RNG | | | | | |
| 7 | A13 lockfile | | | | | |
| 8 | A11 test split | | | | | Depends on A9 |
