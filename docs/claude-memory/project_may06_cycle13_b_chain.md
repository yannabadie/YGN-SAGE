---
name: May 6 — Cycle-13 B chain FULLY CLOSED in 4 defense layers
description: Autonomous session segment 2026-05-06 morning. Stale Rust binary class wrapped in L1 source/L2 regression/L3 boot-ops/L4 release-pipeline. 7 cycle-13 B commits + ~13 prior in the same autonomous segment = 20 total. cgpro `cgpro_pi_mono_pivot_20260505` HITL with 8 deep VERIFY rounds + 3 HARD_STOP fixes (TOPOLOGY_STATE_MANIFEST_FILENAME sourcing, `git rev-parse --git-path` for PyPI sdist + worktrees, sentinel-file guard against unrelated git repos). HEAD `db304bc6`. 3089 Python / 553 Rust / 100 sage-discover. cgpro verdict `E` = defensible stop.
type: project
originSessionId: 3b36b883-89b8-4bbd-a2fe-cc9d305be971
---

## Empirical bug catch

Cycle-13 E Tier 2.1 smoke runtime (2026-05-05 evening): `engine.save_state(dir)` returned `Ok(())` but did NOT write `topology_state_manifest.json`. Each successful pipeline run left `~/.sage/` with state files but no manifest → next boot fail-closed per directive #8 → operator forced to run `python -m sage.ops.a14_reset` per cycle.

Root cause = stale binary class:
- `sage_core.cp313-win_amd64.pyd` dated **2026-04-27 15:22**.
- Manifest-write fix at `engine.rs:1031` shipped commit `f9521616` **2026-04-30** (cycle-8 step 2 A14 VERIFY round-1).
- 4-day gap between binary build and source fix → silent contract violation.

Fix delivered immediately (rebuild via `maturin build --release --features smt,onnx + pip install --force-reinstall`). 4 defense layers shipped to prevent recurrence + detect future drift.

## L1 source code (Rust)

| Commit | Subject |
|---|---|
| `bc662d9a` | `write_bytes_atomic` closure-wrapped + best-effort `remove_file(.tmp)` on rename failure + 2 Rust unit tests (success-no-leak + rename-failure-cleanup). Pre-fix: `.tmp` files leaked monotonically across retries. |
| `b035973e` | `build.rs` injects `SAGE_CORE_BUILD_COMMIT_SHA` (via `git rev-parse HEAD` OR `SAGE_CORE_COMMIT_SHA_OVERRIDE` env) + `SAGE_CORE_BUILD_TIMESTAMP` (UNIX seconds) + `SAGE_CORE_BUILD_PROFILE`. lib.rs exposes 4 PyO3 module attrs `__commit_sha__/__build_timestamp__/__build_profile__/__version__`. cgpro HARD_STOP #2 fixed: hardcoded `../.git/HEAD` was wrong for PyPI sdist + worktrees → resolved via `git rev-parse --git-path HEAD/refs/heads/packed-refs` and emitted `cargo:rerun-if-changed` only when path exists. |

## L2 regression test (Python boundary)

| Commit | Subject |
|---|---|
| `32d39bdf` | `tests/test_save_state_manifest_contract.py` — 2 tests. `sage_core.TopologyEngine().save_state(tmp)` MUST write manifest. cgpro deep VERIFY 2026-05-06 Q3: every `state_files[]` entry's `sha256/size_bytes` MUST equal the file's CURRENT bytes (byte-exact binding). CI builds fresh wheels per commit → passes on every CI run; local dev with stale `.pyd` gets `_REBUILD_HINT` error. |

## L3 boot-time ops

| Commit | Subject |
|---|---|
| `b25c28a6` | `sage.ops.a14_reset` cleans orphaned `.<name>.<id>.tmp` files (pre-`bc662d9a` artifacts). `_orphan_tmp_files` + `_cleanup_orphaned_tmp_files` helpers. `_ATOMIC_WRITTEN_NAMES = (POSTERIOR_EPOCH_FILENAME, TOPOLOGY_STATE_MANIFEST_FILENAME, CONTAMINATED_MARKER_FILENAME)` — cgpro HARD_STOP #1 fixed (was hardcoded string for the manifest filename). Audit `MANIFEST.json.cleaned_orphan_tmp_files` always present (empty list when none). 8 new tests + 7 surviving from before (14/14 PASS). |
| `9e426504` | `sage.ops.sage_core_version` Python helper consuming the L1 build-info attrs. CLI: `python -m sage.ops.sage_core_version` exits 0 when wheel matches source HEAD, 1 on confirmed stale, 0 on unknown (default) / 1 on unknown with `--strict`. cgpro HARD_STOP #3 fixed: validates cwd's git toplevel is a YGN-SAGE checkout via `git rev-parse --show-toplevel` + sentinel-file check (`sage-core/Cargo.toml` + `sage-python/src/sage/__init__.py`) BEFORE returning HEAD — otherwise running from inside an UNRELATED git repo would falsely flag the YGN-SAGE wheel as stale. 24 new tests. |

## L4 release pipeline

| Commit | Subject |
|---|---|
| `db304bc6` | `sage.ops.wheel_smoke` post-install assertion module + CI wiring. 4 phases: `_check_sage_core_imports` / `_check_build_info_attrs` (commit_sha != "unknown" CI smell) / `_check_required_symbols` (canonical 8 pyclasses) / `_check_save_state_manifest_contract` (byte-exact SHA256 binding asserted at runtime, NOT pytest). Wired as a step in `wheels.yml` + `release-test.yml` so the wheel smoke MUST pass on every TestPyPI/PyPI publish OR the publish is blocked. CLI exit 0 on pass / 1 on failure with structured JSON report on stderr. 11 new tests. |

## Documentation (cycle-13 F)

`b5fbe064` runtime-integrity-ledger row added under invariant 3 ("Posterior epoch") listing `tests/test_save_state_manifest_contract.py` as the new Python-boundary verifier. **7th adversarial-threats entry** distinguishes the source-correct + binary-stale sub-class from prior 6 declared-vs-verified traps. CLAUDE.md rebuild-after-pull note + `--include-debuginfo` vs `strip=true` workaround. current.json bumped 32d39bdf → b035973e (intermediate updates).

## Cycle-13 E REAL Pro grader result (cumulative through `db304bc6`)

Instance: `instance_NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5-vnan` (different SHA from cgpro's flagged bad NodeBB instance).

```
total_tests: 300, PASSED: 297, FAILED: 3
fail_to_pass_count: 3, f2p_resolved: 0
pass_to_pass_count: 288, p2p_regressed: 0
Cost $0.749 / Latency 203.7s / Tool calls 46 / Turns 28
Topology: sequential 3-node (planner → coder → synthesizer)
Model: sage/gemini-3.1-pro-preview (reasoner tier)
```

**First end-to-end real Pro grader result in YGN-SAGE history.** Agent attempted but didn't resolve the 3 fail_to_pass tests; importantly **0/288 pass_to_pass regressed** — ideal failure mode (didn't break anything). Smoke result doc at `sage-python/docs/benchmarks/2026-05-05-cycle13-arm-d-reasoner-n1.md`.

Cycle-13 main run cost projection from real data: ~$37.50 / ~3-4h wall-clock for arm D N=50 alone.

## cgpro discipline this segment

**Single conversation thread** `cgpro_pi_mono_pivot_20260505` resumed across 8 pre-commit deep VERIFY rounds + 5 post-push `NEXT_BLOCK_ID` reports. **Token-economy-friendly**: cgpro keeps repo state cached, resumed turns are fast.

**3 HARD_STOP fixes** caught real consistency bugs:

1. **Cycle-13 I (a14_reset orphan cleanup)**: I had hardcoded `"topology_state_manifest.json"` as a string literal in `_ATOMIC_WRITTEN_NAMES` while sourcing the OTHER two via constants. cgpro caught the inconsistency: future rename of `TOPOLOGY_STATE_MANIFEST_FILENAME` would silently miss the PRIMARY orphan class. Fix: import the constant + add explicit single-purpose test `test_orphan_tmp_files_includes_topology_state_manifest`.

2. **Cycle-13 G Rust (build-info exposure)**: hardcoded `cargo:rerun-if-changed=../.git/HEAD` and `../.git/refs/heads`. Wrong for (a) PyPI sdist source builds where `.git` is absent (Cargo treats non-existent rerun-trigger as "rebuild every time", destroying cache), (b) git worktrees where `.git` is a file pointing at the real git-dir. Fix: use `git rev-parse --git-path <sub>` to resolve paths via git's own logic + emit `cargo:rerun-if-changed` only when resolved path exists. Added `packed-refs` to the loop.

3. **Cycle-13 G Python (sage_core_version helper)**: `get_source_head_sha()` ran `git rev-parse HEAD` from arbitrary cwd. Operator running `python -m sage.ops.sage_core_version` from inside an UNRELATED git repo would compare YGN-SAGE wheel's commit_sha against THAT repo's HEAD. Fix: `git rev-parse --show-toplevel` + sentinel-file check (`sage-core/Cargo.toml` + `sage-python/src/sage/__init__.py`) before returning HEAD. Returns "unknown" when cwd is not a YGN-SAGE checkout.

Plus 4 minor Q-trap responses (subprocess argv-list security, --strict default, OverflowError in `datetime.fromtimestamp`, `sage.ops.*` vs `tools/` location) — all preserved.

## Net delivery — autonomous segment 2026-05-05 → 2026-05-06

**20 commits total** since cycle-12 P6-A Phase B closeout `70f3cf4b`:

1. **Cycle-12 P6-A Phase B closure** (4 commits, 18/18 CI green): `9f7783cc / 7e20372e / 8761f0db / 70f3cf4b`.
2. **Cycle-13 E** (8 commits): scaffolding `6710eb0b`, NO-API canary `cdaa7594`, REAL prod bug fix event_log shadowing `d3fc6fe0`, smoke results `15fc82eb`, doc-bump `c34bcdee`, first Pro grader result `84aed606`, Lite→Pro converter `3c88ca3b`, Tier 2.1 closure (real reasoner-tier graded) `6ad0cff9`.
3. **Yann's CSV regression test** `7c44e43b`.
4. **Cycle-13 B chain** (7 commits): `32d39bdf / b5fbe064 / bc662d9a / b25c28a6 / b035973e / 9e426504 / db304bc6`.

**Test surface**: 3089 Python (+68 cycle-13 prelude over cycle-12 closure 3021) / 553 Rust / 100 sage-discover. mypy 0 / ruff clean.

**HEAD `db304bc6`** aligned `origin/main`. CI runs in flight at session-end; previous SHAs `9e426504` etc. all GREEN.

## Open follow-ups (cycle-13 main + later cycles)

- **A**. Cycle-13 main run wiring (arms A/B/C, 3-5 days, $240-460 budget). Validates the harness-effect hypothesis (≥5pp pass@1 lift vs Claude Code direct).
- **C**. `sage run --jsonl` v0 protocol gaps (`cli_progress` heartbeat / `set_budget` mid-run / `cancel` cancellation token / `cli_complete.payload.final_seq`). Per cgpro DESIGN E trap Q5.
- **D**. Patch repair budget extension for diff_verifier observe → repair mode.
- **J**. ADR-015 Phase C façade rewrite + 6 stub deletion (pipeline.py 1801 → thin facade, pure refactor 0 behavioral changes, 3-5 days). Yann reminded me this is open from cycle-12.

## cgpro/VERIFY transcripts (read before resuming)

- `.tmp/cgpro_e_design.md` (E DESIGN), `.tmp/cgpro_e_tier_2_1_verify.md` + `.tmp/cgpro_e_tier_2_1_post_push.md` (Tier 2.1 closure)
- `.tmp/cgpro_b_precommit.md` + `.tmp/cgpro_b_deep_followup.md` (B Q1-Q5 deep VERIFY)
- `.tmp/cgpro_b_q4_precommit.md` (Q4 Rust .tmp cleanup)
- `.tmp/cgpro_i_precommit.md` + `.tmp/cgpro_i_precommit_v2.md` (Q4-bis HARD_STOP + fix)
- `.tmp/cgpro_g_rust_precommit.md` + `.tmp/cgpro_g_rust_precommit_v2.md` (Q1 Rust HARD_STOP + fix)
- `.tmp/cgpro_g_python_precommit.md` + `.tmp/cgpro_g_python_precommit_v2.md` (Q1 Python HARD_STOP + fix)
- `.tmp/cgpro_h_precommit.md` + `.tmp/cgpro_h_clarify.md` (Q2 wheel smoke)

## Niche claim still locked

(cgpro 2026-05-05): *"The coding agent that can show why it chose a topology, why it trusted or rejected a result, and why it did or did not learn from the run."* Promesse: *"Verified where possible, evidence-gated everywhere."*

Cycle-13 B chain reinforces the "verified where possible" promise: the runtime-integrity layer is now **operationally observable + boot-time inspectable + release-pipeline gated** at every layer where the contract could silently break.
