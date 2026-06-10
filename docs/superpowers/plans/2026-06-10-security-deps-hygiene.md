# Security Dependency Upgrades + Repo Hygiene (Plan A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Security + latest-deps CI workflows green again by clearing all June 2026 dependency advisories (1 HIGH wasmtime-wasi sandbox CVE + 10 Python CVEs/PYSECs in 5 packages), then close the 4-week hygiene gap (stale local wheel, orphan benchmark artifacts, stale current.json).

**Architecture:** Lockfile/constraints-only dependency bumps (no API migrations — wasmtime fix is a patch release within v44). Python bumps go through the canonical `scripts/compile_python_constraints.sh` regen under WSL (Linux is the constraints reference platform), extended minimally to support targeted `--upgrade-package`. Hygiene commits follow existing repo conventions (benchmark artifacts are committed evidence).

**Tech Stack:** cargo update (Rust lockfile), pip-tools 7.5.3 under WSL Ubuntu 24.04, maturin (local wheel), repo gate scripts (status_snapshot.py → sync_doc_counters.py order is mandatory).

**Scope note:** Block 3 (`B2_RERUN_UNBLOCKERS` — 3 prod bugs, 9 pre-specified tests) is deliberately OUT of this plan. It touches runtime semantics and gets its own plan + cgpro DESIGN/VERIFY round per project protocol (CLAUDE.md cycle pattern). This plan is mechanical maintenance; cgpro is not invoked for it.

**Advisory facts (verified 2026-06-10 from CI run logs + rustsec.org):**

| Ecosystem | Package | Current | Fixed | Advisory | Severity |
|---|---|---|---|---|---|
| Rust | wasmtime-wasi | 44.0.1 | **44.0.2** (`>=44.0.2, <45.0.0` patched) | RUSTSEC-2026-0149 / GHSA-2r75-cxrj-cmph / CVE-2026-47261 | HIGH 7.5 — WASI `path_open(TRUNCATE)` bypasses `FilePerms::WRITE` |
| Rust | rand 0.9.2 (+dup 0.8.5) | 0.9.2 / 0.8.5 | 0.9.3 / 0.8.6 | RUSTSEC-2026-0097 (unsound, informational) | INFO |
| Rust | gimli | 0.33.1 (yanked) | any non-yanked 0.33.x | yanked warning | INFO |
| Rust | paste 1.0.15, async-std 1.13.2, number_prefix 0.4.0 | — | none (unmaintained, transitive) | RUSTSEC-2024-0436 / 2025-0052 / 2025-0119 | INFO — already tolerated as `informational_warnings`, NOT the CI failure cause |
| Python | aiohttp | 3.13.5 | 3.14.0 | CVE-2026-47265, CVE-2026-34993 | — |
| Python | idna | 3.14 | 3.15 | CVE-2026-45409 | — |
| Python | pydantic-ai-slim | 1.94.0 | 1.99.0 | CVE-2026-46678 | — |
| Python | pyjwt | 2.12.1 | 2.13.0 | PYSEC-2026-175/177/178/179 | — |
| Python | starlette | 1.0.0 | 1.0.1 | PYSEC-2026-161 | — |

All 5 Python packages are transitives or floor-pinned (`pydantic-ai-slim>=1.84`, no cap) — pyproject.toml needs NO edits; constraints regen with targeted upgrades suffices. pip-audit in CI audits the LOCKED constraints.txt files (sage-python + sage-discover), so fixing the pins fixes the gate.

**Environment facts:**
- Windows pip-tools venv: `.venv-piptools/` (repo root). WSL venv: `.venv-linux-piptools/` (untracked, must be gitignored).
- Linux is the constraints reference platform (CI freshness gate regenerates on Linux and diffs). Regen MUST run under WSL: `wsl -e bash -c '...'` with `PYTHON=<repo>/.venv-linux-piptools/bin/python`.
- The regen script builds a sage-core wheel via maturin first; if cargo is unavailable in WSL, reuse the existing manylinux wheel in `sage-core/target/wheels/` (May 12 session lesson) and run the two pip-compile blocks directly with identical flags.
- Local dev python: `C:\ProgramData\miniforge3\python.exe`. Local `sage_core` wheel is stale (built at `b0df5bb9`, missing `c9bc6bc3` model_assigner fixes) — Task 4 rebuilds it.
- maturin develop workaround on this machine (strip=true conflict): `maturin build --release --features smt,onnx --out target/wheels` then `pip install target/wheels/sage_core-0.1.0-cp313-*.whl --force-reinstall --no-deps`.
- Rust tests on Windows git-bash need `PATH="/c/ProgramData/miniforge3:$PATH"` so the test exe finds `python313.dll`.

---

### Task 1: Rust lockfile advisory bumps (wasmtime-wasi 44.0.2 + rand + gimli)

**Files:**
- Modify: `Cargo.lock` (repo root, via cargo update — never hand-edit)
- Modify: `sage-core/Cargo.toml:25` (comment only — document the 44.0.2 advisory)

- [x] **Step 1: Bump the four advisory-affected crates in the lockfile**

```bash
cd /c/Code/YGN-SAGE
cargo update -p wasmtime-wasi --precise 44.0.2
cargo update -p wasmtime --precise 44.0.2
cargo update -p rand@0.9.2 --precise 0.9.3
cargo update -p rand@0.8.5 --precise 0.8.6
cargo update -p gimli@0.33.1
```

Expected: lockfile updates only; `wasmtime-wasi-io`/`wiggle` follow wasmtime to 44.0.2 automatically. If `--precise 44.0.2` complains about a sibling crate version requirement, run `cargo update -p wasmtime-wasi -p wasmtime -p wasmtime-wasi-io -p wiggle` (un-precise within semver range) and verify the resolved version is >=44.0.2.

- [x] **Step 2: Verify the lockfile resolution**

```bash
grep -A1 'name = "wasmtime-wasi"' Cargo.lock | head -4
grep -A1 'name = "rand"' Cargo.lock | head -6
grep -A1 'name = "gimli"' Cargo.lock | head -4
```

Expected: `wasmtime-wasi` version `44.0.2`; rand `0.8.6` and `0.9.3`; gimli not `0.33.1`.

- [x] **Step 3: Update the Cargo.toml provenance comment**

In `sage-core/Cargo.toml` around line 25, extend the existing comment block (which documents the v43→v44 bump for 11 RustSec advisories) with one line noting: lockfile floor is 44.0.2 since 2026-06-10 for RUSTSEC-2026-0149 (WASI path_open TRUNCATE bypass, CVSS 7.5). Keep `version = "44.0"` unchanged (44.0.2 satisfies it).

- [x] **Step 4: Build + run the Rust suite with the bumped deps**

```bash
cd /c/Code/YGN-SAGE/sage-core
PATH="/c/ProgramData/miniforge3:/c/ProgramData/miniforge3/Library/bin:$PATH" cargo test --features smt --lib 2>&1 | tail -3
```

Expected: `test result: ok. 581 passed; 0 failed; 3 ignored` (same as the 2026-06-10 baseline run). Background this (~16 min: wasm sandbox tests dominate) and proceed to Task 2 while it runs; do NOT commit before it lands green.

- [x] **Step 5: cargo fmt gate + commit**

```bash
cd /c/Code/YGN-SAGE/sage-core && cargo fmt --check
cd /c/Code/YGN-SAGE
git add Cargo.lock sage-core/Cargo.toml
git commit -m "fix(security): bump wasmtime-wasi 44.0.2 + rand + gimli (RUSTSEC-2026-0149)

cargo-audit + cargo-deny have been red on the weekly Security run since
2026-06-01. Root causes and fixes, all lockfile-range bumps (Cargo.toml
pins unchanged, no API changes):

- wasmtime-wasi 44.0.1 -> 44.0.2: RUSTSEC-2026-0149 / GHSA-2r75-cxrj-cmph
  / CVE-2026-47261 (HIGH 7.5) — WASI path_open(TRUNCATE) bypasses
  FilePerms::WRITE. SAGE's deny-by-default WASI contract grants no
  preopened dirs, so guest code had no path to exploit it, but the
  sandbox claim (security.sandbox_default_wasi) must not sit on a known
  advisory.
- rand 0.9.2 -> 0.9.3 and transitive rand 0.8.5 -> 0.8.6:
  RUSTSEC-2026-0097 (unsound ThreadRng+custom logger, informational).
- gimli 0.33.1 (yanked) -> current 0.33.x: clears the yanked warning.

paste/async-std/number_prefix unmaintained warnings remain documented
deny.toml informational tolerances (transitive, no patched releases).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 2: Extend constraints script with targeted upgrades

**Files:**
- Modify: `scripts/compile_python_constraints.sh:6-13` (arg parsing only)

- [x] **Step 1: Add `--upgrade-package <name>` passthrough (repeatable)**

Replace lines 6-13 of `scripts/compile_python_constraints.sh`:

```bash
UPGRADE_FLAG=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --upgrade)
      UPGRADE_FLAG+=(--upgrade)
      shift
      ;;
    --upgrade-package)
      if [[ -z "${2:-}" ]]; then
        echo "usage: $0 [--upgrade] [--upgrade-package NAME]..." >&2
        exit 2
      fi
      UPGRADE_FLAG+=(--upgrade-package "$2")
      shift 2
      ;;
    *)
      echo "usage: $0 [--upgrade] [--upgrade-package NAME]..." >&2
      exit 2
      ;;
  esac
done
```

Rationale: full `--upgrade` would let `pydantic-ai-slim>=1.84` (uncapped) jump majors; targeted `--upgrade-package` (native pip-compile flag, applied to both compile blocks via the existing `"${UPGRADE_FLAG[@]}"` expansion) bumps only the advisory packages. The CI freshness gate re-runs the script with NO args and diffs — pip-compile without upgrade flags keeps existing satisfying pins, so the gate stays byte-stable after this commit.

- [x] **Step 2: Syntax-check the script**

```bash
bash -n /c/Code/YGN-SAGE/scripts/compile_python_constraints.sh && echo SYNTAX_OK
```

Expected: `SYNTAX_OK`.

- [x] **Step 3: Commit (script change separate from generated output)**

```bash
cd /c/Code/YGN-SAGE
git add scripts/compile_python_constraints.sh
git commit -m "feat(deps): targeted --upgrade-package mode for constraints regen

Full --upgrade is the only upgrade lever the script had; with
pydantic-ai-slim floor-pinned (>=1.84, no cap) a full upgrade can jump
majors. pip-compile's native --upgrade-package gives a surgical path for
advisory-driven bumps. Repeatable flag, applied to both compile blocks.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 3: Regenerate Python constraints under WSL (advisory bumps)

**Files:**
- Modify: `sage-python/constraints.txt` (generated)
- Modify: `sage-discover/constraints.txt` (generated)

- [x] **Step 1: Verify the WSL venv toolchain**

```bash
wsl -e bash -lc 'cd /mnt/c/Code/YGN-SAGE && .venv-linux-piptools/bin/python --version && .venv-linux-piptools/bin/python -m piptools --version 2>/dev/null || echo PIPTOOLS_MISSING; command -v cargo >/dev/null && echo CARGO_OK || echo CARGO_MISSING'
```

Expected: Python 3.12.x + pip-tools 7.5.3. If `CARGO_MISSING`, confirm an existing manylinux `sage_core` wheel in `sage-core/target/wheels/` (`ls sage-core/target/wheels/sage_core-*manylinux*.whl`) — the script's maturin step will be bypassed by running the pip-compile blocks directly (Step 2 fallback).

- [x] **Step 2: Run the targeted regen (both packages)**

```bash
wsl -e bash -lc 'cd /mnt/c/Code/YGN-SAGE && PYTHON=$PWD/.venv-linux-piptools/bin/python ./scripts/compile_python_constraints.sh --upgrade-package aiohttp --upgrade-package idna --upgrade-package pydantic-ai-slim --upgrade-package pyjwt --upgrade-package starlette 2>&1 | tail -5'
```

Expected: exits 0, regenerates both constraints.txt. Fallback if the maturin step fails in WSL: run the two `piptools compile` blocks from the script manually with identical flags + the five `--upgrade-package` args, from `sage-python/` then `sage-discover/` cwd (per-package cwd is mandatory — May 12 CI-parity lesson).

- [x] **Step 3: Verify the diff is advisory-only**

```bash
cd /c/Code/YGN-SAGE && git diff --stat sage-python/constraints.txt sage-discover/constraints.txt
git diff sage-python/constraints.txt | grep -E "^[+-][a-z0-9-]+==" | sort | head -30
```

Expected: `-aiohttp==3.13.5 / +aiohttp==3.14.0`, `-idna==3.14 / +idna==3.15`, `-pydantic-ai-slim==1.94.0 / +pydantic-ai-slim==1.99.0`, `-pyjwt==2.12.1 / +pyjwt==2.13.0`, `-starlette==1.0.0 / +starlette==1.0.1` (+ possible minimal transitive ripples of pydantic-ai-slim 1.99 — e.g. its own floor bumps). If pydantic-ai-slim resolves above 1.99.x or drags unrelated major bumps, STOP and inspect before committing.

- [x] **Step 4: Upgrade the same five packages in the local miniforge env and run the affected test slices**

```bash
/c/ProgramData/miniforge3/python.exe -m pip install "aiohttp==3.14.0" "idna==3.15" "pydantic-ai-slim==1.99.0" "pyjwt==2.13.0" "starlette==1.0.1" 2>&1 | tail -2
cd /c/Code/YGN-SAGE/sage-python
/c/ProgramData/miniforge3/python.exe -m pytest tests/test_pydantic_ai_integration.py tests/test_providers_registry.py tests/test_provider_policy*.py tests/test_provider_execution_witness.py -q 2>&1 | tail -3
```

Expected: existing pass/skip profile (API-key-gated tests skip; no NEW failures vs the suite state at HEAD). pydantic-ai-slim 1.94→1.99 is the only behavior-bearing bump — if its tests break, check the pydantic-ai 1.95-1.99 changelog (Context7 `/pydantic/pydantic-ai`) before any code adaptation, and keep adaptations in a separate commit.

- [x] **Step 5: Commit the regenerated constraints**

```bash
cd /c/Code/YGN-SAGE
git add sage-python/constraints.txt sage-discover/constraints.txt
git commit -m "fix(security): constraints regen — clear 10 pip-audit advisories in 5 packages

Weekly Security pip-audit has been red since 2026-06-01. Targeted
Linux-side (WSL) regen via compile_python_constraints.sh
--upgrade-package, no pyproject changes (all five are transitives or
floor-pinned):

- aiohttp  3.13.5 -> 3.14.0  (CVE-2026-47265, CVE-2026-34993)
- idna     3.14   -> 3.15    (CVE-2026-45409)
- pydantic-ai-slim 1.94.0 -> 1.99.0 (CVE-2026-46678)
- pyjwt    2.12.1 -> 2.13.0  (PYSEC-2026-175/177/178/179)
- starlette 1.0.0 -> 1.0.1   (PYSEC-2026-161)

Local validation: pydantic-ai/provider test slices green on the bumped
versions.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 4: Local wheel rebuild + stale-binary defense re-verify

**Files:** none committed (local environment only)

- [x] **Step 1: Rebuild sage_core wheel at HEAD (post-Task-1 lockfile)**

```bash
cd /c/Code/YGN-SAGE/sage-core
/c/ProgramData/miniforge3/python.exe -m maturin build --release --features smt,onnx --out target/wheels 2>&1 | tail -3
/c/ProgramData/miniforge3/python.exe -m pip install target/wheels/sage_core-0.1.0-cp313-*.whl --force-reinstall --no-deps 2>&1 | tail -2
```

Expected: wheel builds; install succeeds. (Direct `maturin develop` hits the documented strip=true conflict on this machine.)

- [x] **Step 2: Verify the L3 stale-binary check now passes**

```bash
cd /c/Code/YGN-SAGE && /c/ProgramData/miniforge3/python.exe -m sage.ops.sage_core_version --strict; echo "exit=$?"
```

Expected: `matches: true`, `commit_sha` = current HEAD, exit=0.

- [x] **Step 3: Run the wheel smoke (L4 defense) + manifest contract (L2)**

```bash
cd /c/Code/YGN-SAGE && /c/ProgramData/miniforge3/python.exe -m sage.ops.wheel_smoke 2>&1 | tail -3
cd /c/Code/YGN-SAGE/sage-python && /c/ProgramData/miniforge3/python.exe -m pytest tests/test_save_state_manifest_contract.py -q 2>&1 | tail -2
```

Expected: wheel_smoke exit 0 (4 phases pass); manifest contract 2/2 PASS.

### Task 5: Hygiene — orphan artifacts, gitignore, current.json

**Files:**
- Modify: `.gitignore` (add `.venv-linux-piptools/`)
- Create (commit existing untracked): `docs/benchmarks/2026-05-11-canary-n5-graded/`, `docs/benchmarks/2026-05-11-canary-n5-graded-slice8/`, `docs/benchmarks/2026-05-11-canary-n5-graded-postfix/`, `sage-python/docs/benchmarks/2026-05-12-e2e-campaign.json`
- Modify: `docs/status/current.json` (regenerated)

- [x] **Step 1: Gitignore the WSL venv**

Add to `.gitignore` next to the existing venv entries: `.venv-linux-piptools/`.

- [x] **Step 2: Elucidate the e2e campaign c7 gap before committing the artifact**

```bash
grep -nE '"c7"|def test_c7|c7' /c/Code/YGN-SAGE/sage-python/tests/test_e2e_campaign.py | head -10
grep -l '"c7"' /c/Code/YGN-SAGE/sage-python/docs/benchmarks/2026-05-10-e2e-campaign.json /c/Code/YGN-SAGE/docs/benchmarks/2026-05-08-e2e-campaign.json 2>/dev/null
```

Expected: identifies what c7 covers and whether prior campaign files include it. Outcome decides the commit message wording: either c7 was skipped intentionally (note it) or it crashed mid-campaign on 2026-05-12 (note it as a known gap; do NOT re-run paid work in this plan).

- [x] **Step 3: Commit the orphan benchmark evidence**

```bash
cd /c/Code/YGN-SAGE
git add .gitignore docs/benchmarks/2026-05-11-canary-n5-graded docs/benchmarks/2026-05-11-canary-n5-graded-slice8 docs/benchmarks/2026-05-11-canary-n5-graded-postfix sage-python/docs/benchmarks/2026-05-12-e2e-campaign.json
git commit -m "bench: commit orphan 2026-05-11/12 canary + e2e artifacts left by B2 close

Three intermediate graded-canary runs (predecessors of the committed
2026-05-12-b2-n5-graded final) and the 2026-05-12 e2e campaign JSON were
left untracked when the May 12 session closed on NO_GO_N50. Committed
as-is for provenance. [c7 note from Step 2]. Also gitignores the WSL
pip-tools venv (.venv-linux-piptools/).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [x] **Step 4: Regenerate current.json (canonical order: snapshot THEN propagate)**

```bash
cd /c/Code/YGN-SAGE
/c/ProgramData/miniforge3/python.exe scripts/status_snapshot.py 2>&1 | tail -3
/c/ProgramData/miniforge3/python.exe scripts/sync_doc_counters.py 2>&1 | tail -3
/c/ProgramData/miniforge3/python.exe scripts/sync_doc_counters.py --check 2>&1 | tail -2
git diff --stat
```

Expected: current.json regenerated at HEAD (3662 Python / 584 Rust / 100 discover at the 2026-06-10 baseline; counts may not change if no tests were added — the SHA + dirty flag must update). sync check: `No drift detected`.

- [x] **Step 5: Commit the status refresh**

```bash
cd /c/Code/YGN-SAGE
git add docs/status/current.json $(git diff --name-only | grep -E "README|CLAUDE|\.claude/rules" || true)
git commit -m "chore(status): refresh current.json at post-security-bump HEAD

status_snapshot.py then sync_doc_counters.py (canonical order). Closes
the 3659-vs-3662 collect drift (current.json was generated at 0cdf253c,
6 commits behind the May 12 close).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 6: Push, CI verification, remaining red workflows

**Files:** none (CI observation + possible follow-up fixes)

- [x] **Step 1: Final local gates then push**

```bash
cd /c/Code/YGN-SAGE/sage-python
/c/ProgramData/miniforge3/python.exe -m ruff check src/ tests/ 2>&1 | tail -2
/c/ProgramData/miniforge3/python.exe -m mypy src/sage/ --ignore-missing-imports 2>&1 | tail -2
cd /c/Code/YGN-SAGE
/c/ProgramData/miniforge3/python.exe -m sage.ops.claims_audit --strict 2>&1 | tail -2
/c/ProgramData/miniforge3/python.exe scripts/narrative_guard_phase22.py 2>&1 | tail -1
git push origin main
```

Expected: ruff clean, mypy 0 errors, claims OK, guard PASS, push accepted.

- [x] **Step 2: Watch the push-triggered workflows to completion**

```bash
gh run list --limit 8 --json status,conclusion,workflowName,headSha --jq '.[] | "\(.workflowName): \(.conclusion // .status) (\(.headSha[:8]))"'
gh run watch $(gh run list --workflow Security --limit 1 --json databaseId --jq '.[0].databaseId') --exit-status 2>&1 | tail -3
```

Expected: Security (cargo-audit, cargo-deny, pip-audit) GREEN; CI green; both coherence gates green. If Security still red, read the failing job log and fix root-cause (no gate suppression) before continuing.

- [x] **Step 3: Triage the failing "Latest Python dependencies" weekly workflow**

```bash
RUN_ID=$(gh run list --workflow "Latest Python dependencies" --limit 1 --json databaseId --jq '.[0].databaseId')
gh run view $RUN_ID --log-failed 2>&1 | grep -iE "error|ERROR|failed" | head -20
```

Classify: (a) same advisory pins → already fixed by this plan, re-run via `gh workflow run latest-deps.yml`; (b) genuine latest-version incompatibility (e.g. mypy 2.x class) → root-cause fix if ≤1h scope, else document in a dated note under `docs/status/` and queue as its own ticket. No sweeping under the rug; no `|| true`.

- [x] **Step 4: Dependabot — failing updater runs + open checkout PR**

```bash
gh pr list --json number,title,headRefName --jq '.[] | "#\(.number) \(.title)"'
gh run list --workflow "Dependabot Updates" --limit 2 --json databaseId --jq '.[].databaseId' | head -1 | xargs -I{} gh run view {} --log-failed 2>&1 | grep -iE "error" | head -10
```

For the `actions/checkout` 6.0.3 PR: after main is green, comment `@dependabot rebase`, wait for green checks (Security included), then `gh pr merge --squash --auto`. For the failing Dependabot Updates runs: identify the updater error (likely a manifest it cannot parse or a grouping config issue) and fix `.github/dependabot.yml` if that is the cause.

- [x] **Step 5: Close out**

Report: advisories cleared (with before/after versions), CI statuses, anything deferred with reasons. Update `docs/status/` is NOT needed beyond current.json (no semantic change). Then proceed to Plan B (`B2_RERUN_UNBLOCKERS`) with its own plan + cgpro round.

---

## Self-Review (done at write time)

1. **Spec coverage**: Block 1 (Rust T1, Python T2-T3, CI T6) ✓; Block 2 (wheel T4, artifacts+current.json T5) ✓; Block 3 explicitly out-of-scope with rationale ✓.
2. **Placeholder scan**: one intentional bracket — the c7 commit-message note in Task 5 Step 3 depends on Step 2's finding (investigation step precedes it; wording resolved at execution).
3. **Type/command consistency**: all paths verified against the live repo this session (script lines 6-13, constraints locations, venv paths, advisory versions from cargo-audit JSON).

**Known risks:** (1) WSL venv may lack cargo → documented fallback (existing manylinux wheel + direct pip-compile). (2) pydantic-ai-slim 1.99 behavior drift → test slice in T3 Step 4 + Context7 changelog check before any adaptation. (3) `--precise` version dance on linked wasmtime crates → documented alternate command. (4) Constraints freshness gate expects byte-stable no-arg regen → guaranteed by pip-compile pin-keeping semantics; CI will prove it.
