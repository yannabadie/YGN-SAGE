---
name: status_snapshot.py first, then sync_doc_counters.py
description: Operational discipline locked by cgpro Phase 1.5h round-8 (2026-05-06) — status_snapshot.py is the canonical generator of docs/status/current.json; sync_doc_counters.py only propagates from it.
type: feedback
originSessionId: 88857be6-7048-463a-8ee4-cb3b4cca20fd
---
After ANY variation in the test surface (Python, Rust, sage-discover), the workflow is:

1. **`python scripts/status_snapshot.py`** — generates `docs/status/current.json` from live `pytest --collect-only` + `cargo test --list`. This is the canonical generator.
2. **`python scripts/sync_doc_counters.py`** — propagates the counters FROM `current.json` INTO README/AI-ARCHITECTURE.md/etc.

**Why:** confused this once in Phase 1.5g — ran `status_snapshot.py` after manually patching `current.json` to v2 with 3 SHA fields, which OVERWROTE my v2 fields because the generator was still v1. cgpro Phase 1.5h EDIT_REQUIRED forced fixing the generator itself to emit v2 with `snapshot_commit_sha` + `generated_for_commit_sha` so future regenerations stay strict-gate-clean.

**How to apply:**

- ANY commit that changes test count → run BOTH scripts in sequence (status_snapshot first), then commit `current.json` + downstream docs.
- The `doc-counters-coherence` CI workflow runs `sync_doc_counters.py --check` only — it does NOT run `status_snapshot.py --check`. So a stale `current.json` (out of sync with live counts) can slip past CI if the docs match it. Phase 0.7 future hardening item: add `status_snapshot --check` to the CI workflow.
- The `strict-current-json-coherence` CI workflow checks: `schema_version == v2`, `snapshot_commit_sha == generated_for_commit_sha`, `generated_for_commit_sha` distance to `${{ github.sha }}` ≤ 1.
- Default behavior of `status_snapshot.py` (cycle-13 K Phase 1.5h `560731b2`): emits `schema_version: "v2"` + 3 SHA fields, all = HEAD sha at run time. So a single `python scripts/status_snapshot.py && git add docs/status/current.json && git commit` keeps the strict gate happy.

**Distance ≤ 1 grace window:** if you bump `current.json` in commit N and create commit N+1 without re-running the script, distance becomes 1 — still within grace. Distance becomes 2 = strict-gate FAIL on next push. Recovery: regenerate via `status_snapshot.py` in a follow-up commit (proactive bump pattern, see Phase 1.5f-bump `3d76d67e`).
