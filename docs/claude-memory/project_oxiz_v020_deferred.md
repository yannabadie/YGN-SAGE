---
name: OxiZ v0.2.0 — upgraded 2026-04-21 (was deferred)
description: Bumped from 0.1 to 0.2 in commit 4aa29e7 after audit (docs/audits/OxiZ v0.2.0 upgrade — unparks the bump). QualityLabeler equivalence verified. No longer deferred.
type: project
originSessionId: e6496ce0-f81e-4f1f-bc19-bd2fd75b67ef
---
**Status:** UPGRADED 2026-04-21 in commit `4aa29e7` (chore(deps): bump oxiz 0.1 → 0.2 (QualityLabeler SMT backend)). Preceded by audit commit `ca12d63` (docs(audits): OxiZ v0.2.0 upgrade audit — unparks the bump).

**Why it was deferred originally (2026-04-20):** `cool-japan/oxiz` v0.2.0 was a 300-file workspace restructure (`oxiz-core/` + `bench/` + `oxiz-cli/` split) with no release notes. Raw upgrade would have almost certainly broken the QualityLabeler wiring on module-path changes.

**How the unpark happened:** Audit in `docs/audits/` (date stamp 2026-04-21) walked the equivalence: the public SMT-solver surface QualityLabeler touches was preserved across the restructure. Bump landed and tests stayed green.

**Current pin:** `sage-core/Cargo.toml — oxiz = { version = "0.2", optional = true }`. SMT feature gate unchanged.

**How to apply going forward:** For future 0.2.x → 0.3 bumps, repeat the same pattern: read upstream CHANGELOG inside the repo (release body is still thin), run the QualityLabeler equivalence test suite, THEN bump. Don't reopen this as an active concern unless a new major lands.
