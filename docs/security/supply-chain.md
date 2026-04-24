# Supply-chain security — baseline gates

This document tracks the minimum-viable supply-chain posture introduced
for AUDIT-annex ticket **A17** (see
`docs/audits/2026-04-24-audit3-triage/AUDIT1-AUDIT2-annex.md`). The
finding motivating this work: prior to A17, the repository shipped
zero automated dependency-advisory gates, ran GitHub Actions pinned by
floating tag, and did not use Trusted Publishing on PyPI.

## Gates at a glance

| Gate            | Workflow job          | Status (initial landing)      | Scope                                   |
|-----------------|-----------------------|-------------------------------|-----------------------------------------|
| CodeQL          | `codeql`              | enforced                      | Python static analysis                  |
| SBOM export     | `sbom`                | enforced                      | CycloneDX JSON for `sage-python`        |
| pip-audit       | `pip-audit`           | **advisory** (continue-on-error) | Python deps in `sage-python` + `sage-discover` |
| cargo-audit     | `cargo-audit`         | **advisory** (continue-on-error) | RustSec advisories for `sage-core/`     |
| cargo-deny      | `cargo-deny`          | **advisory** (continue-on-error) | License / bans / duplicate dependencies |

"Advisory" means the job runs, surfaces findings in the Actions UI, and
is green-checked even on hit. Once the team has eyeballed the first
two full weekly cycles of output, flip `continue-on-error: false` per
the enforcement timeline below.

## Workflow triggers

All security jobs run on:

- `push` to `main`
- `pull_request` targeting `main`
- weekly `schedule` (Monday 06:00 UTC)

Weekly runs catch newly-disclosed advisories against an otherwise-idle
branch.

## Reading the reports

### pip-audit

`pip-audit` reports any PyPI package in the installed dependency tree
that has an OSV / PyPA advisory. The output format is human-readable
text by default:

```
Found 1 known vulnerability in 1 package
Name    Version ID                  Fix Versions
------- ------- ------------------- ------------
certifi 2023.7  PYSEC-2024-XYZ      2024.2.2
```

- `ID` — OSV or PYSEC advisory identifier. Look it up on
  https://osv.dev/list for severity + upstream fix.
- `Fix Versions` — the minimum upgrade that closes the advisory. If
  empty, no fix has been published yet; document under
  `[advisories].ignore` in `sage-core/deny.toml` (Rust only) or track
  in the weekly security review.
- CI status: the `pip-audit` job prints findings and exits non-zero
  (filtered by `continue-on-error: true` during the advisory phase).

### cargo-audit

`cargo-audit` reports RustSec Advisory DB entries. Output format:

```
error: Vulnerable crates found!
Crate:   example-crate
Version: 0.1.2
Title:   Description of CVE
Date:    2026-01-15
ID:      RUSTSEC-2026-0123
URL:     https://rustsec.org/advisories/RUSTSEC-2026-0123
Solution: Upgrade to >=0.2.0
```

- `ID` — RustSec advisory identifier. Full description + technical
  detail at the `URL` line.
- `Solution` — required version bump. Applied via `cargo update -p
  <crate>` or by bumping the direct dep in `Cargo.toml`.

### cargo-deny

`cargo-deny` runs three checks in one pass (`advisories`, `licenses`,
`bans`). Output example:

```
error[L001]: failed to satisfy license requirements
   ┌─ path/to/dep/Cargo.toml:1:1
   │
 1 │ some-crate 1.0.0
   │ ^^^^^^^^^^^^^^^^ license "GPL-3.0" not in the allow list

warning[B001]: duplicate versions for crate `serde_json`
   ┌─ Cargo.toml:1:1
```

- License failures (`L*`) — extend the `[licenses].allow` list in
  `sage-core/deny.toml` ONLY after a license-compatibility review.
- Ban failures (`B*`) — usually mean a new transitive dep introduced
  either a duplicate version (noise) or a hard-denied crate (security).
- Advisory failures (`A*`) — same flow as `cargo-audit`.

## Enforcement timeline

| Phase                      | Window                 | `continue-on-error` | Action on failure                                                |
|----------------------------|------------------------|---------------------|------------------------------------------------------------------|
| 1. Observe (current)       | Landing week           | `true`              | Record findings, no blocking                                     |
| 2. Soft-enforce            | Week 2                 | `true`              | Findings = PR review block (manual); file issues per hit         |
| 3. Hard-enforce            | Week 3+                | `false`             | Findings fail the PR. Overrides require explicit security review |
| 4. Expand                  | Month 2+               | `false` + new gates | Add SBOM scanning, Trusted Publishing verification, etc.         |

Phase-3 flip means editing `security.yml` to remove
`continue-on-error: true` from the three audit jobs. Keep the comment
next to the `jobs:` header that documents the history of the flip.

## Follow-ups (out of A17 scope)

### PyPI Trusted Publishing

The `ygn-sage` package is published via `TWINE_USERNAME` + API-token
credentials stored as GitHub Actions secrets. Migrating to PyPI
Trusted Publishing eliminates the long-lived secret and binds
publishing to the GitHub OIDC identity of the workflow run.

Rollout plan (tracked as a separate ALIRE3 ticket):

1. Configure a Trusted Publisher on https://pypi.org/manage/account/
   pointing at `yannabadie/YGN-SAGE` with the publish workflow name
   and environment.
2. Add a `environment: release` gate on the publish job and a
   `permissions: id-token: write` block.
3. Replace `twine upload` with `pypa/gh-action-pypi-publish@<sha>`.
4. Rotate + revoke the existing `PYPI_API_TOKEN` secret.

### SHA-pinning actions

All third-party actions in `.github/workflows/security.yml` are now
pinned to full commit SHAs with a trailing `# <tag>` comment. The
only exception is `dtolnay/rust-toolchain` which publishes only
floating branch refs (`stable`, `nightly`) and has no signed release
tags; TODO is filed inline in the workflow. Consider mirroring to a
fork for a pinnable artefact.

`.github/workflows/ci.yml` is NOT yet SHA-pinned — scope of A17 was
intentionally limited to the security workflow as the "canary" for
the pinning convention. Extend to `ci.yml` in a follow-up PR once the
SHA-pin comment style proves ergonomic.

### Dependabot

Not yet configured. Enabling `github/dependabot` with a tight update
cadence (weekly for actions, daily for security updates on
`sage-python` + `sage-core`) would automate the bump flow that the
audit gates enforce. Out of A17 scope; filed as a follow-up.

## Related audit tickets

- **A17** (this doc) — minimum-viable CI gates.
- **A18** — `tools/forge.py` hardening (handled in parallel).
- **A19+** — SBOM retention, Trusted Publishing, Dependabot —
  follow-ups.

## Ownership

- Workflow file: `.github/workflows/security.yml`
- cargo-deny config: `sage-core/deny.toml`
- This doc: `docs/security/supply-chain.md`

Questions / advisory reviews / allow-list additions: open an issue
tagged `security` and reference the finding ID (e.g. `RUSTSEC-2026-
0123` or `PYSEC-2026-0045`).
