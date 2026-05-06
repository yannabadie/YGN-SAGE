"""Claims registry audit for YGN-SAGE.

Cycle-13 K Phase 0.1a (cgpro `Analyse approfondie de repo` 2026-05-06):
the repository's narrative (README, AI-ARCHITECTURE.md, CLAUDE.md) cites
performance numbers, capabilities, and runtime guarantees that are not
always backed by a pinned test or benchmark artefact. This module loads
the per-category YAML files under `docs/claims/*.yaml` and audits each
claim against four contracts:

  1. `evidence_test`        : a pytest node id that exists in the test
                              collection (resolved via pytest --collect-only)
  2. `evidence_benchmark`   : a file under `docs/benchmarks/` or
                              `docs/audits/` that exists on disk
  3. `evidence_commit`      : a SHA reachable from `main`
  4. `last_verified_utc`    : an ISO 8601 timestamp older than 90 days
                              triggers a warning (not an error)

Status vocabulary:
  delivered, default-on, opt-in, planned, parked, retired, evidence_pending

The runtime contract is:
  * status in {delivered, default-on}
      → MUST have evidence_test AND evidence_commit (else violation)
      → SHOULD have evidence_benchmark (else warning, never error)
  * status in {opt-in, planned, parked, retired, evidence_pending}
      → evidence_* fields are optional; "evidence_pending" string is the
        explicit placeholder used until Phase 0.4 pins them
  * any claim is allowed to use the literal string "evidence_pending" or
    "n/a" instead of a real path / SHA when the field is genuinely not
    applicable. AUDIT mode warns; STRICT mode escalates only for
    delivered / default-on missing evidence_test or evidence_commit.

Modes:
  AUDIT  (default in Phase 0.1a): print warnings, exit 0.
  STRICT (Phase 0.1b onwards): exit 1 on any contract violation.

The CLI prints a structured JSON report to stdout when --json is set,
otherwise a human-readable summary.

Usage:
  python -m sage.ops.claims_audit                  # AUDIT, human output
  python -m sage.ops.claims_audit --strict         # STRICT, human output
  python -m sage.ops.claims_audit --json           # AUDIT, JSON output
  python -m sage.ops.claims_audit --strict --json  # STRICT, JSON output
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

log = logging.getLogger("sage.ops.claims_audit")

# Status vocabulary — kept in sync with docs/claims/routing.yaml header comment.
_VALID_STATUSES: frozenset[str] = frozenset({
    "delivered",
    "default-on",
    "opt-in",
    "planned",
    "parked",
    "retired",
    "evidence_pending",
})

# Statuses that REQUIRE evidence_test + evidence_commit (else STRICT fails).
_EVIDENCE_REQUIRED_STATUSES: frozenset[str] = frozenset({"delivered", "default-on"})

_EVIDENCE_PLACEHOLDERS: frozenset[str] = frozenset({"evidence_pending", "n/a", ""})

_REQUIRED_CLAIM_FIELDS: tuple[str, ...] = (
    "id",
    "status",
    "description",
    "evidence_test",
    "evidence_benchmark",
    "evidence_commit",
    "last_verified_utc",
    "owner",
)

_STALE_THRESHOLD = timedelta(days=90)


@dataclass
class Violation:
    """A single contract violation found during audit."""

    claim_id: str
    severity: str  # "error" | "warning"
    field_name: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class AuditReport:
    """Structured report of an audit run."""

    total_claims: int
    by_status: dict[str, int] = field(default_factory=dict)
    violations: list[Violation] = field(default_factory=list)
    mode: str = "audit"
    ok: bool = True

    @property
    def errors(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "error"]

    @property
    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "warning"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "mode": self.mode,
            "total_claims": self.total_claims,
            "by_status": self.by_status,
            "errors": [v.to_dict() for v in self.errors],
            "warnings": [v.to_dict() for v in self.warnings],
        }


def _claims_dir(repo_root: Path) -> Path:
    return repo_root / "docs" / "claims"


def load_claims(repo_root: Path) -> list[dict[str, Any]]:
    """Load and concatenate every `docs/claims/*.yaml` file.

    Each YAML file MUST have a top-level `claims:` list. Files without
    that shape are skipped with a warning — `docs/CLAIMS.yaml` itself
    has the same shape so loading it would double-count.
    """
    claims_dir = _claims_dir(repo_root)
    if not claims_dir.is_dir():
        return []

    out: list[dict[str, Any]] = []
    for yaml_path in sorted(claims_dir.glob("*.yaml")):
        # Skip the generated index; it would otherwise double-count.
        if yaml_path.name == "CLAIMS.yaml":
            continue
        try:
            payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            log.warning("YAML parse error in %s: %s", yaml_path, exc)
            continue

        if not isinstance(payload, dict) or not isinstance(payload.get("claims"), list):
            log.warning("Skipping %s: no top-level `claims:` list", yaml_path)
            continue

        for entry in payload["claims"]:
            if isinstance(entry, dict) and "id" in entry:
                entry["_source_file"] = str(yaml_path.relative_to(repo_root)).replace("\\", "/")
                out.append(entry)
    return out


def _is_placeholder(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in _EVIDENCE_PLACEHOLDERS


def _evidence_test_resolves(repo_root: Path, value: str) -> bool:
    """Best-effort resolution of a pytest node id.

    A full pytest collection round-trip is too slow for a CLI gate, so
    this function only verifies that the FILE part of the node id exists
    on disk. The test-name part (after `::`) is a soft assertion: a
    later strict pass can run pytest --collect-only against the file.
    """
    if "::" in value:
        file_part = value.split("::", 1)[0]
    else:
        file_part = value
    return (repo_root / file_part).is_file()


def _evidence_benchmark_resolves(repo_root: Path, value: str) -> bool:
    return (repo_root / value).is_file()


_SHA_PATTERN = re.compile(r"^[0-9a-f]{7,40}$")


def _evidence_commit_well_formed(value: str) -> bool:
    """We don't reach git here for portability; just shape-validate the SHA.

    A later step (`scripts/sync_doc_counters.py` or a CI step) can run
    `git merge-base --is-ancestor <sha> origin/main` to confirm reachability.
    Phase 0.1a stays offline-friendly.
    """
    return bool(_SHA_PATTERN.fullmatch(value.strip()))


def _is_stale(last_verified_utc: str) -> bool:
    try:
        ts = datetime.fromisoformat(last_verified_utc.replace("Z", "+00:00"))
    except ValueError:
        return True
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - ts) > _STALE_THRESHOLD


def audit_claim(repo_root: Path, claim: dict[str, Any]) -> list[Violation]:
    """Apply every contract to a single claim. Returns its violations."""
    cid = claim.get("id", "<missing-id>")
    out: list[Violation] = []

    # Required-field presence (errors regardless of status).
    for field_name in _REQUIRED_CLAIM_FIELDS:
        if field_name not in claim:
            out.append(Violation(cid, "error", field_name, f"Missing required field `{field_name}`"))

    status = claim.get("status", "")
    if status not in _VALID_STATUSES:
        out.append(
            Violation(
                cid,
                "error",
                "status",
                f"Invalid status `{status}`. Must be one of {sorted(_VALID_STATUSES)}",
            )
        )

    # Evidence contract (only enforced for delivered / default-on).
    #
    # Phase 0.4 (cgpro 2026-05-06): the contract is a LOOSE-OR. A delivered
    # claim must have AT LEAST ONE of {evidence_test, evidence_benchmark}
    # pointing at a real file, AND evidence_commit must be a well-formed SHA.
    # This accommodates benchmark claims (the .json/.md artefact IS the
    # evidence, no separate test exists) without forcing them to flip to
    # `evidence_pending`. If neither evidence_test nor evidence_benchmark is
    # populated, a single `evidence_anchor` error is raised — clearer than
    # two separate "missing X" errors.
    if status in _EVIDENCE_REQUIRED_STATUSES:
        ev_test = claim.get("evidence_test", "")
        ev_bench = claim.get("evidence_benchmark", "")
        ev_commit = claim.get("evidence_commit", "")

        test_pinned = (not _is_placeholder(ev_test)) and ev_test != "n/a"
        bench_pinned = (not _is_placeholder(ev_bench)) and ev_bench != "n/a"

        if not (test_pinned or bench_pinned):
            out.append(
                Violation(
                    cid,
                    "error",
                    "evidence_anchor",
                    f"status={status} requires at least one of evidence_test "
                    f"or evidence_benchmark to be a real file path (both "
                    f"placeholders: ev_test={ev_test!r}, ev_bench={ev_bench!r})",
                )
            )

        if test_pinned and not _evidence_test_resolves(repo_root, ev_test):
            out.append(
                Violation(
                    cid,
                    "error",
                    "evidence_test",
                    f"evidence_test path `{ev_test}` does not exist on disk",
                )
            )

        if bench_pinned and not _evidence_benchmark_resolves(repo_root, ev_bench):
            out.append(
                Violation(
                    cid,
                    "error",
                    "evidence_benchmark",
                    f"evidence_benchmark path `{ev_bench}` does not exist on disk",
                )
            )

        if _is_placeholder(ev_commit):
            out.append(
                Violation(
                    cid,
                    "error",
                    "evidence_commit",
                    f"status={status} requires a non-placeholder evidence_commit (got `{ev_commit!r}`)",
                )
            )
        elif not _evidence_commit_well_formed(ev_commit):
            out.append(
                Violation(
                    cid,
                    "error",
                    "evidence_commit",
                    f"evidence_commit `{ev_commit}` is not a 7-40 char hex SHA",
                )
            )

    # Stale verification timestamp (warning only).
    last_verified = claim.get("last_verified_utc", "")
    if isinstance(last_verified, str) and last_verified and _is_stale(last_verified):
        out.append(
            Violation(
                cid,
                "warning",
                "last_verified_utc",
                f"last_verified_utc `{last_verified}` is older than 90 days",
            )
        )

    return out


def run_audit(repo_root: Path, strict: bool = False) -> AuditReport:
    claims = load_claims(repo_root)
    report = AuditReport(total_claims=len(claims), mode="strict" if strict else "audit")

    for claim in claims:
        status = claim.get("status", "<missing>")
        report.by_status[status] = report.by_status.get(status, 0) + 1
        report.violations.extend(audit_claim(repo_root, claim))

    if strict:
        report.ok = len(report.errors) == 0
    else:
        report.ok = True  # AUDIT mode never fails

    return report


def _find_repo_root(start: Path) -> Path:
    """Walk up until we find `docs/claims/` (the registry directory)."""
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "docs" / "claims").is_dir():
            return candidate
    return start.resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit YGN-SAGE claims registry.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 on contract violations. Default = AUDIT mode (warns only).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON report to stdout instead of human summary.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Override repo-root detection (defaults to walking up from cwd).",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root if args.repo_root else _find_repo_root(Path.cwd())
    report = run_audit(repo_root, strict=args.strict)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"Claims audit ({report.mode} mode) — repo {repo_root}")
        print(f"  Total claims: {report.total_claims}")
        print("  By status:")
        for status, count in sorted(report.by_status.items()):
            print(f"    {status}: {count}")
        if report.errors:
            print(f"  Errors ({len(report.errors)}):")
            for v in report.errors:
                print(f"    [{v.claim_id}] {v.field_name}: {v.message}")
        if report.warnings:
            print(f"  Warnings ({len(report.warnings)}):")
            for v in report.warnings:
                print(f"    [{v.claim_id}] {v.field_name}: {v.message}")
        if report.ok:
            print("  Result: OK")
        else:
            print("  Result: FAIL (strict mode)")

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
