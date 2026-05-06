"""Tests for `sage.ops.claims_audit` and `scripts/regenerate_claims_index.py`.

Cycle-13 K Phase 0.1a: the registry is non-blocking on first ship — these
tests pin the contract between Phase 0.1a (AUDIT mode warns, exit 0) and
Phase 0.1b (STRICT mode fails on missing evidence for delivered/default-on).
The schema and the placeholder vocabulary stay constant; only the exit-code
contract differs between modes.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml

from sage.ops import claims_audit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_claims_yaml(claims_dir: Path, name: str, claims: list[dict[str, Any]]) -> Path:
    target = claims_dir / name
    target.write_text(yaml.safe_dump({"claims": claims}, sort_keys=False), encoding="utf-8")
    return target


def _seed_evidence_test(repo_root: Path, rel_path: str) -> None:
    target = repo_root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# placeholder evidence_test\n", encoding="utf-8")


def _seed_evidence_benchmark(repo_root: Path, rel_path: str) -> None:
    target = repo_root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{\"placeholder\": true}\n", encoding="utf-8")


def _make_repo_root(tmp_path: Path) -> Path:
    """Lay out a minimal `docs/claims/` shell under tmp_path."""
    claims = tmp_path / "docs" / "claims"
    claims.mkdir(parents=True)
    return tmp_path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stale_iso() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=120)).isoformat(timespec="seconds")


def _full_claim(**overrides: Any) -> dict[str, Any]:
    base = {
        "id": "test.example",
        "status": "delivered",
        "description": "Example claim.",
        "evidence_test": "sage-python/tests/test_example.py::test_thing",
        "evidence_benchmark": "n/a",
        "evidence_commit": "1234567",
        "last_verified_utc": _now_iso(),
        "owner": "test",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# load_claims
# ---------------------------------------------------------------------------


def test_load_claims_returns_empty_when_directory_missing(tmp_path: Path) -> None:
    assert claims_audit.load_claims(tmp_path) == []


def test_load_claims_concatenates_per_category_files(tmp_path: Path) -> None:
    repo = _make_repo_root(tmp_path)
    _write_claims_yaml(repo / "docs" / "claims", "a.yaml", [_full_claim(id="a.one")])
    _write_claims_yaml(repo / "docs" / "claims", "b.yaml", [_full_claim(id="b.one")])
    out = claims_audit.load_claims(repo)
    assert sorted(c["id"] for c in out) == ["a.one", "b.one"]


def test_load_claims_skips_generated_index(tmp_path: Path) -> None:
    repo = _make_repo_root(tmp_path)
    _write_claims_yaml(repo / "docs" / "claims", "routing.yaml", [_full_claim(id="r.one")])
    # If we ever accidentally drop CLAIMS.yaml into docs/claims/ it must NOT
    # double-count. (The real index lives in docs/CLAIMS.yaml — outside.)
    _write_claims_yaml(repo / "docs" / "claims", "CLAIMS.yaml", [_full_claim(id="r.one")])
    out = claims_audit.load_claims(repo)
    assert [c["id"] for c in out] == ["r.one"]


def test_load_claims_skips_malformed_yaml(tmp_path: Path) -> None:
    repo = _make_repo_root(tmp_path)
    (repo / "docs" / "claims" / "broken.yaml").write_text(": this is not yaml :", encoding="utf-8")
    _write_claims_yaml(repo / "docs" / "claims", "ok.yaml", [_full_claim(id="x.one")])
    out = claims_audit.load_claims(repo)
    assert [c["id"] for c in out] == ["x.one"]


# ---------------------------------------------------------------------------
# audit_claim — required fields and status vocabulary
# ---------------------------------------------------------------------------


def test_audit_claim_passes_complete_delivered_claim(tmp_path: Path) -> None:
    repo = _make_repo_root(tmp_path)
    _seed_evidence_test(repo, "sage-python/tests/test_example.py")
    violations = claims_audit.audit_claim(repo, _full_claim())
    assert violations == []


def test_audit_claim_flags_missing_required_field(tmp_path: Path) -> None:
    repo = _make_repo_root(tmp_path)
    claim = _full_claim()
    del claim["owner"]
    violations = claims_audit.audit_claim(repo, claim)
    assert any(v.field_name == "owner" and v.severity == "error" for v in violations)


def test_audit_claim_flags_invalid_status(tmp_path: Path) -> None:
    repo = _make_repo_root(tmp_path)
    violations = claims_audit.audit_claim(repo, _full_claim(status="bogus"))
    assert any(v.field_name == "status" and v.severity == "error" for v in violations)


# ---------------------------------------------------------------------------
# audit_claim — evidence contract
# ---------------------------------------------------------------------------


def test_delivered_with_both_anchors_placeholder_is_error(tmp_path: Path) -> None:
    """Phase 0.4 loose-OR contract: a delivered claim with placeholder
    evidence_test AND placeholder evidence_benchmark fails with a single
    `evidence_anchor` error (not two separate field errors)."""
    repo = _make_repo_root(tmp_path)
    violations = claims_audit.audit_claim(
        repo,
        _full_claim(evidence_test="evidence_pending", evidence_benchmark="n/a"),
    )
    anchor_errors = [v for v in violations if v.field_name == "evidence_anchor"]
    assert len(anchor_errors) == 1
    assert anchor_errors[0].severity == "error"


def test_delivered_with_only_benchmark_anchor_passes(tmp_path: Path) -> None:
    """Phase 0.4 loose-OR: pinning evidence_benchmark alone is sufficient.
    Required for benchmark claims where the JSON/MD artefact IS the evidence."""
    repo = _make_repo_root(tmp_path)
    _seed_evidence_benchmark(repo, "docs/benchmarks/anchor.json")
    violations = claims_audit.audit_claim(
        repo,
        _full_claim(
            evidence_test="evidence_pending",
            evidence_benchmark="docs/benchmarks/anchor.json",
        ),
    )
    assert all(v.severity != "error" for v in violations)


def test_planned_with_placeholder_evidence_test_is_ok(tmp_path: Path) -> None:
    repo = _make_repo_root(tmp_path)
    violations = claims_audit.audit_claim(
        repo,
        _full_claim(status="planned", evidence_test="evidence_pending", evidence_commit="evidence_pending"),
    )
    assert all(v.severity != "error" for v in violations)


def test_delivered_with_orphan_evidence_test_path_is_error(tmp_path: Path) -> None:
    repo = _make_repo_root(tmp_path)
    # No file seeded — evidence_test points at a path that doesn't exist.
    violations = claims_audit.audit_claim(
        repo,
        _full_claim(evidence_test="sage-python/tests/test_does_not_exist.py::test_x"),
    )
    assert any(v.field_name == "evidence_test" and v.severity == "error" for v in violations)


def test_delivered_with_malformed_commit_sha_is_error(tmp_path: Path) -> None:
    repo = _make_repo_root(tmp_path)
    _seed_evidence_test(repo, "sage-python/tests/test_example.py")
    violations = claims_audit.audit_claim(repo, _full_claim(evidence_commit="not-a-sha"))
    assert any(v.field_name == "evidence_commit" and v.severity == "error" for v in violations)


def test_evidence_benchmark_prefix_ok_accepts_canonical_paths() -> None:
    """Phase 0.6 (cgpro post-push): allowed prefixes are docs/benchmarks/
    and docs/audits/. Any other location fails the contract."""
    assert claims_audit._evidence_benchmark_prefix_ok("docs/benchmarks/2026-04-08-foo.json")
    assert claims_audit._evidence_benchmark_prefix_ok("docs/audits/2026-04-22-bar.md")


def test_evidence_benchmark_prefix_ok_rejects_other_locations() -> None:
    assert not claims_audit._evidence_benchmark_prefix_ok("docs/random/foo.json")
    assert not claims_audit._evidence_benchmark_prefix_ok("README.md")
    assert not claims_audit._evidence_benchmark_prefix_ok("sage-python/tests/test_x.py")
    assert not claims_audit._evidence_benchmark_prefix_ok("/etc/passwd")


def test_evidence_benchmark_prefix_ok_normalizes_backslashes() -> None:
    """Windows path separators must not slip past the gate."""
    assert claims_audit._evidence_benchmark_prefix_ok("docs\\benchmarks\\foo.json")
    assert claims_audit._evidence_benchmark_prefix_ok("docs\\audits\\bar.md")


def test_audit_claim_rejects_benchmark_outside_allowed_prefix(tmp_path: Path) -> None:
    """A delivered claim whose evidence_benchmark exists on disk but lives
    outside docs/benchmarks/ or docs/audits/ MUST fail strict mode. The
    docstring promised this; pre-Phase-0.6 the implementation accepted
    any existing file in the repo."""
    repo = _make_repo_root(tmp_path)
    # Seed a file at a non-canonical location.
    bogus_dir = repo / "scripts" / "stuff"
    bogus_dir.mkdir(parents=True)
    bogus_file = bogus_dir / "leak.json"
    bogus_file.write_text("{}", encoding="utf-8")

    violations = claims_audit.audit_claim(
        repo,
        _full_claim(
            evidence_test="evidence_pending",
            evidence_benchmark="scripts/stuff/leak.json",
        ),
    )
    prefix_violations = [
        v for v in violations
        if v.field_name == "evidence_benchmark"
        and v.severity == "error"
        and "outside the allowed prefixes" in v.message
    ]
    assert prefix_violations, (
        f"Audit accepted a benchmark anchor outside docs/benchmarks/ or "
        f"docs/audits/; got violations={violations}"
    )


def test_evidence_benchmark_missing_file_is_error_when_pinned(tmp_path: Path) -> None:
    """Phase 0.4: if a claim explicitly pins evidence_benchmark to a path,
    that path must exist (otherwise the registry is lying about the anchor).
    Pre-Phase-0.4 this was a warning; loose-OR contract upgraded it to error
    because evidence_benchmark is now a load-bearing anchor for benchmark
    claims, not a SHOULD."""
    repo = _make_repo_root(tmp_path)
    _seed_evidence_test(repo, "sage-python/tests/test_example.py")
    violations = claims_audit.audit_claim(
        repo,
        _full_claim(evidence_benchmark="docs/benchmarks/missing.json"),
    )
    bench_errors = [
        v for v in violations
        if v.field_name == "evidence_benchmark" and v.severity == "error"
    ]
    assert bench_errors


def test_stale_last_verified_is_warning(tmp_path: Path) -> None:
    repo = _make_repo_root(tmp_path)
    _seed_evidence_test(repo, "sage-python/tests/test_example.py")
    violations = claims_audit.audit_claim(repo, _full_claim(last_verified_utc=_stale_iso()))
    stale = [v for v in violations if v.field_name == "last_verified_utc"]
    assert stale and all(v.severity == "warning" for v in stale)


# ---------------------------------------------------------------------------
# run_audit — mode contract
# ---------------------------------------------------------------------------


def test_run_audit_audit_mode_never_fails_on_violations(tmp_path: Path) -> None:
    repo = _make_repo_root(tmp_path)
    _write_claims_yaml(
        repo / "docs" / "claims",
        "a.yaml",
        [_full_claim(evidence_test="evidence_pending")],
    )
    report = claims_audit.run_audit(repo, strict=False)
    assert report.errors  # there IS a violation
    assert report.ok is True  # but AUDIT mode says OK


def test_run_audit_strict_mode_fails_on_violations(tmp_path: Path) -> None:
    repo = _make_repo_root(tmp_path)
    _write_claims_yaml(
        repo / "docs" / "claims",
        "a.yaml",
        [_full_claim(evidence_test="evidence_pending")],
    )
    report = claims_audit.run_audit(repo, strict=True)
    assert report.errors
    assert report.ok is False


def test_run_audit_strict_mode_passes_when_clean(tmp_path: Path) -> None:
    repo = _make_repo_root(tmp_path)
    _seed_evidence_test(repo, "sage-python/tests/test_example.py")
    _write_claims_yaml(repo / "docs" / "claims", "a.yaml", [_full_claim()])
    report = claims_audit.run_audit(repo, strict=True)
    assert report.errors == []
    assert report.ok is True


def test_run_audit_counts_by_status(tmp_path: Path) -> None:
    repo = _make_repo_root(tmp_path)
    _seed_evidence_test(repo, "sage-python/tests/test_example.py")
    _write_claims_yaml(
        repo / "docs" / "claims",
        "a.yaml",
        [
            _full_claim(id="a.one", status="delivered"),
            _full_claim(id="a.two", status="planned", evidence_test="evidence_pending", evidence_commit="evidence_pending"),
            _full_claim(id="a.three", status="planned", evidence_test="evidence_pending", evidence_commit="evidence_pending"),
        ],
    )
    report = claims_audit.run_audit(repo, strict=True)
    assert report.by_status == {"delivered": 1, "planned": 2}


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------


def test_cli_audit_mode_returns_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _make_repo_root(tmp_path)
    _write_claims_yaml(
        repo / "docs" / "claims",
        "a.yaml",
        [_full_claim(evidence_test="evidence_pending")],
    )
    rc = claims_audit.main(["--repo-root", str(repo)])
    assert rc == 0


def test_cli_strict_mode_returns_one_on_violations(tmp_path: Path) -> None:
    repo = _make_repo_root(tmp_path)
    _write_claims_yaml(
        repo / "docs" / "claims",
        "a.yaml",
        [_full_claim(evidence_test="evidence_pending")],
    )
    rc = claims_audit.main(["--strict", "--repo-root", str(repo)])
    assert rc == 1


def test_cli_json_output_is_valid_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _make_repo_root(tmp_path)
    _seed_evidence_test(repo, "sage-python/tests/test_example.py")
    _write_claims_yaml(repo / "docs" / "claims", "a.yaml", [_full_claim()])
    claims_audit.main(["--json", "--repo-root", str(repo)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["mode"] == "audit"
    assert payload["total_claims"] == 1


# ---------------------------------------------------------------------------
# Integration: real repo claims registry passes AUDIT mode
# ---------------------------------------------------------------------------


def test_real_repo_claims_pass_audit_mode() -> None:
    """The committed Phase 0.1a registry MUST be loadable + audit-clean.

    AUDIT mode never fails, but this test also asserts that no claim has
    a structural error (invalid status, missing required field). That's
    the contract Phase 0.1a ships.
    """
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / "docs" / "claims").is_dir():
        pytest.skip("docs/claims/ not present in this checkout")

    report = claims_audit.run_audit(repo_root, strict=False)
    assert report.total_claims > 0, "no claims loaded — registry empty?"

    # Phase 0.1a: structural errors (missing fields, invalid status) are not
    # acceptable even in AUDIT mode. Evidence-related errors ARE acceptable
    # because Phase 0.4 closes them.
    structural_fields = {"id", "status", "description", "owner"}
    structural_errors = [
        v for v in report.errors
        if v.field_name in structural_fields
        or (v.field_name == "status" and v.message.startswith("Invalid status"))
    ]
    assert structural_errors == [], (
        f"Registry has structural errors that even AUDIT mode rejects: {structural_errors}"
    )


# ---------------------------------------------------------------------------
# scripts/regenerate_claims_index.py
# ---------------------------------------------------------------------------


def _load_regenerator() -> Any:
    """Import scripts/regenerate_claims_index.py as a module by path."""
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "regenerate_claims_index.py"
    if not script_path.is_file():
        pytest.skip(f"{script_path} not present")
    spec = importlib.util.spec_from_file_location("regenerate_claims_index", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["regenerate_claims_index"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_regenerator_writes_aggregate(tmp_path: Path) -> None:
    repo = _make_repo_root(tmp_path)
    _write_claims_yaml(repo / "docs" / "claims", "a.yaml", [_full_claim(id="z.one")])
    _write_claims_yaml(repo / "docs" / "claims", "b.yaml", [_full_claim(id="a.one")])
    regen = _load_regenerator()
    rc = regen.main(["--repo-root", str(repo)])
    assert rc == 0
    out = (repo / "docs" / "CLAIMS.yaml").read_text(encoding="utf-8")
    payload = yaml.safe_load(out)
    # Sorted by id for stable diff.
    assert [c["id"] for c in payload["claims"]] == ["a.one", "z.one"]
    assert payload["_total_claims"] == 2


def test_regenerator_check_mode_passes_when_up_to_date(tmp_path: Path) -> None:
    repo = _make_repo_root(tmp_path)
    _write_claims_yaml(repo / "docs" / "claims", "a.yaml", [_full_claim(id="a.one")])
    regen = _load_regenerator()
    regen.main(["--repo-root", str(repo)])
    rc = regen.main(["--check", "--repo-root", str(repo)])
    assert rc == 0


def test_regenerator_check_mode_fails_when_drifted(tmp_path: Path) -> None:
    repo = _make_repo_root(tmp_path)
    _write_claims_yaml(repo / "docs" / "claims", "a.yaml", [_full_claim(id="a.one")])
    regen = _load_regenerator()
    regen.main(["--repo-root", str(repo)])
    # Add a new claim AFTER initial generation — the index is now stale.
    _write_claims_yaml(repo / "docs" / "claims", "b.yaml", [_full_claim(id="b.one")])
    rc = regen.main(["--check", "--repo-root", str(repo)])
    assert rc == 1


def test_regenerator_idempotent_on_same_inputs(tmp_path: Path) -> None:
    repo = _make_repo_root(tmp_path)
    _write_claims_yaml(repo / "docs" / "claims", "a.yaml", [_full_claim(id="a.one")])
    regen = _load_regenerator()
    regen.main(["--repo-root", str(repo)])
    first = (repo / "docs" / "CLAIMS.yaml").read_text(encoding="utf-8")
    regen.main(["--repo-root", str(repo)])
    second = (repo / "docs" / "CLAIMS.yaml").read_text(encoding="utf-8")
    # Strip the volatile timestamp line for the comparison.
    def _strip(s: str) -> str:
        return "\n".join(line for line in s.splitlines() if not line.strip().startswith("_generated_at_utc:"))
    assert _strip(first) == _strip(second)
