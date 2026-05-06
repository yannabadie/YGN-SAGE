"""Sync test counters across YGN-SAGE canonical docs.

Cycle-13 K Phase 0.2 (cgpro VERIFY conv 'Analyse approfondie de repo'
2026-05-06): four canonical docs cite test counters and disagree
(`docs/status/current.json` 3089/553/100, README badge 2953/553,
AI-ARCHITECTURE.md 2940/549/100, .claude/rules/architecture.md 2940/549).
ALIRE.md flagged this as evidence the repo is not yet truthful.

This script makes `docs/status/current.json` the single source of truth
and propagates its `sage_python_tests.total`, `sage_core_tests.total`,
and `sage_discover_tests.total` values into:

  - README.md badge:           ![tests-3114 Py + 553 Rust]
  - AI-ARCHITECTURE.md header: | **Tests Python collected** | **3114** ...
  - .claude/rules/architecture.md: **3114 collected** / **553 tests** / **100 tests**

Usage:
  python scripts/sync_doc_counters.py            # write
  python scripts/sync_doc_counters.py --check    # exit 1 if drifted
  python scripts/sync_doc_counters.py --json     # report as JSON

A separate workflow `.github/workflows/doc-counters-coherence.yml`
runs `--check` on every push to fail CI if any doc drifts away from
`current.json`.

This script does NOT compute test counts itself — that's
`pytest --collect-only` + `cargo test --list`. It only propagates.
The CI gate also runs `pytest --collect-only` and asserts the
result equals `current.json#sage_python_tests.total` so a stale
`current.json` is caught at the same moment as a stale README.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Counters:
    """The authoritative test counts loaded from `current.json`."""

    python: int
    rust: int
    discover: int
    invariants: int = 0  # populated from the runtime-integrity-ledger heading

    @classmethod
    def from_current_json(cls, payload: dict[str, Any]) -> "Counters":
        return cls(
            python=int(payload["sage_python_tests"]["total"]),
            rust=int(payload["sage_core_tests"]["total"]),
            discover=int(payload["sage_discover_tests"]["total"]),
        )


_LEDGER_HEADING_RE = re.compile(r"^##\s+The\s+(\d+)\s+invariants\s*$", re.MULTILINE)
_LEDGER_TABLE_ROW_RE = re.compile(r"^\|\s*\*\*[^|]+\*\*\s*\|", re.MULTILINE)


def load_invariant_count(repo_root: Path) -> int:
    """Read `## The N invariants` heading from the runtime-integrity-ledger.

    Source-of-truth for the invariant count: cgpro 2026-05-06 confirmed
    `runtime-integrity-ledger.md` is the authoritative number. This
    function ALSO counts the table rows under that heading and asserts
    the heading agrees with them — a mismatch is itself a drift signal
    and raises ValueError.

    Returns 0 if the ledger file doesn't exist (test fixtures may omit it);
    invariant propagation then becomes a no-op for that run.
    """
    ledger_path = repo_root / "docs" / "contracts" / "runtime-integrity-ledger.md"
    if not ledger_path.is_file():
        return 0
    text = ledger_path.read_text(encoding="utf-8")

    heading_match = _LEDGER_HEADING_RE.search(text)
    if not heading_match:
        raise ValueError(
            f"Cannot find `## The N invariants` heading in {ledger_path}"
        )
    heading_count = int(heading_match.group(1))

    # Count table rows from the heading to the next top-level "##" heading.
    after_heading = text[heading_match.end():]
    next_heading = re.search(r"^##\s", after_heading, re.MULTILINE)
    section = after_heading[: next_heading.start()] if next_heading else after_heading
    table_row_count = len(_LEDGER_TABLE_ROW_RE.findall(section))

    if heading_count != table_row_count:
        raise ValueError(
            f"Ledger heading claims {heading_count} invariants but section "
            f"has {table_row_count} table rows. Fix the ledger before syncing."
        )
    return heading_count


@dataclass
class Drift:
    """A single counter mismatch found during --check."""

    file: str
    counter: str
    expected: int
    found_in_text: str  # the offending substring


def load_current_json(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "docs" / "status" / "current.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Per-doc rewriters. Each returns (new_text, drifts_found_under_old_text).
# Even when --check is true, we compute the "would-be" new text to confirm
# the propagation rule is unambiguous (no regex partial match).
# ---------------------------------------------------------------------------


_README_BADGE_RE = re.compile(
    r"(badge/tests-)(\d+)(%20Py%20%2B%20)(\d+)(%20Rust)",
)


def update_readme(text: str, c: Counters) -> tuple[str, list[Drift]]:
    drifts: list[Drift] = []

    def _sub(m: re.Match[str]) -> str:
        existing_py = int(m.group(2))
        existing_rust = int(m.group(4))
        if existing_py != c.python:
            drifts.append(Drift("README.md", "python", c.python, m.group(0)))
        if existing_rust != c.rust:
            drifts.append(Drift("README.md", "rust", c.rust, m.group(0)))
        return f"{m.group(1)}{c.python}{m.group(3)}{c.rust}{m.group(5)}"

    new_text = _README_BADGE_RE.sub(_sub, text)
    return new_text, drifts


# AI-ARCHITECTURE.md header table:
#   | **Tests Python collected** | **2940** (canonical : ...) |
#   | **Tests Rust** | **549** (`cargo test --features smt,...`) |
#   | **Tests sage-discover** | **100** |
_AIARCH_PY_RE = re.compile(r"(\*\*Tests Python collected\*\*\s*\|\s*\*\*)(\d+)(\*\*)")
_AIARCH_RUST_RE = re.compile(r"(\*\*Tests Rust\*\*\s*\|\s*\*\*)(\d+)(\*\*)")
_AIARCH_DISC_RE = re.compile(r"(\*\*Tests sage-discover\*\*\s*\|\s*\*\*)(\d+)(\*\*)")


def update_ai_architecture(text: str, c: Counters) -> tuple[str, list[Drift]]:
    drifts: list[Drift] = []

    def _make_sub(counter_name: str, expected: int) -> Any:
        def _sub(m: re.Match[str]) -> str:
            found = int(m.group(2))
            if found != expected:
                drifts.append(Drift("AI-ARCHITECTURE.md", counter_name, expected, m.group(0)))
            return f"{m.group(1)}{expected}{m.group(3)}"

        return _sub

    text = _AIARCH_PY_RE.sub(_make_sub("python", c.python), text)
    text = _AIARCH_RUST_RE.sub(_make_sub("rust", c.rust), text)
    text = _AIARCH_DISC_RE.sub(_make_sub("discover", c.discover), text)
    return text, drifts


# .claude/rules/architecture.md project-structure block:
#   - `sage-core/` — Rust orchestrator (PyO3). **549 tests** ...
#   - `sage-python/` — Python SDK. **2940 collected** ...
#   - `sage-discover/` — Knowledge pipeline (arXiv → ExoCortex). **100 tests** ...
_RULES_RUST_RE = re.compile(r"(`sage-core/`[^\n]*?\*\*)(\d+)( tests\*\*)")
_RULES_PY_RE = re.compile(r"(`sage-python/`[^\n]*?\*\*)(\d+)( collected\*\*)")
_RULES_DISC_RE = re.compile(r"(`sage-discover/`[^\n]*?\*\*)(\d+)( tests\*\*)")


def update_rules_architecture(text: str, c: Counters) -> tuple[str, list[Drift]]:
    drifts: list[Drift] = []

    specs = [
        (_RULES_RUST_RE, "rust", c.rust, ".claude/rules/architecture.md"),
        (_RULES_PY_RE, "python", c.python, ".claude/rules/architecture.md"),
        (_RULES_DISC_RE, "discover", c.discover, ".claude/rules/architecture.md"),
    ]

    for regex, name, expected, file in specs:
        def _make_sub(regex_name: str, regex_expected: int, regex_file: str) -> Any:
            def _sub(m: re.Match[str]) -> str:
                found = int(m.group(2))
                if found != regex_expected:
                    drifts.append(Drift(regex_file, regex_name, regex_expected, m.group(0)))
                return f"{m.group(1)}{regex_expected}{m.group(3)}"

            return _sub

        text = regex.sub(_make_sub(name, expected, file), text)

    return text, drifts


_INVARIANT_COUNT_RE = re.compile(r"(\b)(\d+)(\s+invariants\b)")


def update_invariant_count_in_readme(text: str, c: Counters) -> tuple[str, list[Drift]]:
    """README.md mentions invariant count in two places (line 41 + capability table).

    Both references describe the running ledger count and are safe to rewrite.
    """
    if c.invariants <= 0:
        return text, []
    drifts: list[Drift] = []

    def _sub(m: re.Match[str]) -> str:
        existing = int(m.group(2))
        if existing != c.invariants:
            drifts.append(Drift("README.md", "invariants", c.invariants, m.group(0)))
        return f"{m.group(1)}{c.invariants}{m.group(3)}"

    return _INVARIANT_COUNT_RE.sub(_sub, text), drifts


def update_invariant_count_in_ai_architecture(text: str, c: Counters) -> tuple[str, list[Drift]]:
    """AI-ARCHITECTURE.md has 6 mentions of `<N> invariants` — all reference the
    running ledger count, all safe to rewrite. CLAUDE.md is NOT covered here
    because it contains historical timeline references like "5 invariants at
    cycle-8 closure" that must NOT be rewritten.
    """
    if c.invariants <= 0:
        return text, []
    drifts: list[Drift] = []

    def _sub(m: re.Match[str]) -> str:
        existing = int(m.group(2))
        if existing != c.invariants:
            drifts.append(Drift("AI-ARCHITECTURE.md", "invariants", c.invariants, m.group(0)))
        return f"{m.group(1)}{c.invariants}{m.group(3)}"

    return _INVARIANT_COUNT_RE.sub(_sub, text), drifts


def _compose_readme(text: str, c: Counters) -> tuple[str, list[Drift]]:
    text, d1 = update_readme(text, c)
    text, d2 = update_invariant_count_in_readme(text, c)
    return text, d1 + d2


def _compose_ai_arch(text: str, c: Counters) -> tuple[str, list[Drift]]:
    text, d1 = update_ai_architecture(text, c)
    text, d2 = update_invariant_count_in_ai_architecture(text, c)
    return text, d1 + d2


_DOC_TARGETS: list[tuple[str, Any]] = [
    ("README.md", _compose_readme),
    ("AI-ARCHITECTURE.md", _compose_ai_arch),
    (".claude/rules/architecture.md", update_rules_architecture),
]


def sync(repo_root: Path, check_only: bool) -> tuple[bool, list[Drift]]:
    """Propagate `current.json` counters into all canonical docs.

    Returns:
        (ok, drifts) where ok is True when no drift is found (or all writes
        succeed in non-check mode), and drifts is the list of mismatches
        found during the propagation pass.
    """
    counters = Counters.from_current_json(load_current_json(repo_root))
    counters.invariants = load_invariant_count(repo_root)
    all_drifts: list[Drift] = []

    for rel_path, updater in _DOC_TARGETS:
        target = repo_root / rel_path
        if not target.is_file():
            # Missing canonical doc is itself a drift signal.
            all_drifts.append(Drift(rel_path, "<file-missing>", -1, ""))
            continue

        original = target.read_text(encoding="utf-8")
        new_text, drifts = updater(original, counters)
        all_drifts.extend(drifts)

        if check_only:
            continue
        if new_text != original:
            target.write_text(new_text, encoding="utf-8")

    if check_only:
        return (len(all_drifts) == 0), all_drifts
    return True, all_drifts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync test counters from current.json into canonical docs.")
    parser.add_argument("--check", action="store_true", help="Don't write; exit 1 on drift.")
    parser.add_argument("--json", action="store_true", help="Print drift report as JSON.")
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args(argv)

    repo_root = args.repo_root if args.repo_root else _find_repo_root(Path.cwd())
    ok, drifts = sync(repo_root, check_only=args.check)

    if args.json:
        print(
            json.dumps(
                {
                    "ok": ok,
                    "mode": "check" if args.check else "write",
                    "drifts": [d.__dict__ for d in drifts],
                },
                indent=2,
            )
        )
    else:
        print(f"sync_doc_counters ({'check' if args.check else 'write'} mode) — repo {repo_root}")
        if not drifts:
            print("  No drift detected.")
        else:
            print(f"  {len(drifts)} drift entries:")
            for d in drifts:
                print(f"    [{d.file}] {d.counter}: expected {d.expected}, found in `{d.found_in_text}`")
        if args.check:
            print("  Result:", "OK" if ok else "FAIL")
        else:
            print("  Result: written.")

    if args.check:
        return 0 if ok else 1
    return 0


def _find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "docs" / "status" / "current.json").is_file():
            return candidate
    return start.resolve()


if __name__ == "__main__":
    sys.exit(main())
