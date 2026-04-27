#!/usr/bin/env python3
"""Reset YGN-SAGE bandit SQLite state for the A14 causality migration.

Background
----------
Pre-A14 (commit ``48dc7c3f``), the Python pipeline silently corrupted the
ContextualBandit's learned posteriors:

1. Stage 0 routed via the legacy ``SystemRouter::route()`` API — it never
   consulted the bandit. Stage 4 separately called
   ``bandit.select_with_context()`` and stored only ``decision_id`` on the
   pipeline context. The chosen ``model_id`` and ``template`` were dropped.
   Stage 5 recorded outcomes against this orphan id, so posteriors updated
   for arms whose ``(model_id, template)`` never executed.

2. The PyO3 wrapper exposed the recording method as ``record`` while
   ``pipeline.py`` checked ``hasattr(self.bandit, "record_outcome")`` —
   the check always failed, so even off-policy attribution wasn't
   actually recorded. The bandit was effectively learning nothing from
   production traffic.

A14 (commit ``48dc7c3f``) fixes both bugs by introducing
``record_outcome_checked`` that refuses off-policy attribution, plus a
single-agent-only causal selection path. Going forward, posteriors update
only when the executed ``(model_id, template)`` matches the selected one.

But the historical SQLite state cannot be salvaged. The legacy schema
never stored both selected and executed (model_id, template) pairs, so
no audit can prove ≥95% causal attribution per cgpro's threshold. Under
the cgpro 2026-04-26 review verdict, **unverifiable state must be reset
by default**.

This script
-----------
Provides a safe, reviewable reset:

* ``--dry-run``: report tables and row counts; do not modify anything.
* ``--apply``: take a timestamped full-file backup, then EITHER:

  - default: TRUNCATE ``bandit_arms`` only (preserves ``bandit_config``
    so any explicit decay/exploration overrides survive); or
  - ``--full-reset``: rename the DB aside so the next bandit warm-start
    rebuilds priors from ``cards.toml`` affinities from scratch.

Refuses to run if neither ``--dry-run`` nor ``--apply`` is supplied.

Usage
-----
::

    # Inspect first
    python scripts/a14_reset_bandit_state.py \\
        --db /path/to/bandit_state.sqlite \\
        --dry-run

    # Targeted truncate (recommended)
    python scripts/a14_reset_bandit_state.py \\
        --db /path/to/bandit_state.sqlite \\
        --apply

    # Full reset (whole DB moved aside)
    python scripts/a14_reset_bandit_state.py \\
        --db /path/to/bandit_state.sqlite \\
        --apply --full-reset

After ``--apply`` the next bandit boot warm-starts from ``cards.toml``
affinities (see ``sage-python/src/sage/boot_topology.py``).

Operational note
----------------
Run this during a maintenance window. The reset preserves a backup so
the operation is reversible; restoration is just a file copy back into
place. Do NOT run autonomously against a live deployment without
operator approval.

See ``docs/migrations/2026-04-27-a14-reset.md`` for the full migration
runbook.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import shutil
import sqlite3
import sys
from pathlib import Path

_BANDIT_TABLES = ("bandit_arms", "bandit_config")


def _list_tables(conn: sqlite3.Connection) -> list[str]:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    )
    return [row[0] for row in cur.fetchall()]


def _row_count(conn: sqlite3.Connection, table: str) -> int:
    cur = conn.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608 — table from allow-list
    return int(cur.fetchone()[0])


def _report(db: Path) -> None:
    print(f"DB: {db}")
    print(f"  exists: {db.exists()}")
    if not db.exists():
        return
    print(f"  size:   {db.stat().st_size:,} bytes")
    with sqlite3.connect(db) as conn:
        tables = _list_tables(conn)
        print(f"  tables: {tables}")
        for t in _BANDIT_TABLES:
            if t in tables:
                print(f"    {t}: {_row_count(conn, t):,} rows")


def _backup(db: Path) -> Path:
    """Copy ``db`` to ``<stem>_pre_a14_reset_<timestamp>.sqlite.bak``.

    Returns the backup path. Raises if backup already exists at that path
    (unlikely given the second-precision timestamp).
    """
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = db.with_name(f"{db.stem}_pre_a14_reset_{stamp}{db.suffix}.bak")
    if backup.exists():
        raise FileExistsError(f"refusing to overwrite existing backup: {backup}")
    shutil.copy2(db, backup)
    return backup


def _truncate(db: Path) -> None:
    """Drop all rows from ``bandit_arms`` (preserves config)."""
    with sqlite3.connect(db) as conn:
        if "bandit_arms" not in _list_tables(conn):
            print("  bandit_arms not present; nothing to truncate")
            return
        conn.execute("DELETE FROM bandit_arms")
        conn.commit()
        # VACUUM frees the disk space the deleted rows held.
        conn.execute("VACUUM")
    print("  bandit_arms truncated; bandit_config preserved")


def _full_reset(db: Path) -> Path:
    """Rename the DB aside; next boot recreates from cards.toml."""
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    aside = db.with_name(f"{db.stem}_pre_a14_full_reset_{stamp}{db.suffix}")
    if aside.exists():
        raise FileExistsError(f"refusing to overwrite existing aside: {aside}")
    db.rename(aside)
    print(f"  moved aside: {aside}")
    return aside


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="a14_reset_bandit_state",
        description="Reset YGN-SAGE bandit SQLite state for the A14 causality migration.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        required=True,
        help="path to the bandit SQLite DB (e.g. ~/.sage/bandit_state.sqlite)",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="print tables + row counts; do not modify anything",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="back up and reset the bandit state",
    )
    parser.add_argument(
        "--full-reset",
        action="store_true",
        help=(
            "with --apply: move the entire DB aside instead of truncating "
            "bandit_arms (drops bandit_config too)"
        ),
    )
    args = parser.parse_args(argv)

    db: Path = args.db.expanduser().resolve()
    print(f"=== A14 reset (mode: {'apply' if args.apply else 'dry-run'}) ===")
    _report(db)

    if args.dry_run:
        print()
        print("Dry run only. No backup taken; no rows touched.")
        return 0

    if not db.exists():
        print()
        print(f"DB does not exist; nothing to reset: {db}")
        return 0

    print()
    backup = _backup(db)
    print(f"  backup: {backup}")

    if args.full_reset:
        _full_reset(db)
    else:
        _truncate(db)

    print()
    print("=== post-reset state ===")
    _report(db)
    print()
    print("Next bandit boot will warm-start from cards.toml affinities.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
