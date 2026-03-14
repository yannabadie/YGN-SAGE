# RESERVED FOR PHASE C: Checkpoint — SQLite-based agent loop resumption.
# To be integrated with TopologyExecutor for long-running multi-agent topologies.
"""Durable execution: checkpoint agent loop state at phase boundaries."""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_DB = Path.home() / ".sage" / "checkpoints.db"


class Checkpoint:
    """SQLite-backed checkpoint for agent loop resumption."""

    def __init__(self, db_path: Path | None = None, session_id: str | None = None):
        self._db_path = db_path or _DEFAULT_DB
        self._session_id = session_id or "default"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS checkpoints ("
            "  session_id TEXT NOT NULL,"
            "  phase TEXT NOT NULL,"
            "  step_count INTEGER NOT NULL,"
            "  state_json TEXT NOT NULL,"
            "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
            "  PRIMARY KEY (session_id, phase)"
            ")"
        )
        self._conn.commit()

    def save(self, phase: str, step_count: int, state: dict[str, Any]) -> None:
        """Save checkpoint at a phase boundary."""
        self._conn.execute(
            "INSERT OR REPLACE INTO checkpoints (session_id, phase, step_count, state_json) "
            "VALUES (?, ?, ?, ?)",
            (self._session_id, phase, step_count, json.dumps(state, default=str)),
        )
        self._conn.commit()
        log.debug("Checkpoint saved: session=%s phase=%s step=%d", self._session_id, phase, step_count)

    def load(self, phase: str) -> tuple[int, dict[str, Any]] | None:
        """Load checkpoint for a phase. Returns (step_count, state) or None."""
        row = self._conn.execute(
            "SELECT step_count, state_json FROM checkpoints WHERE session_id = ? AND phase = ?",
            (self._session_id, phase),
        ).fetchone()
        if row is None:
            return None
        return row[0], json.loads(row[1])

    def clear(self, phase: str | None = None) -> None:
        """Clear checkpoint(s). If phase is None, clear all for this session."""
        if phase:
            self._conn.execute(
                "DELETE FROM checkpoints WHERE session_id = ? AND phase = ?",
                (self._session_id, phase),
            )
        else:
            self._conn.execute(
                "DELETE FROM checkpoints WHERE session_id = ?",
                (self._session_id,),
            )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
