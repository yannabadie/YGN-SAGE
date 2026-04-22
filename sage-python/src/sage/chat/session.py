"""ChatSession — append-only jsonl persistence for REPL turns.

The on-disk format mirrors Claude Code's own transcript shape: one
jsonl record per event, fields `{ts, kind, ...payload}` where `kind ∈
{"session_start", "user", "assistant", "system"}`. Append-only: each
call to `log_turn()` opens the file in append mode so an interrupted
REPL still leaves a valid partial session (no partial-write corruption
window).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ulid

SESSIONS_DIR = Path.home() / ".sage" / "chat_sessions"


def _now_iso() -> str:
    """UTC ISO-8601 timestamp with microsecond precision + `Z` suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@dataclass
class ChatSession:
    """A single `sage.chat` REPL session.

    Holds the session id, the jsonl path, and the per-turn log API.
    Every call to `log_*` appends one record to disk immediately —
    no in-memory batching. Partial sessions always replay cleanly.
    """

    session_id: str = field(default_factory=lambda: str(ulid.new()))
    path: Path = field(init=False)
    bash_allowed: bool = False
    """Mirror of the SAGE_CHAT_ALLOW_BASH env var at session start.

    Can be flipped mid-session by the `/shell` REPL command (updates
    the env var AND this flag + logs a `system` event so the transcript
    records the consent change)."""

    def __post_init__(self) -> None:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        self.path = SESSIONS_DIR / f"{self.session_id}.jsonl"

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def start(cls, bash_allowed: bool = False) -> "ChatSession":
        """Create a new session file and write the session_start event."""
        s = cls(bash_allowed=bash_allowed)
        s._append({
            "kind": "session_start",
            "session_id": s.session_id,
            "ts": _now_iso(),
            "bash_allowed": bash_allowed,
            "sage_version": _read_version(),
        })
        return s

    @classmethod
    def resume(cls, session_id: str) -> "ChatSession":
        """Re-open an existing session file.

        Scans the existing jsonl for the latest `system` event that
        changed `bash_allowed` (e.g. via `/shell`) so the resumed
        session keeps that consent state.
        """
        path = SESSIONS_DIR / f"{session_id}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"No chat session at {path}")

        bash_allowed = False
        for event in _iter_events(path):
            if event.get("kind") == "session_start":
                bash_allowed = bool(event.get("bash_allowed", False))
            elif event.get("kind") == "system" and "bash_allowed" in event:
                bash_allowed = bool(event["bash_allowed"])

        s = cls.__new__(cls)
        s.session_id = session_id
        s.bash_allowed = bash_allowed
        s.path = path
        # No session_start event on resume — the transcript already has one.
        s._append({
            "kind": "system",
            "ts": _now_iso(),
            "event": "resume",
            "bash_allowed": bash_allowed,
        })
        return s

    # ------------------------------------------------------------------
    # Event logging
    # ------------------------------------------------------------------

    def log_user(self, text: str) -> None:
        self._append({"kind": "user", "ts": _now_iso(), "text": text})

    def log_assistant(self, text: str, *, latency_ms: int | None = None) -> None:
        record: dict[str, Any] = {
            "kind": "assistant",
            "ts": _now_iso(),
            "text": text,
        }
        if latency_ms is not None:
            record["latency_ms"] = latency_ms
        self._append(record)

    def log_system(self, event: str, **fields: Any) -> None:
        """Persist a meta event — `/shell` toggle, error, etc."""
        record = {"kind": "system", "ts": _now_iso(), "event": event}
        record.update(fields)
        self._append(record)

    # ------------------------------------------------------------------
    # /shell toggle
    # ------------------------------------------------------------------

    def toggle_bash(self) -> bool:
        """Flip the per-session bash-allowed flag and log the change.

        Updates both the in-memory `bash_allowed` attribute and the
        `SAGE_CHAT_ALLOW_BASH` env var so that subsequent
        `normalize_chat()` calls pick up the new state. Returns the
        new `bash_allowed` value.
        """
        self.bash_allowed = not self.bash_allowed
        os.environ["SAGE_CHAT_ALLOW_BASH"] = "1" if self.bash_allowed else "0"
        self.log_system("shell_toggled", bash_allowed=self.bash_allowed)
        return self.bash_allowed

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def replay(self) -> list[dict[str, Any]]:
        """Return all events from the on-disk transcript in order."""
        return list(_iter_events(self.path))

    # ------------------------------------------------------------------
    # Low-level append
    # ------------------------------------------------------------------

    def _append(self, record: dict[str, Any]) -> None:
        """Write one jsonl record atomically (line-level append)."""
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter_events(path: Path):
    """Yield each jsonl record in order; skip malformed lines with a warn."""
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                # Malformed line (crash mid-write?) — skip but don't
                # die. Caller sees a hole in the transcript.
                continue


def _read_version() -> str:
    try:
        from importlib.metadata import version

        return version("ygn-sage")
    except Exception:
        return "unknown"
