"""Writer for the Learning Side-Effect Ledger v0 sidecar."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from sage.runtime.credit_assignment.schema import (
    SCHEMA_VERSION,
    LearningSideEffectSchemaError,
    canonical_json,
    record_hash_input,
    validate_record_shape,
)
from sage.runtime.event_log import current_event_log


LEDGER_FILENAME = "learning_side_effects.jsonl"


def emit_learning_side_effect(record: dict[str, Any]) -> bool:
    """Append one audit-only side-effect record.

    Runtime behavior is fail-open: a sidecar write problem must not alter the
    pipeline result or learning gate. Benchmark code can run the validator and
    fail a gate-quality artifact if this sidecar is absent or invalid.
    """
    try:
        writer = LearningSideEffectLedgerWriter.from_current_event_log()
        if writer is None:
            return False
        writer.emit(record)
        return True
    except Exception:  # noqa: BLE001 - audit sidecar must not affect runtime
        return False


class LearningSideEffectLedgerWriter:
    def __init__(self, path: Path, *, run_id: str, task_hash: str) -> None:
        self.path = path
        self.run_id = run_id
        self.task_hash = task_hash

    @classmethod
    def from_current_event_log(cls) -> "LearningSideEffectLedgerWriter | None":
        event_log = current_event_log()
        if event_log is None or event_log.path is None:
            return None
        return cls(
            event_log.path.parent / LEDGER_FILENAME,
            run_id=event_log.run_id,
            task_hash=getattr(event_log, "_cached_task_hash", ""),
        )

    def emit(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        previous = _read_last_record(self.path)
        prev_hash = previous.get("record_hash") if previous else None
        seq = int(previous.get("seq", -1)) + 1 if previous else 0

        record = dict(record)
        record.update(
            {
                "schema_version": SCHEMA_VERSION,
                "seq": seq,
                "timestamp_ns": time.time_ns(),
                "run_id": self.run_id,
                "trace_id": self.run_id,
                "task_hash": self.task_hash,
                "redaction_state": "redacted",
                "prev_record_hash": prev_hash,
            }
        )
        record["record_hash"] = _hash_record(record)
        validate_record_shape(record)
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _read_last_record(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last_line = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            last_line = line
    if not last_line:
        return None
    value = json.loads(last_line)
    if not isinstance(value, dict):
        raise LearningSideEffectSchemaError("last ledger line is not an object")
    return value


def _hash_record(record: dict[str, Any]) -> str:
    data = record_hash_input(record)
    digest = hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
