"""Validator for Learning Side-Effect Ledger v0 sidecars."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sage.runtime.credit_assignment.schema import (
    LearningSideEffectSchemaError,
    canonical_json,
    record_hash_input,
    validate_record_shape,
)
from sage.runtime.credit_assignment.writer import LEDGER_FILENAME


def validate_trace_dir(trace_dir: Path, *, run_id: str | None = None) -> list[dict[str, Any]]:
    runtime_events = _load_runtime_events(trace_dir, run_id=run_id)
    ledger_path = trace_dir / LEDGER_FILENAME
    if not ledger_path.exists():
        raise LearningSideEffectSchemaError(f"missing {LEDGER_FILENAME}")
    records = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise LearningSideEffectSchemaError("empty learning side-effect ledger")

    previous_hash: str | None = None
    for expected_seq, record in enumerate(records):
        if not isinstance(record, dict):
            raise LearningSideEffectSchemaError("ledger line is not an object")
        validate_record_shape(record)
        if record["seq"] != expected_seq:
            raise LearningSideEffectSchemaError("non-monotonic ledger seq")
        if record["prev_record_hash"] != previous_hash:
            raise LearningSideEffectSchemaError("invalid prev_record_hash chain")
        expected_hash = _hash_record(record)
        if record["record_hash"] != expected_hash:
            raise LearningSideEffectSchemaError("invalid record_hash")
        _validate_parent_refs(record, runtime_events)
        _validate_oracle_consistency(record, runtime_events)
        previous_hash = record["record_hash"]
    return records


def _load_runtime_events(
    trace_dir: Path,
    *,
    run_id: str | None,
) -> dict[tuple[str, int], dict[str, Any]]:
    paths = [trace_dir / f"{run_id}.jsonl"] if run_id else sorted(trace_dir.glob("*.jsonl"))
    runtime_paths = [path for path in paths if path.name != LEDGER_FILENAME and path.exists()]
    if not runtime_paths:
        raise LearningSideEffectSchemaError("missing RuntimeEventLog JSONL")
    events: dict[tuple[str, int], dict[str, Any]] = {}
    for path in runtime_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                continue
            event_type = event.get("event_type")
            seq = event.get("seq")
            if isinstance(event_type, str) and isinstance(seq, int):
                events[(event_type, seq)] = event
    return events


def _validate_parent_refs(
    record: dict[str, Any],
    runtime_events: dict[tuple[str, int], dict[str, Any]],
) -> None:
    for ref in record["parent_event_refs"]:
        event = runtime_events.get((ref["event_type"], ref["seq"]))
        if event is None:
            raise LearningSideEffectSchemaError("parent event ref not found")
        if event.get("payload_hash") != ref["payload_hash"]:
            raise LearningSideEffectSchemaError("parent event payload_hash mismatch")


def _validate_oracle_consistency(
    record: dict[str, Any],
    runtime_events: dict[tuple[str, int], dict[str, Any]],
) -> None:
    oracle_ref = record["oracle_verdict_ref"]
    if oracle_ref is None:
        if (
            record["decision"] == "allowed"
            and record["gate"].get("oracle_enabled") is True
            and record["side_effect"] != "bandit_cancel_pending"
        ):
            raise LearningSideEffectSchemaError(
                "allowed oracle-on learning update requires oracle verdict ref"
            )
        return
    event = runtime_events.get(("oracle_verdict", oracle_ref["seq"]))
    if event is None:
        raise LearningSideEffectSchemaError("oracle_verdict ref not found")
    if event.get("payload_hash") != oracle_ref["payload_hash"]:
        raise LearningSideEffectSchemaError("oracle_verdict payload_hash mismatch")
    if (
        record["decision"] == "allowed"
        and record["gate"].get("oracle_enabled") is True
        and record["side_effect"] != "bandit_cancel_pending"
        and oracle_ref["trainable"] is not True
    ):
        raise LearningSideEffectSchemaError(
            "allowed oracle-on learning update requires trainable oracle verdict"
        )
    if (
        record["decision"] == "allowed"
        and record["side_effect"] != "bandit_cancel_pending"
        and record["gate"].get("allow_training_updates") is not True
    ):
        raise LearningSideEffectSchemaError(
            "allowed learning update requires allow_training_updates"
        )


def _hash_record(record: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        canonical_json(record_hash_input(record)).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    records = validate_trace_dir(args.trace_dir, run_id=args.run_id)
    print(json.dumps({"ok": True, "records": len(records)}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
