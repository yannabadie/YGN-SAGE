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

_MINIMAL_STAGE5_SIDE_EFFECTS = frozenset(
    {
        "bandit_record_outcome",
        "map_elites_record_outcome",
        "online_evolution_should_evolve",
    }
)


def validate_trace_dir(
    trace_dir: Path,
    *,
    run_id: str | None = None,
    _runtime_events: dict[tuple[str, str, int], dict[str, Any]] | None = None,
    _require_oracle_payload_trainable: bool = False,
) -> list[dict[str, Any]]:
    runtime_events = (
        _load_runtime_events(trace_dir) if _runtime_events is None else _runtime_events
    )
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
        _validate_oracle_consistency(
            record,
            runtime_events,
            require_oracle_payload_trainable=_require_oracle_payload_trainable,
        )
        previous_hash = record["record_hash"]
    if run_id is not None:
        return [record for record in records if record["run_id"] == run_id]
    return records


def validate_evidence_boundary(
    trace_dir: Path,
    *,
    run_id: str,
    expect_default_pipeline_learn: bool = False,
    allow_oracle_disabled: bool = False,
) -> list[dict[str, Any]]:
    if not run_id:
        raise LearningSideEffectSchemaError("evidence-boundary mode requires run_id")
    if not (trace_dir / f"{run_id}.jsonl").exists():
        raise LearningSideEffectSchemaError("missing RuntimeEventLog JSONL for run_id")

    runtime_events = _load_runtime_events(
        trace_dir,
        run_id=run_id,
        canonical_only=True,
    )
    records = validate_trace_dir(
        trace_dir,
        run_id=run_id,
        _runtime_events=runtime_events,
        _require_oracle_payload_trainable=not allow_oracle_disabled,
    )
    if not records:
        raise LearningSideEffectSchemaError("no ledger records for run_id")

    if not allow_oracle_disabled:
        if not any(
            event_run_id == run_id and event_type == "oracle_verdict"
            for event_run_id, event_type, _seq in runtime_events
        ):
            raise LearningSideEffectSchemaError(
                "evidence-boundary mode requires oracle_verdict for run_id"
            )
        if any(record["gate"].get("oracle_enabled") is not True for record in records):
            raise LearningSideEffectSchemaError(
                "oracle-disabled trace is not eligible for oracle-gated evidence"
            )
        _validate_boundary_oracle_backing(records)

    if expect_default_pipeline_learn:
        _validate_stage5_decision_coverage(records)
    return records


def _load_runtime_events(
    trace_dir: Path,
    *,
    run_id: str | None = None,
    canonical_only: bool = False,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    if canonical_only:
        if not run_id:
            raise LearningSideEffectSchemaError(
                "canonical RuntimeEventLog loading requires run_id"
            )
        runtime_paths = [trace_dir / f"{run_id}.jsonl"]
    else:
        paths = sorted(trace_dir.glob("*.jsonl"))
        runtime_paths = [
            path for path in paths if path.name != LEDGER_FILENAME and path.exists()
        ]
    if not runtime_paths:
        raise LearningSideEffectSchemaError("missing RuntimeEventLog JSONL")
    events: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in runtime_paths:
        if not path.exists():
            raise LearningSideEffectSchemaError("missing RuntimeEventLog JSONL")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                continue
            event_run_id = event.get("run_id")
            event_type = event.get("event_type")
            seq = event.get("seq")
            if (
                isinstance(event_run_id, str)
                and isinstance(event_type, str)
                and isinstance(seq, int)
            ):
                if canonical_only and event_run_id != run_id:
                    raise LearningSideEffectSchemaError(
                        "RuntimeEventLog run_id mismatch"
                    )
                key = (event_run_id, event_type, seq)
                if key in events:
                    raise LearningSideEffectSchemaError(
                        "duplicate RuntimeEventLog event key"
                    )
                events[key] = event
    return events


def _validate_parent_refs(
    record: dict[str, Any],
    runtime_events: dict[tuple[str, str, int], dict[str, Any]],
) -> None:
    for ref in record["parent_event_refs"]:
        event = runtime_events.get((record["run_id"], ref["event_type"], ref["seq"]))
        if event is None:
            raise LearningSideEffectSchemaError("parent event ref not found")
        if event.get("payload_hash") != ref["payload_hash"]:
            raise LearningSideEffectSchemaError("parent event payload_hash mismatch")


def _validate_oracle_consistency(
    record: dict[str, Any],
    runtime_events: dict[tuple[str, str, int], dict[str, Any]],
    *,
    require_oracle_payload_trainable: bool = False,
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
    event = runtime_events.get((record["run_id"], "oracle_verdict", oracle_ref["seq"]))
    if event is None:
        raise LearningSideEffectSchemaError("oracle_verdict ref not found")
    if event.get("payload_hash") != oracle_ref["payload_hash"]:
        raise LearningSideEffectSchemaError("oracle_verdict payload_hash mismatch")
    event_trainable = oracle_ref["trainable"]
    payload = event.get("payload")
    if isinstance(payload, dict) and isinstance(payload.get("trainable"), bool):
        event_trainable = payload["trainable"]
        if oracle_ref["trainable"] is not event_trainable:
            raise LearningSideEffectSchemaError(
                "oracle_verdict_ref.trainable does not match RuntimeEventLog payload"
            )
    elif require_oracle_payload_trainable:
        raise LearningSideEffectSchemaError(
            "evidence-boundary oracle_verdict requires payload.trainable"
        )
    if (
        record["decision"] == "allowed"
        and record["gate"].get("oracle_enabled") is True
        and record["side_effect"] != "bandit_cancel_pending"
        and event_trainable is not True
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


def _validate_boundary_oracle_backing(records: list[dict[str, Any]]) -> None:
    for record in records:
        if record["side_effect"] == "bandit_cancel_pending":
            continue
        if record["oracle_verdict_ref"] is None:
            raise LearningSideEffectSchemaError(
                "evidence-boundary record requires oracle_verdict_ref"
            )


def _validate_stage5_decision_coverage(
    records: list[dict[str, Any]],
) -> None:
    """Fail closed for RC evidence traces declared as post-learn captures.

    This is intentionally a minimum coverage check, not a proof that every
    possible side-effect path is exhaustively represented. RuntimeEventLog does
    not expose consolidator state or online-evolution True branches, so the
    defensible gate is the three decisions Stage 5 should always account for
    in the current default oracle-enabled pipeline when a harness explicitly
    declares that learning completed.
    """
    observed = {record["side_effect"] for record in records}
    missing = sorted(_MINIMAL_STAGE5_SIDE_EFFECTS - observed)
    if missing:
        raise LearningSideEffectSchemaError(
            "missing required stage5 side-effect decision(s): "
            + ", ".join(missing)
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
    parser.add_argument(
        "--mode",
        choices=("audit", "evidence-boundary"),
        default="audit",
    )
    parser.add_argument(
        "--expect-default-pipeline-learn",
        action="store_true",
        help=(
            "In evidence-boundary mode, require the current default pipeline's "
            "minimal Stage 5 learning side-effect decision set."
        ),
    )
    parser.add_argument(
        "--allow-oracle-disabled",
        action="store_true",
        help="Allow evidence-boundary validation for oracle-disabled legacy traces.",
    )
    args = parser.parse_args(argv)
    if args.mode == "evidence-boundary":
        if not args.run_id:
            parser.error("--mode evidence-boundary requires --run-id")
        records = validate_evidence_boundary(
            args.trace_dir,
            run_id=args.run_id,
            expect_default_pipeline_learn=args.expect_default_pipeline_learn,
            allow_oracle_disabled=args.allow_oracle_disabled,
        )
    else:
        records = validate_trace_dir(args.trace_dir, run_id=args.run_id)
    print(
        json.dumps(
            {"ok": True, "mode": args.mode, "records": len(records)},
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
