"""Learning Side-Effect Ledger v0 sidecar APIs."""

from pathlib import Path
from typing import Any

from sage.runtime.credit_assignment.schema import (
    DECISIONS,
    REASON_CODES,
    SCHEMA_VERSION,
    SIDE_EFFECTS,
    LearningSideEffectSchemaError,
)
from sage.runtime.credit_assignment.writer import (
    LEDGER_FILENAME,
    emit_learning_side_effect,
)


def validate_trace_dir(
    trace_dir: Path,
    *,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Validate a Learning Side-Effect Ledger trace directory.

    Kept lazy so `python -m sage.runtime.credit_assignment.validate` does not
    pre-import its target module through this package initializer.
    """
    from sage.runtime.credit_assignment.validate import (
        validate_trace_dir as _validate_trace_dir,
    )

    return _validate_trace_dir(trace_dir, run_id=run_id)


def validate_evidence_boundary(
    trace_dir: Path,
    *,
    run_id: str,
    expect_default_pipeline_learn: bool = False,
    allow_oracle_disabled: bool = False,
) -> list[dict[str, Any]]:
    """Fail-closed validation for explicitly claimed evidence-boundary traces."""
    from sage.runtime.credit_assignment.validate import (
        validate_evidence_boundary as _validate_evidence_boundary,
    )

    return _validate_evidence_boundary(
        trace_dir,
        run_id=run_id,
        expect_default_pipeline_learn=expect_default_pipeline_learn,
        allow_oracle_disabled=allow_oracle_disabled,
    )


__all__ = [
    "DECISIONS",
    "LEDGER_FILENAME",
    "REASON_CODES",
    "SCHEMA_VERSION",
    "SIDE_EFFECTS",
    "LearningSideEffectSchemaError",
    "emit_learning_side_effect",
    "validate_evidence_boundary",
    "validate_trace_dir",
]
