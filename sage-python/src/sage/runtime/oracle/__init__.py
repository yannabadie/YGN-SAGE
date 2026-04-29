"""OracleStack v0 - quality verdict hierarchy for trainable-gate."""
from sage.runtime.oracle.config import OracleConfig
from sage.runtime.oracle.errors import OracleUnavailable
from sage.runtime.oracle.verdict import (
    ORACLE_VERDICT_SCHEMA_VERSION,
    QUALITY_LABELS,
    VERDICT_SOURCES,
    EvidenceRef,
    OracleVerdict,
    QualityLabel,
    VerdictSource,
)


def evaluate(*args, **kwargs):
    """Lazy-imported entry point to break circular import with run_frame."""
    from sage.runtime.oracle.stack import evaluate as _evaluate

    return _evaluate(*args, **kwargs)


__all__ = [
    "OracleVerdict",
    "EvidenceRef",
    "VerdictSource",
    "QualityLabel",
    "OracleConfig",
    "OracleUnavailable",
    "evaluate",
    "VERDICT_SOURCES",
    "QUALITY_LABELS",
    "ORACLE_VERDICT_SCHEMA_VERSION",
]
