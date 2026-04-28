"""StateCore v0 - typed state channel for topology runners."""
from sage.runtime.state.errors import StateConflict
from sage.runtime.state.frame import (
    EvidenceRef,
    StateApplyResult,
    StateDelta,
    StateFrame,
)
from sage.runtime.state.reducer import (
    apply_delta,
    apply_deltas,
    normalize_assumption_id,
)

__all__ = [
    "StateFrame",
    "StateDelta",
    "StateApplyResult",
    "EvidenceRef",
    "StateConflict",
    "apply_delta",
    "apply_deltas",
    "normalize_assumption_id",
]
