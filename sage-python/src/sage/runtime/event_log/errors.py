"""Typed RuntimeEventLog exceptions."""


class EventLogUnavailable(RuntimeError):
    """Raised when SAGE_TRACE_FAIL_CLOSED=1 and the event log sink fails.

    In default mode, the writer warns once and disables itself; this
    exception propagates only in fail-closed mode.
    """


class EventLogSchemaError(RuntimeError):
    """Raised when an event payload violates the registered payload schema."""


class EventLogInvariantViolation(RuntimeError):
    """Raised when a runtime-integrity-ledger invariant is violated.

    Invariants per ``docs/contracts/runtime-integrity-ledger.md``
    bind declared event-log labels to verified runtime side effects
    (directive #9: declared ≠ verified). When ``SAGE_TRACE_FAIL_CLOSED=1``,
    a mismatch raises this exception BEFORE the protected side effect
    fires — fail-closed at the side-effect boundary.

    First user: I-11 (slice 10D, 2026-05-11) binds
    ``provider_execution_witness.policy.routing_candidate_decision``
    to ``enforce_provider_policy``'s policy evaluation + the
    subsequent ``failure(error_type=provider_policy_violation)`` event.
    """
