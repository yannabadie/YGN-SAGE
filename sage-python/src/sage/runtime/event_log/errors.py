"""Typed RuntimeEventLog exceptions."""


class EventLogUnavailable(RuntimeError):
    """Raised when SAGE_TRACE_FAIL_CLOSED=1 and the event log sink fails.

    In default mode, the writer warns once and disables itself; this
    exception propagates only in fail-closed mode.
    """


class EventLogSchemaError(RuntimeError):
    """Raised when an event payload violates the registered payload schema."""
