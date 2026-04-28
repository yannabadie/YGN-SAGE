"""RuntimeEventLog v0: typed runtime event taxonomy and JSONL durable sink.

Internal execution contract ledger for debugging and R6 StateCore prep.
Distinct from sage.observability.spans (OTel external observability).
"""
from sage.runtime.event_log.errors import EventLogUnavailable
from sage.runtime.event_log.schema import (
    CONTROLLER_ACTIONS,
    EVENT_TYPES,
    REDACTION_STATES,
    SCHEMA_VERSION,
    SOURCE_COMPONENTS,
)
from sage.runtime.event_log.writer import (
    RuntimeEventLog,
    current_event_log,
    install_event_log,
)

__all__ = [
    "RuntimeEventLog",
    "install_event_log",
    "current_event_log",
    "SCHEMA_VERSION",
    "EVENT_TYPES",
    "REDACTION_STATES",
    "SOURCE_COMPONENTS",
    "CONTROLLER_ACTIONS",
    "EventLogUnavailable",
]
