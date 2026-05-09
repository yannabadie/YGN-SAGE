"""RuntimeEventLog v0: typed runtime event taxonomy and JSONL durable sink.

Internal execution contract ledger for debugging and R6 StateCore prep.
Distinct from sage.observability.spans (OTel external observability).
"""
from sage.runtime.event_log.errors import EventLogSchemaError, EventLogUnavailable
from sage.runtime.event_log.payload_schemas import (
    CURRENT_PAYLOAD_SCHEMA_VERSIONS,
    DEFAULT_PAYLOAD_STRING_MAX_BYTES,
    PAYLOAD_SCHEMAS,
    EventPayloadSchema,
    PayloadFieldSpec,
    PayloadSchemaVersion,
    get_current_payload_schema_version,
    get_schema_for_event,
)
from sage.runtime.event_log.schema import (
    CONTROLLER_ACTIONS,
    EVENT_TYPES,
    REDACTION_STATES,
    SCHEMA_VERSION,
    SOURCE_COMPONENTS,
)
from sage.runtime.event_log.writer import (
    EventRef,
    RuntimeEventLog,
    current_event_log,
    install_event_log,
)

__all__ = [
    "RuntimeEventLog",
    "EventRef",
    "install_event_log",
    "current_event_log",
    "SCHEMA_VERSION",
    "EVENT_TYPES",
    "REDACTION_STATES",
    "SOURCE_COMPONENTS",
    "CONTROLLER_ACTIONS",
    "EventLogUnavailable",
    "EventLogSchemaError",
    "PayloadSchemaVersion",
    "PayloadFieldSpec",
    "EventPayloadSchema",
    "PAYLOAD_SCHEMAS",
    "CURRENT_PAYLOAD_SCHEMA_VERSIONS",
    "DEFAULT_PAYLOAD_STRING_MAX_BYTES",
    "get_schema_for_event",
    "get_current_payload_schema_version",
]
