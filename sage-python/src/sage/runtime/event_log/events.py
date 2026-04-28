"""Private event dataclasses. Do not export publicly in v0."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class _EventCore:
    schema_version: str
    run_id: str
    trace_id: str
    parent_event_id: int | None
    seq: int
    timestamp_ns: int
    event_type: str
    source_component: str
    task_hash: str
    payload_hash: str
    redaction_state: str
    payload: Any | None = None
    edge_type: str | None = None
    channel: str | None = None
    state_version: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d.get("payload") is None:
            d.pop("payload", None)
        for key in ("edge_type", "channel", "state_version"):
            if d.get(key) is None:
                d.pop(key, None)
        return d


@dataclass(frozen=True)
class _TaskStarted(_EventCore):
    pass


@dataclass(frozen=True)
class _RoutingDecision(_EventCore):
    routing_source: str = ""
    system: int = 0
    domain: str = ""
    confidence: float | None = None
    model_id: str = ""


@dataclass(frozen=True)
class _TopologySelected(_EventCore):
    topology_id: str = ""
    template_type: str = ""
    node_count: int = 0
    edge_count: int = 0


@dataclass(frozen=True)
class _ModelAssigned(_EventCore):
    node_id: str = ""
    node_role: str = ""
    model_id: str = ""
    provider_id: str = ""
    required_capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class _NodeStarted(_EventCore):
    topology_id: str = ""
    node_id: str = ""
    node_role: str = ""
    attempt: int = 1
    model_id: str = ""
    provider_id: str = ""
    predecessor_ids: tuple[str, ...] = ()
    edge_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _NodeCompleted(_EventCore):
    node_id: str = ""
    node_role: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    output_length: int = 0
    model_id: str = ""
    provider_id: str = ""


@dataclass(frozen=True)
class _ControllerDecision(_EventCore):
    node_id: str = ""
    action: str = "continue"
    target_node_id: str = ""
    gate_source_id: str | None = None
    gate_target_id: str | None = None


@dataclass(frozen=True)
class _StateApplied(_EventCore):
    target_node_id: str = ""
    source_node_ids: tuple[str, ...] = ()
    before_version: int = 0
    after_version: int = 0
    delta_count: int = 0
    conflict_count: int = 0
    applied: bool = True
    invalidated_assumption_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Failure(_EventCore):
    kind: str = ""
    error_type: str = ""
    node_id: str = ""


@dataclass(frozen=True)
class _Budget(_EventCore):
    kind: str = ""
    budget_limit_usd: float = 0.0
    budget_remaining_usd: float = 0.0
    cost_so_far_usd: float = 0.0


@dataclass(frozen=True)
class _FinalResult(_EventCore):
    status: str = "success"
    output_length: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    node_count: int = 0


@dataclass(frozen=True)
class _RunFrameSummary(_EventCore):
    pass
