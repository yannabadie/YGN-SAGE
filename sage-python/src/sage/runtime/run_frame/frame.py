from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import TYPE_CHECKING, Any, Literal, Mapping

from sage.runtime.state import StateDelta, StateFrame

if TYPE_CHECKING:
    from sage.runtime.oracle import OracleVerdict
    from sage.runtime.evidence import RuntimeDelta

RUN_FRAME_SCHEMA_VERSION: Literal["0"] = "0"

NodeRunStatus = Literal[
    "running",
    "success",
    "failure",
    "skipped",
    "pruned",
    "budget_exceeded",
]
RunStatus = Literal["success", "failure", "budget_exceeded", "unknown"]


@dataclass(frozen=True, slots=True)
class NodeRunRecord:
    node_run_id: str  # f"{topology_epoch}:{node_id}:{attempt}"
    topology_epoch: int
    node_id: str
    attempt: int

    status: NodeRunStatus
    provider_id: str
    model_id: str

    predecessor_ids: tuple[str, ...]
    predecessors_by_channel: Mapping[str, tuple[str, ...]] | None

    node_started_seq: int | None
    node_completed_seq: int | None
    failure_seq: int | None
    controller_decision_seqs: tuple[int, ...]
    state_applied_seqs: tuple[int, ...]
    event_seqs: tuple[int, ...]

    input_context_hash: str | None
    output_sha256: str | None
    output_length: int | None

    state_before_version: int | None
    state_after_version: int | None
    state_delta: StateDelta | None

    quality_snapshot: Mapping[str, Any]
    cost_snapshot: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TopologyRunRef:
    topology_epoch: int
    topology_id: str | None
    graph_digest: str | None
    topology_selected_seq: int | None
    reason: str  # "initial" | "reroute" | "fallback"


@dataclass(frozen=True, slots=True)
class RunFrameView:
    """Read-only in-flight RunFrame view for OracleStack. No raw outputs."""

    run_id: str
    task_id: str
    task_hash: str
    feature_flags: Mapping[str, str]
    topology_id: str | None
    graph_digest: str | None
    topology_history: tuple[TopologyRunRef, ...]
    state_frames: Mapping[str, StateFrame]
    node_records: Mapping[str, NodeRunRecord]
    routing_decision: Mapping[str, Any] | None
    controller_decisions: tuple[Mapping[str, Any], ...]
    budget_snapshot: Mapping[str, Any] | None
    final_result_seq: int | None
    failure_seqs: tuple[int, ...]
    terminal_failure_seq: int | None
    status: RunStatus
    runtime_deltas: tuple["RuntimeDelta", ...] = ()


@dataclass(frozen=True, slots=True)
class RunFrame:
    schema_version: Literal["0"]
    run_id: str
    task_id: str
    task_hash: str
    feature_flags: Mapping[str, str]
    topology_id: str | None
    graph_digest: str | None
    topology_history: tuple[TopologyRunRef, ...]
    state_frames: Mapping[str, StateFrame]
    node_records: Mapping[str, NodeRunRecord]  # keyed by node_run_id
    routing_decision: Mapping[str, Any] | None
    controller_decisions: tuple[Mapping[str, Any], ...]
    budget_snapshot: Mapping[str, Any] | None
    final_result_seq: int | None
    failure_seqs: tuple[int, ...]
    terminal_failure_seq: int | None
    status: RunStatus
    runtime_deltas: tuple["RuntimeDelta", ...] = ()
    oracle_verdict: "OracleVerdict | None" = None

    def to_summary_dict(self, *, redacted: bool = True) -> dict[str, Any]:
        """Return a raw-output-free summary for the diagnostic JSONL event."""
        node_records = {
            node_run_id: {
                "node_run_id": record.node_run_id,
                "topology_epoch": record.topology_epoch,
                "node_id": record.node_id,
                "attempt": record.attempt,
                "status": record.status,
                "node_started_seq": record.node_started_seq,
                "node_completed_seq": record.node_completed_seq,
                "failure_seq": record.failure_seq,
                "controller_decision_seqs": list(record.controller_decision_seqs),
                "state_applied_seqs": list(record.state_applied_seqs),
                "event_seqs": list(record.event_seqs),
                "input_context_hash": record.input_context_hash,
                "output_sha256": record.output_sha256,
                "output_length": record.output_length,
                "state_before_version": record.state_before_version,
                "state_after_version": record.state_after_version,
            }
            for node_run_id, record in sorted(self.node_records.items())
        }
        summary: dict[str, Any] = {
            "run_frame_schema_version": self.schema_version,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "task_hash": self.task_hash,
            "status": self.status,
            "topology_id": self.topology_id,
            "graph_digest": self.graph_digest,
            "topology_history": [
                {
                    "topology_epoch": ref.topology_epoch,
                    "topology_id": ref.topology_id,
                    "graph_digest": ref.graph_digest,
                    "topology_selected_seq": ref.topology_selected_seq,
                    "reason": ref.reason,
                }
                for ref in self.topology_history
            ],
            "node_record_count": len(self.node_records),
            "node_records": node_records,
            "state_frame_count": len(self.state_frames),
            "controller_decision_count": len(self.controller_decisions),
            "runtime_delta_count": len(self.runtime_deltas),
            "runtime_delta_hashes": [
                delta.evidence_hash
                for delta in self.runtime_deltas
                if delta.evidence_hash is not None
            ],
            "final_result_seq": self.final_result_seq,
            "failure_seqs": list(self.failure_seqs),
            "terminal_failure_seq": self.terminal_failure_seq,
            "budget_snapshot": _plain(self.budget_snapshot),
            "feature_flags": dict(sorted(self.feature_flags.items())),
            "redacted": redacted,
        }
        if self.oracle_verdict is not None:
            summary["oracle_verdict"] = self.oracle_verdict.to_dict()
        canonical = json.dumps(
            {key: value for key, value in summary.items() if key != "run_frame_hash"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        summary["run_frame_hash"] = hashlib.sha256(canonical).hexdigest()
        return summary


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value
