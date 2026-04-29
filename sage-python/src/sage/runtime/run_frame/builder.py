from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from types import MappingProxyType
from typing import Any, Mapping

from sage.runtime.oracle.verdict import OracleVerdict
from sage.runtime.run_frame.frame import (
    RUN_FRAME_SCHEMA_VERSION,
    NodeRunRecord,
    NodeRunStatus,
    RunFrame,
    RunFrameView,
    RunStatus,
    TopologyRunRef,
)
from sage.runtime.state import StateDelta, StateFrame


class _RunFrameBuilder:
    """Private hot accumulator for RunFrame data during a pipeline run.

    NOT thread-safe. NOT public. Only the pipeline holds the live builder.
    On finalize(), produces a frozen RunFrame.
    """

    _ALLOWED_FEATURE_FLAGS: frozenset[str] = frozenset(
        {
            "SAGE_RUN_FRAME",
            "SAGE_STATECORE",
            "SAGE_TRACE_JSONL_DIR",
            "SAGE_TRACE_RAW",
            "SAGE_TRACE_FAIL_CLOSED",
            "SAGE_DIFF_VERIFIER_MODE",
            "SAGE_ENABLE_PATH6",
            "SAGE_ORACLE",
        }
    )
    _PATH_LIKE_FLAGS: frozenset[str] = frozenset({"SAGE_TRACE_JSONL_DIR"})

    def __init__(self, *, run_id: str, task_id: str, task_hash: str) -> None:
        self.run_id = run_id
        self._task_id = task_id
        self._task_hash = task_hash
        self._feature_flags: dict[str, str] = {}
        self._topology_history: list[TopologyRunRef] = []
        self._state_frames: dict[str, StateFrame] = {}
        self._node_records: dict[str, NodeRunRecord] = {}
        self._open_node_runs: dict[tuple[int, str], str] = {}
        self._model_assignments: dict[str, dict[str, Any]] = {}
        self._pending_state: dict[str, dict[str, Any]] = {}
        self._routing_decision: dict[str, Any] | None = None
        self._controller_decisions: list[dict[str, Any]] = []
        self._budget_snapshot: dict[str, Any] | None = None
        self._final_result_seq: int | None = None
        self._failure_seqs: list[int] = []
        self._terminal_failure_seq: int | None = None
        self._status: RunStatus = "unknown"
        self._oracle_verdict: OracleVerdict | None = None
        self._oracle_verdict_seq: int | None = None
        self._current_topology_epoch = 0
        self._current_topology_id: str | None = None
        self._current_graph_digest: str | None = None

    def capture_feature_flags(self, env: Mapping[str, str] | None = None) -> None:
        """Snapshot the allowlisted env vars at run init. Called once."""
        source = env if env is not None else os.environ
        captured: dict[str, str] = {}
        for key in sorted(self._ALLOWED_FEATURE_FLAGS):
            value = source.get(key, "")
            if key in self._PATH_LIKE_FLAGS and value:
                captured[key] = "<path>"
            else:
                captured[key] = value
        self._feature_flags = captured

    def record_routing_decision(self, *, seq: int | None, **kwargs: Any) -> None:
        decision = {key: _copy_value(value) for key, value in sorted(kwargs.items())}
        decision["seq"] = seq
        self._routing_decision = decision

    def record_topology_selected(
        self,
        *,
        seq: int | None,
        topology_id: str | None,
        graph_digest: str | None,
        reason: str = "initial",
    ) -> None:
        if self._topology_history and reason in {"reroute", "fallback"}:
            self._current_topology_epoch += 1
        self._current_topology_id = topology_id
        self._current_graph_digest = graph_digest
        self._topology_history.append(
            TopologyRunRef(
                topology_epoch=self._current_topology_epoch,
                topology_id=topology_id,
                graph_digest=graph_digest,
                topology_selected_seq=seq,
                reason=reason,
            )
        )

    def record_model_assigned(
        self,
        *,
        seq: int | None,
        node_id: str,
        node_role: str,
        model_id: str,
        provider_id: str,
        required_capabilities: tuple[str, ...] = (),
    ) -> None:
        self._model_assignments[str(node_id)] = {
            "seq": seq,
            "node_role": node_role,
            "model_id": model_id,
            "provider_id": provider_id,
            "required_capabilities": tuple(required_capabilities),
        }

    def record_node_started(
        self,
        *,
        seq: int | None,
        node_id: str,
        provider_id: str,
        model_id: str,
        predecessor_ids: tuple[str, ...] = (),
        predecessors_by_channel: Mapping[str, tuple[str, ...]] | None = None,
        input_context_hash: str | None = None,
    ) -> str:
        node_id = str(node_id)
        attempt = self._next_attempt(self._current_topology_epoch, node_id)
        node_run_id = f"{self._current_topology_epoch}:{node_id}:{attempt}"
        pending = self._pending_state.pop(node_id, {})
        assignment = self._model_assignments.get(node_id, {})
        model_assigned_seq = assignment.get("seq")
        record = NodeRunRecord(
            node_run_id=node_run_id,
            topology_epoch=self._current_topology_epoch,
            node_id=node_id,
            attempt=attempt,
            status="running",
            provider_id=str(provider_id or assignment.get("provider_id", "")),
            model_id=str(model_id or assignment.get("model_id", "")),
            predecessor_ids=tuple(str(item) for item in predecessor_ids),
            predecessors_by_channel=(
                {
                    str(key): tuple(str(item) for item in values)
                    for key, values in sorted(predecessors_by_channel.items())
                }
                if predecessors_by_channel is not None
                else None
            ),
            node_started_seq=seq,
            node_completed_seq=None,
            failure_seq=None,
            controller_decision_seqs=(),
            state_applied_seqs=tuple(pending.get("state_applied_seqs", ())),
            event_seqs=_seq_tuple(
                model_assigned_seq,
                *pending.get("state_applied_seqs", ()),
                seq,
            ),
            input_context_hash=input_context_hash,
            output_sha256=None,
            output_length=None,
            state_before_version=pending.get("state_before_version"),
            state_after_version=pending.get("state_after_version"),
            state_delta=pending.get("state_delta"),
            quality_snapshot={},
            cost_snapshot={},
        )
        self._node_records[node_run_id] = record
        self._open_node_runs[(self._current_topology_epoch, node_id)] = node_run_id
        return node_run_id

    def record_node_completed(
        self,
        *,
        seq: int | None,
        node_run_id: str,
        output: str = "",
        latency_ms: float | None = None,
        cost_usd: float | None = None,
        quality_snapshot: Mapping[str, Any] | None = None,
        cost_snapshot: Mapping[str, Any] | None = None,
    ) -> None:
        record = self._node_records.get(node_run_id)
        if record is None:
            return
        merged_cost: dict[str, Any] = {}
        if cost_snapshot:
            merged_cost.update(dict(cost_snapshot))
        if latency_ms is not None:
            merged_cost["latency_ms"] = latency_ms
        if cost_usd is not None:
            merged_cost["cost_usd"] = cost_usd
        self._node_records[node_run_id] = replace(
            record,
            status="success",
            node_completed_seq=seq,
            event_seqs=_append_seq(record.event_seqs, seq),
            output_sha256=_sha256_text(output),
            output_length=len(output),
            quality_snapshot=dict(quality_snapshot or record.quality_snapshot),
            cost_snapshot=merged_cost or dict(record.cost_snapshot),
        )
        self._open_node_runs.pop((record.topology_epoch, record.node_id), None)

    def record_failure(
        self,
        *,
        seq: int | None,
        node_run_id: str | None,
        kind: str,
        error_type: str = "",
        message: str = "",
    ) -> None:
        if seq is not None:
            self._failure_seqs.append(seq)
            self._terminal_failure_seq = seq
        if node_run_id is None:
            return
        record = self._node_records.get(node_run_id)
        if record is None:
            return
        status: NodeRunStatus = "budget_exceeded" if kind == "budget_exceeded" else "failure"
        quality_snapshot = dict(record.quality_snapshot)
        if kind:
            quality_snapshot["failure_kind"] = kind
        if error_type:
            quality_snapshot["error_type"] = error_type
        if message:
            quality_snapshot["message_hash"] = _sha256_text(message)
        self._node_records[node_run_id] = replace(
            record,
            status=status,
            failure_seq=seq,
            event_seqs=_append_seq(record.event_seqs, seq),
            quality_snapshot=quality_snapshot,
        )
        self._open_node_runs.pop((record.topology_epoch, record.node_id), None)

    def record_controller_decision(
        self,
        *,
        seq: int | None,
        node_run_id: str | None,
        action: str,
        **kwargs: Any,
    ) -> None:
        decision = {"seq": seq, "node_run_id": node_run_id, "action": action}
        decision.update({key: _copy_value(value) for key, value in sorted(kwargs.items())})
        self._controller_decisions.append(decision)
        if node_run_id is None:
            return
        record = self._node_records.get(node_run_id)
        if record is None:
            return
        self._node_records[node_run_id] = replace(
            record,
            controller_decision_seqs=_append_seq(record.controller_decision_seqs, seq),
            event_seqs=_append_seq(record.event_seqs, seq),
        )

    def record_state_applied(
        self,
        *,
        seq: int | None,
        target_node_id: str,
        before_version: int,
        after_version: int,
        state_delta: StateDelta | None = None,
        state_frame: StateFrame | None = None,
        **_kwargs: Any,
    ) -> None:
        target = str(target_node_id)
        if state_frame is not None:
            self._state_frames[target] = state_frame
        node_run_id = self._open_node_runs.get((self._current_topology_epoch, target))
        if node_run_id is None:
            pending = self._pending_state.setdefault(target, {})
            pending["state_before_version"] = before_version
            pending["state_after_version"] = after_version
            pending["state_delta"] = state_delta
            pending["state_applied_seqs"] = _append_seq(
                tuple(pending.get("state_applied_seqs", ())),
                seq,
            )
            return
        record = self._node_records.get(node_run_id)
        if record is None:
            return
        self._node_records[node_run_id] = replace(
            record,
            state_before_version=before_version,
            state_after_version=after_version,
            state_delta=state_delta,
            state_applied_seqs=_append_seq(record.state_applied_seqs, seq),
            event_seqs=_append_seq(record.event_seqs, seq),
        )

    def record_budget(self, *, seq: int | None, **kwargs: Any) -> None:
        snapshot = {key: _copy_value(value) for key, value in sorted(kwargs.items())}
        snapshot["seq"] = seq
        self._budget_snapshot = snapshot

    def record_final_result(self, *, seq: int | None, status: RunStatus, **_kwargs: Any) -> None:
        self._final_result_seq = seq
        self._status = status

    def record_oracle_verdict(self, *, seq: int | None, verdict: OracleVerdict) -> None:
        self._oracle_verdict_seq = seq
        self._oracle_verdict = verdict

    def open_node_run_id(self, *, node_id: str) -> str | None:
        return self._open_node_runs.get((self._current_topology_epoch, str(node_id)))

    def snapshot_view(self) -> RunFrameView:
        """Return a raw-output-free, immutable view for OracleStack."""
        status = self._status
        if status == "unknown" and self._terminal_failure_seq is not None:
            status = "failure"
        return RunFrameView(
            run_id=self.run_id,
            task_id=self._task_id,
            task_hash=self._task_hash,
            feature_flags=_freeze_mapping(self._feature_flags),
            topology_id=self._current_topology_id,
            graph_digest=self._current_graph_digest,
            topology_history=tuple(self._topology_history),
            state_frames=_freeze_mapping(
                {key: _freeze_state_frame(value) for key, value in self._state_frames.items()}
            ),
            node_records=_freeze_mapping(
                {
                    key: _freeze_node_record(value)
                    for key, value in self._node_records.items()
                }
            ),
            routing_decision=(
                _freeze_mapping(self._routing_decision)
                if self._routing_decision is not None
                else None
            ),
            controller_decisions=tuple(
                _freeze_mapping(decision) for decision in self._controller_decisions
            ),
            budget_snapshot=(
                _freeze_mapping(self._budget_snapshot)
                if self._budget_snapshot is not None
                else None
            ),
            final_result_seq=self._final_result_seq,
            failure_seqs=tuple(self._failure_seqs),
            terminal_failure_seq=self._terminal_failure_seq,
            status=status,
        )

    def finalize(self) -> RunFrame:
        """Produce a frozen RunFrame snapshot. Called once per run."""
        status = self._status
        if status == "unknown" and self._terminal_failure_seq is not None:
            status = "failure"
        return RunFrame(
            schema_version=RUN_FRAME_SCHEMA_VERSION,
            run_id=self.run_id,
            task_id=self._task_id,
            task_hash=self._task_hash,
            feature_flags=_freeze_mapping(self._feature_flags),
            topology_id=self._current_topology_id,
            graph_digest=self._current_graph_digest,
            topology_history=tuple(self._topology_history),
            state_frames=_freeze_mapping(
                {key: _freeze_state_frame(value) for key, value in self._state_frames.items()}
            ),
            node_records=_freeze_mapping(
                {
                    key: _freeze_node_record(value)
                    for key, value in self._node_records.items()
                }
            ),
            routing_decision=(
                _freeze_mapping(self._routing_decision)
                if self._routing_decision is not None
                else None
            ),
            controller_decisions=tuple(
                _freeze_mapping(decision) for decision in self._controller_decisions
            ),
            budget_snapshot=(
                _freeze_mapping(self._budget_snapshot)
                if self._budget_snapshot is not None
                else None
            ),
            final_result_seq=self._final_result_seq,
            failure_seqs=tuple(self._failure_seqs),
            terminal_failure_seq=self._terminal_failure_seq,
            status=status,
            oracle_verdict=self._oracle_verdict,
        )

    def _next_attempt(self, topology_epoch: int, node_id: str) -> int:
        attempts = [
            record.attempt
            for record in self._node_records.values()
            if record.topology_epoch == topology_epoch and record.node_id == node_id
        ]
        return max(attempts, default=0) + 1


def _seq_tuple(*seqs: int | None) -> tuple[int, ...]:
    return tuple(seq for seq in seqs if seq is not None)


def _append_seq(existing: tuple[int, ...], seq: int | None) -> tuple[int, ...]:
    if seq is None:
        return existing
    return (*existing, seq)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _copy_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _copy_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_copy_value(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, StateFrame):
        return _freeze_state_frame(value)
    if isinstance(value, StateDelta):
        return _freeze_state_delta(value)
    return value


def _freeze_state_delta(delta: StateDelta) -> StateDelta:
    return StateDelta(
        add_constraints=tuple(delta.add_constraints),
        remove_constraints=tuple(delta.remove_constraints),
        add_assumptions=tuple(delta.add_assumptions),
        invalidate_assumptions=tuple(delta.invalidate_assumptions),
        update_entities=_freeze_mapping(
            {
                key: _freeze_mapping(dict(value))
                for key, value in delta.update_entities.items()
            }
        ),
        add_decisions=tuple(_freeze_mapping(dict(item)) for item in delta.add_decisions),
        add_tool_facts=tuple(_freeze_mapping(dict(item)) for item in delta.add_tool_facts),
        add_open_questions=tuple(delta.add_open_questions),
        close_open_questions=tuple(delta.close_open_questions),
        evidence=tuple(delta.evidence),
    )


def _freeze_state_frame(frame: StateFrame) -> StateFrame:
    return StateFrame(
        task_id=frame.task_id,
        version=frame.version,
        objective=frame.objective,
        constraints=tuple(frame.constraints),
        assumptions=tuple(frame.assumptions),
        invalidated_assumptions=tuple(frame.invalidated_assumptions),
        entities=_freeze_mapping(
            {
                key: _freeze_mapping(dict(value))
                for key, value in frame.entities.items()
            }
        ),
        decisions=tuple(_freeze_mapping(dict(item)) for item in frame.decisions),
        tool_facts=tuple(_freeze_mapping(dict(item)) for item in frame.tool_facts),
        open_questions=tuple(frame.open_questions),
        causal_edges=tuple(frame.causal_edges),
        confidence=frame.confidence,
    )


def _freeze_node_record(record: NodeRunRecord) -> NodeRunRecord:
    return replace(
        record,
        predecessors_by_channel=(
            _freeze_mapping(dict(record.predecessors_by_channel))
            if record.predecessors_by_channel is not None
            else None
        ),
        state_delta=(
            _freeze_state_delta(record.state_delta)
            if record.state_delta is not None
            else None
        ),
        quality_snapshot=_freeze_mapping(record.quality_snapshot),
        cost_snapshot=_freeze_mapping(record.cost_snapshot),
    )
