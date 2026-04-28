"""RuntimeEventLog writer. Per-run JSONL files, fail-open default."""
from __future__ import annotations

import contextvars
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, TextIO

from sage.runtime.event_log.errors import EventLogUnavailable
from sage.runtime.event_log.events import (
    _Budget,
    _ControllerDecision,
    _EventCore,
    _Failure,
    _FinalResult,
    _ModelAssigned,
    _NodeCompleted,
    _NodeStarted,
    _RoutingDecision,
    _StateApplied,
    _TaskStarted,
    _TopologySelected,
)
from sage.runtime.event_log.redaction import (
    _hash_payload,
    _hash_text,
    _redact_payload,
)
from sage.runtime.event_log.schema import FINAL_RESULT_STATUSES, SCHEMA_VERSION

log = logging.getLogger(__name__)

_TRACE_DIR_ENV = "SAGE_TRACE_JSONL_DIR"
_TRACE_RAW_ENV = "SAGE_TRACE_RAW"
_FAIL_CLOSED_ENV = "SAGE_TRACE_FAIL_CLOSED"

_current_event_log: contextvars.ContextVar[RuntimeEventLog | None] = (
    contextvars.ContextVar("_current_event_log", default=None)
)


def install_event_log(log_obj: RuntimeEventLog | None) -> contextvars.Token[RuntimeEventLog | None]:
    """Install a RuntimeEventLog into the current async context."""
    return _current_event_log.set(log_obj)


def current_event_log() -> RuntimeEventLog | None:
    """Return the RuntimeEventLog installed in the current context, or None."""
    return _current_event_log.get()


class _SinkHandle:
    """Tiny wrapper around TextIO so tests can monkeypatch write on the instance."""

    def __init__(self, fh: TextIO) -> None:
        self._fh = fh

    @property
    def closed(self) -> bool:
        return self._fh.closed

    def write(self, value: str) -> int:
        return self._fh.write(value)

    def flush(self) -> None:
        self._fh.flush()

    def fileno(self) -> int:
        return self._fh.fileno()

    def close(self) -> None:
        self._fh.close()


class RuntimeEventLog:
    """Per-run event log writing one JSON object per line.

    Default mode is fail-open: the first sink failure logs a warning and
    disables the writer for the rest of the run. Fail-closed mode
    (SAGE_TRACE_FAIL_CLOSED=1) raises EventLogUnavailable instead.
    """

    def __init__(self, run_id: str, trace_dir: Path | str | None = None) -> None:
        self.run_id = run_id
        self.trace_id = run_id
        self.disabled = False
        self._seq = 0
        self._last_event_seq: int | None = None
        self._fail_closed = os.environ.get(_FAIL_CLOSED_ENV) == "1"
        self._raw_payload = os.environ.get(_TRACE_RAW_ENV) == "1"
        self._fh: _SinkHandle | None = None
        self._cached_task_hash = ""
        self._path: Path | None = None

        if trace_dir is None:
            trace_dir_env = os.environ.get(_TRACE_DIR_ENV)
            if not trace_dir_env:
                self.disabled = True
                return
            trace_dir = trace_dir_env

        try:
            trace_path = Path(trace_dir)
            trace_path.mkdir(parents=True, exist_ok=True)
            self._path = trace_path / f"{self.run_id}.jsonl"
            self._fh = _SinkHandle(open(self._path, "x", encoding="utf-8"))
        except (OSError, IOError) as exc:
            self._handle_sink_failure(exc, phase="init")

    def set_task_text(self, task: str) -> None:
        """Cache the task hash used by every event in this run."""
        self._cached_task_hash = _hash_text(task)

    def emit_task_started(self, task: str) -> None:
        self.set_task_text(task)
        self._emit(_TaskStarted, "task_started", "pipeline", payload=task)

    def emit_routing_decision(
        self,
        *,
        routing_source: str,
        system: int,
        domain: str,
        confidence: float | None,
        model_id: str = "",
    ) -> None:
        payload = {
            "routing_source": routing_source,
            "system": system,
            "domain": domain,
            "confidence": round(confidence, 4) if confidence is not None else None,
            "model_id": model_id,
        }
        self._emit(
            _RoutingDecision,
            "routing_decision",
            "pipeline",
            payload=payload,
            routing_source=routing_source,
            system=system,
            domain=domain,
            confidence=confidence,
            model_id=model_id,
        )

    def emit_topology_selected(
        self,
        *,
        topology_id: str,
        template_type: str,
        node_count: int,
        edge_count: int,
        nodes_summary: list[dict[str, Any]],
        edges_summary: list[dict[str, Any]],
    ) -> None:
        payload = {
            "topology_id": topology_id,
            "template_type": template_type,
            "node_count": node_count,
            "edge_count": edge_count,
            "nodes": sorted(nodes_summary, key=lambda item: item.get("node_id", "")),
            "edges": sorted(
                edges_summary,
                key=lambda item: (
                    item.get("source_id", ""),
                    item.get("target_id", ""),
                    item.get("edge_id", ""),
                ),
            ),
        }
        self._emit(
            _TopologySelected,
            "topology_selected",
            "pipeline",
            payload=payload,
            topology_id=topology_id,
            template_type=template_type,
            node_count=node_count,
            edge_count=edge_count,
        )

    def emit_model_assigned(
        self,
        *,
        node_id: str,
        node_role: str,
        model_id: str,
        provider_id: str,
        required_capabilities: tuple[str, ...],
    ) -> None:
        capabilities = tuple(sorted(required_capabilities))
        payload = {
            "node_id": node_id,
            "node_role": node_role,
            "model_id": model_id,
            "provider_id": provider_id,
            "required_capabilities": list(capabilities),
        }
        self._emit(
            _ModelAssigned,
            "model_assigned",
            "pipeline",
            payload=payload,
            node_id=node_id,
            node_role=node_role,
            model_id=model_id,
            provider_id=provider_id,
            required_capabilities=capabilities,
        )

    def emit_node_started(
        self,
        *,
        topology_id: str,
        node_id: str,
        node_role: str,
        attempt: int,
        model_id: str,
        provider_id: str,
        predecessor_ids: tuple[str, ...],
        edge_ids: tuple[str, ...],
        predecessors_by_channel: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        predecessors = tuple(sorted(predecessor_ids))
        edges = tuple(sorted(edge_ids))
        payload = {
            "topology_id": topology_id,
            "node_id": node_id,
            "node_role": node_role,
            "attempt": attempt,
            "model_id": model_id,
            "provider_id": provider_id,
            "predecessor_ids": list(predecessors),
            "edge_ids": list(edges),
        }
        if predecessors_by_channel is not None:
            payload["predecessors_by_channel"] = {
                key: list(values)
                for key, values in sorted(predecessors_by_channel.items())
            }
        self._emit(
            _NodeStarted,
            "node_started",
            "topology_runner",
            payload=payload,
            topology_id=topology_id,
            node_id=node_id,
            node_role=node_role,
            attempt=attempt,
            model_id=model_id,
            provider_id=provider_id,
            predecessor_ids=predecessors,
            edge_ids=edges,
        )

    def emit_state_applied(
        self,
        *,
        target_node_id: str,
        source_node_ids: tuple[str, ...],
        before_version: int,
        after_version: int,
        delta_count: int,
        conflict_count: int,
        applied: bool,
        invalidated_assumption_ids: tuple[str, ...] = (),
    ) -> None:
        sources = tuple(source_node_ids)
        invalidated = tuple(invalidated_assumption_ids)
        payload = {
            "target_node_id": target_node_id,
            "source_node_ids": list(sources),
            "before_version": before_version,
            "after_version": after_version,
            "delta_count": delta_count,
            "conflict_count": conflict_count,
            "applied": applied,
            "invalidated_assumption_ids": list(invalidated),
        }
        self._emit(
            _StateApplied,
            "state_applied",
            "topology_runner",
            payload=payload,
            target_node_id=target_node_id,
            source_node_ids=sources,
            before_version=before_version,
            after_version=after_version,
            delta_count=delta_count,
            conflict_count=conflict_count,
            applied=applied,
            invalidated_assumption_ids=invalidated,
        )

    def emit_node_completed(
        self,
        *,
        node_id: str,
        node_role: str,
        output: str,
        latency_ms: float,
        cost_usd: float,
        model_id: str,
        provider_id: str,
    ) -> None:
        self._emit(
            _NodeCompleted,
            "node_completed",
            "topology_runner",
            payload=output,
            node_id=node_id,
            node_role=node_role,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            output_length=len(output),
            model_id=model_id,
            provider_id=provider_id,
        )

    def emit_controller_decision(
        self,
        *,
        node_id: str,
        action: str,
        target_node_id: str = "",
        gate_source_id: str | None = None,
        gate_target_id: str | None = None,
        reason: str = "",
    ) -> None:
        payload = {
            "action": action,
            "target_node_id": target_node_id,
            "gate_source_id": gate_source_id,
            "gate_target_id": gate_target_id,
            "reason": reason,
        }
        self._emit(
            _ControllerDecision,
            "controller_decision",
            "controller",
            payload=payload,
            node_id=node_id,
            action=action,
            target_node_id=target_node_id,
            gate_source_id=gate_source_id,
            gate_target_id=gate_target_id,
        )

    def emit_failure(
        self,
        *,
        kind: str,
        error_type: str,
        message: str,
        node_id: str = "",
    ) -> None:
        payload = {"kind": kind, "error_type": error_type, "message": message}
        self._emit(
            _Failure,
            "failure",
            "topology_runner",
            payload=payload,
            kind=kind,
            error_type=error_type,
            node_id=node_id,
        )

    def emit_budget(
        self,
        *,
        kind: str,
        budget_limit_usd: float,
        budget_remaining_usd: float,
        cost_so_far_usd: float,
    ) -> None:
        payload = {
            "kind": kind,
            "budget_limit_usd": budget_limit_usd,
            "budget_remaining_usd": budget_remaining_usd,
            "cost_so_far_usd": cost_so_far_usd,
        }
        self._emit(
            _Budget,
            "budget",
            "topology_runner",
            payload=payload,
            kind=kind,
            budget_limit_usd=budget_limit_usd,
            budget_remaining_usd=budget_remaining_usd,
            cost_so_far_usd=cost_so_far_usd,
        )

    def emit_final_result(
        self,
        *,
        status: str,
        output: str,
        total_cost_usd: float,
        total_latency_ms: float,
        node_count: int,
    ) -> None:
        if status not in FINAL_RESULT_STATUSES:
            status = "failure"
        self._emit(
            _FinalResult,
            "final_result",
            "pipeline",
            payload=output,
            status=status,
            output_length=len(output),
            total_cost_usd=total_cost_usd,
            total_latency_ms=total_latency_ms,
            node_count=node_count,
            _is_final=True,
        )

    def close(self) -> None:
        if self._fh is not None and not self._fh.closed:
            try:
                self._fh.close()
            except Exception:  # noqa: BLE001
                pass
        self._fh = None
        self.disabled = True

    def _emit(
        self,
        cls: type[_EventCore],
        event_type: str,
        source_component: str,
        *,
        payload: Any,
        _is_final: bool = False,
        **fields: Any,
    ) -> None:
        if self.disabled or self._fh is None:
            return

        seq = self._seq
        self._seq += 1
        parent = self._last_event_seq
        try:
            payload_redacted = _redact_payload(payload)
            event = cls(
                schema_version=SCHEMA_VERSION,
                run_id=self.run_id,
                trace_id=self.trace_id,
                parent_event_id=parent,
                seq=seq,
                timestamp_ns=time.time_ns(),
                event_type=event_type,
                source_component=source_component,
                task_hash=self._cached_task_hash,
                payload_hash=_hash_payload(event_type, payload),
                redaction_state=self._redaction_state_for(payload),
                payload=payload_redacted if self._raw_payload else None,
                **fields,
            )
            line = json.dumps(
                event.to_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            ) + "\n"
            self._fh.write(line)
            if _is_final:
                self._fh.flush()
                os.fsync(self._fh.fileno())
            self._last_event_seq = seq
        except (OSError, IOError, ValueError, TypeError) as exc:
            self._handle_sink_failure(exc, phase=event_type)

    def _redaction_state_for(self, payload: Any) -> str:
        if payload is None:
            return "none_applicable"
        if self._raw_payload:
            return "raw"
        return "redacted"

    def _handle_sink_failure(self, exc: BaseException, *, phase: str) -> None:
        if self._fail_closed:
            self.close()
            raise EventLogUnavailable(
                f"RuntimeEventLog sink failure during {phase}: {exc}"
            ) from exc
        log.warning(
            "RuntimeEventLog disabled after sink failure during %s: %s. "
            "Pipeline result unaffected; subsequent events for run %s will not be written.",
            phase,
            exc,
            self.run_id,
        )
        self.close()
