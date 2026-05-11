"""RuntimeEventLog writer. Per-run JSONL files, fail-open default."""
from __future__ import annotations

import contextvars
import json
import logging
import math
import os
import re
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Literal, TextIO, get_args, cast

from sage.runtime.event_log.errors import EventLogSchemaError, EventLogUnavailable
from sage.runtime.event_log.events import (
    _Budget,
    ControllerAction,
    _ControllerDecision,
    _EventCore,
    _Failure,
    _FinalResult,
    _ModelAssigned,
    _NodeCompleted,
    _NodeStarted,
    QualitySource,
    _OracleVerdict,
    _RoutingDecision,
    _RunFrameSummary,
    _StateApplied,
    _TaskStarted,
    _TopologySelected,
    ThresholdBand,
)
from sage.runtime.event_log.redaction import (
    _hash_payload,
    _hash_text,
    _redact_payload,
)
from sage.runtime.event_log.schema import FINAL_RESULT_STATUSES, SCHEMA_VERSION
from sage.runtime.event_log.payload_schemas import (
    _assert_current_payload_schema_for_emit,
    _validate_payload_against_schema,
)

log = logging.getLogger(__name__)

_TRACE_DIR_ENV = "SAGE_TRACE_JSONL_DIR"
_TRACE_RAW_ENV = "SAGE_TRACE_RAW"
_FAIL_CLOSED_ENV = "SAGE_TRACE_FAIL_CLOSED"

_current_event_log: contextvars.ContextVar[RuntimeEventLog | None] = (
    contextvars.ContextVar("_current_event_log", default=None)
)

_CONTROLLER_ACTIONS = set(get_args(ControllerAction))
_QUALITY_SOURCES = set(get_args(QualitySource))
_THRESHOLD_BANDS = set(get_args(ThresholdBand))


def _coerce_controller_action(value: str) -> ControllerAction:
    if value in _CONTROLLER_ACTIONS:
        return cast(ControllerAction, value)
    return "continue"


def _coerce_quality_source(value: str | None) -> QualitySource:
    if value in _QUALITY_SOURCES:
        return cast(QualitySource, value)
    return "abstain"


def _coerce_threshold_band(value: str | None) -> ThresholdBand:
    if value in _THRESHOLD_BANDS:
        return cast(ThresholdBand, value)
    return "continue"


_REASON_CODE_RE = re.compile(r"^[a-z0-9_.:-]{1,80}$")


@dataclass(frozen=True, slots=True)
class EventRef:
    event_type: str
    seq: int
    payload_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "seq": self.seq,
            "payload_hash": self.payload_hash,
        }


def _safe_reason_code(value: str | None) -> str:
    text = (value or "").strip().lower()
    if _REASON_CODE_RE.fullmatch(text):
        return text
    return "abstain_no_signal"


def _safe_quality_score(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return max(0.0, min(1.0, score))


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

    def tell(self) -> int:
        return self._fh.tell()

    def truncate(self, size: int | None = None) -> int:
        return self._fh.truncate(size)

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
        self._last_event_ref: EventRef | None = None

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

    def emit_task_started(self, task: str) -> int | None:
        """Returns the seq of the emitted event, or None if writer is disabled."""
        self.set_task_text(task)
        return self._emit(_TaskStarted, "task_started", "pipeline", payload=task)

    def emit_routing_decision(
        self,
        *,
        routing_source: str,
        system: int,
        domain: str,
        confidence: float | None,
        model_id: str = "",
    ) -> int | None:
        payload = {
            "routing_source": routing_source,
            "system": system,
            "domain": domain,
            "confidence": round(confidence, 4) if confidence is not None else None,
            "model_id": model_id,
        }
        return self._emit(
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
    ) -> int | None:
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
        return self._emit(
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
    ) -> int | None:
        capabilities = tuple(sorted(required_capabilities))
        payload = {
            "node_id": node_id,
            "node_role": node_role,
            "model_id": model_id,
            "provider_id": provider_id,
            "required_capabilities": list(capabilities),
        }
        return self._emit(
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
    ) -> int | None:
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
        return self._emit(
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
    ) -> int | None:
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
        return self._emit(
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
    ) -> int | None:
        return self._emit(
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
        quality_score: float | None = None,
        quality_source: QualitySource | str | None = None,
        threshold_band: ThresholdBand | str | None = None,
        reason_code: str = "",
    ) -> int | None:
        safe_action = _coerce_controller_action(action)
        safe_quality_source = _coerce_quality_source(
            quality_source if isinstance(quality_source, str) else None
        )
        safe_threshold_band = _coerce_threshold_band(
            threshold_band if isinstance(threshold_band, str) else None
        )
        safe_reason_code = _safe_reason_code(reason_code)
        safe_quality_score = _safe_quality_score(quality_score)
        _ = reason  # API compatibility only; current payload schema forbids it.
        payload: dict[str, Any] = {
            "node_id": str(node_id),
            "action": safe_action,
            "target_node_id": str(target_node_id or ""),
            "gate_source_id": str(gate_source_id or ""),
            "gate_target_id": str(gate_target_id or ""),
            "quality_score": safe_quality_score,
            "quality_source": safe_quality_source,
            "threshold_band": safe_threshold_band,
            "reason_code": safe_reason_code,
        }
        return self._emit(
            _ControllerDecision,
            "controller_decision",
            "controller",
            payload=payload,
            _force_payload=True,
            node_id=node_id,
            action=safe_action,
            quality_score=safe_quality_score,
            quality_source=safe_quality_source,
            threshold_band=safe_threshold_band,
            reason_code=safe_reason_code,
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
    ) -> int | None:
        payload = {"kind": kind, "error_type": error_type, "message": message}
        return self._emit(
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
    ) -> int | None:
        payload = {
            "kind": kind,
            "budget_limit_usd": budget_limit_usd,
            "budget_remaining_usd": budget_remaining_usd,
            "cost_so_far_usd": cost_so_far_usd,
        }
        return self._emit(
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
    ) -> int | None:
        if status not in FINAL_RESULT_STATUSES:
            status = "failure"
        return self._emit(
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


    def emit_prompt_injection_detected(
        self,
        *,
        pattern_name: str,
        match_text: str,
        span_start: int,
        span_end: int,
        severity: str = "medium",
        parent_event_id: int | None = None,
    ) -> int | None:
        """Emit a prompt-injection detection event (P1-3 pipeline ingress)."""
        payload = {
            "pattern_name": pattern_name,
            "match_text": match_text[:200],
            "span_start": span_start,
            "span_end": span_end,
            "severity": severity,
        }
        return self._emit(
            _Failure,  # reuse event core; distinguished by event_type
            "prompt_injection_detected",
            "pipeline",
            payload=payload,
            kind="prompt_injection",
            error_type="detected",
            parent_event_id=parent_event_id,
        )
    

    def emit_provider_execution_witness(
        self,
        *,
        witness_schema_version: str = "v0",
        assignment_phase: str = "initial",
        routing: dict[str, Any],
        policy: dict[str, Any],
        per_node_assignments: list[dict[str, Any]],
        substitution_summary: dict[str, Any],
        parent_event_id: int | None = None,
    ) -> int | None:
        """Emit a `provider_execution_witness` event (slice 10D Route A v0).

        cgpro DESIGN_LOCK 2026-05-11: makes the chain
        ``routing_chosen_model → policy_decision → per_node_assignments``
        explicit in the event log. NOT an invariant yet — v0 witness only.

        ``_force_payload=True`` so the structured payload is visible
        regardless of ``SAGE_TRACE_RAW``. The payload contains no
        credential-shaped strings; only model IDs, provider IDs,
        capability lists, and small reason-code enums.
        """
        payload = {
            "witness_schema_version": witness_schema_version,
            "assignment_phase": assignment_phase,
            "routing": dict(routing),
            "policy": dict(policy),
            "per_node_assignments": [dict(n) for n in per_node_assignments],
            "substitution_summary": dict(substitution_summary),
        }
        return self._emit(
            _RunFrameSummary,  # reuse event core; distinguished by event_type
            "provider_execution_witness",
            "pipeline",
            payload=payload,
            parent_event_id=parent_event_id,
            _force_payload=True,
        )

    def emit_run_frame_summary(
        self,
        *,
        parent_event_id: int,
        summary: dict[str, Any],
    ) -> int | None:
        """Emit the trailing run_frame_summary diagnostic.

        ONLY called when SAGE_RUN_FRAME=1. Best-effort: callers must catch
        EventLogUnavailable and not propagate.
        """
        payload = dict(summary)
        return self._emit(
            _RunFrameSummary,
            "run_frame_summary",
            "pipeline",
            payload=payload,
            parent_event_id=parent_event_id,
            _force_payload=True,
        )

    def emit_oracle_verdict(
        self,
        *,
        parent_event_id: int | None,
        verdict: Any,
    ) -> int | None:
        """Emit the OracleStack verdict event after final_result."""
        return self._emit(
            _OracleVerdict,
            "oracle_verdict",
            "pipeline",
            payload=verdict.to_dict(),
            parent_event_id=parent_event_id,
            _force_payload=True,
        )

    def close(self) -> None:
        if self._fh is not None and not self._fh.closed:
            try:
                self._fh.close()
            except Exception:  # noqa: BLE001
                pass
        self._fh = None
        self.disabled = True

    @property
    def path(self) -> Path | None:
        """Return the JSONL file path for sidecar consumers."""
        return self._path

    def last_event_ref(self) -> EventRef | None:
        """Return the last successfully written event reference."""
        return self._last_event_ref

    def _emit(
        self,
        cls: type[_EventCore],
        event_type: str,
        source_component: str,
        *,
        payload: Any,
        _is_final: bool = False,
        _force_payload: bool = False,
        parent_event_id: int | None = None,
        **fields: Any,
    ) -> int | None:
        if self.disabled or self._fh is None:
            return None

        schema = _assert_current_payload_schema_for_emit(event_type)
        statecore_profile = self._statecore_profile_for_emit(event_type)
        try:
            _validate_payload_against_schema(
                payload,
                schema,
                top_level_event=fields,
                allow_absent_payload=False,
                statecore_profile=statecore_profile,
            )
            payload_redacted = _redact_payload(payload)
            if self._raw_payload or _force_payload:
                _validate_payload_against_schema(
                    payload_redacted,
                    schema,
                    top_level_event=fields,
                    allow_absent_payload=False,
                    statecore_profile=statecore_profile,
                )
        except EventLogSchemaError as exc:
            self._handle_schema_failure(exc, phase=event_type)
            return None

        seq = self._seq
        self._seq += 1
        parent = self._last_event_seq if parent_event_id is None else parent_event_id
        offset: int | None = None
        try:
            offset = self._fh.tell()
            payload_hash = _hash_payload(event_type, payload)
            event = cls(
                schema_version=SCHEMA_VERSION,
                payload_schema_version=schema.version,
                run_id=self.run_id,
                trace_id=self.trace_id,
                parent_event_id=parent,
                seq=seq,
                timestamp_ns=time.time_ns(),
                event_type=event_type,
                source_component=source_component,
                task_hash=self._cached_task_hash,
                payload_hash=payload_hash,
                redaction_state=self._redaction_state_for(payload),
                payload=payload_redacted if self._raw_payload or _force_payload else None,
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
            self._last_event_ref = EventRef(
                event_type=event_type,
                seq=seq,
                payload_hash=payload_hash,
            )
            return seq
        except (OSError, IOError, ValueError, TypeError) as exc:
            if offset is not None and self._fh is not None and not self._fh.closed:
                try:
                    self._fh.truncate(offset)
                    self._fh.flush()
                except Exception:  # noqa: BLE001
                    pass
            self._handle_sink_failure(exc, phase=event_type)
            return None

    def _redaction_state_for(self, payload: Any) -> str:
        if payload is None:
            return "none_applicable"
        if self._raw_payload:
            return "raw"
        return "redacted"

    def _statecore_profile_for_emit(
        self,
        event_type: str,
    ) -> Literal["on", "off"] | None:
        if event_type != "node_started":
            return None
        return "on" if os.environ.get("SAGE_STATECORE") == "1" else "off"

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

    def _handle_schema_failure(self, exc: EventLogSchemaError, *, phase: str) -> None:
        if self._fail_closed:
            self.close()
            raise exc
        log.error(
            "event_log_error phase=%s run_id=%s error_type=EventLogSchemaError "
            "message=%s",
            phase,
            self.run_id,
            exc,
        )
        self.close()
