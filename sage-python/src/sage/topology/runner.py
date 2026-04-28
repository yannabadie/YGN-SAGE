"""TopologyRunner: execute TopologyGraph as real multi-agent system.

Bridges the gap between topology IR (Rust petgraph) and agent execution.
Uses TopologyExecutor for readiness-based scheduling and spawns per-node LLM calls.

Architecture follows MASFactory (2603.06007):
- Node lifecycle: aggregate predecessor outputs → build prompt → LLM call → store output
- Readiness: node executes when TopologyExecutor marks it ready
- Context: predecessor outputs injected via TopologyGraph.get_predecessors()
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

from sage._python import PYTHON
from sage.llm.base import LLMConfig, LLMProvider, Message, Role
from sage.runtime.event_log import RuntimeEventLog
from sage.runtime.run_frame.builder import _RunFrameBuilder
from sage.runtime.state import StateApplyResult, StateDelta, StateFrame, apply_deltas
from sage.tools.sage_recurse import sage_recurse_origin_node

log = logging.getLogger(__name__)

execute_isolated: Any
try:
    from sage.sandbox.isolated_executor import BWRAP_AVAILABLE, execute_isolated

    _SANDBOX_IMPORT_OK = True
except ImportError as _import_exc:
    execute_isolated = None
    BWRAP_AVAILABLE = False
    _SANDBOX_IMPORT_OK = False
    _SANDBOX_IMPORT_ERR: ImportError | None = _import_exc
else:
    _SANDBOX_IMPORT_ERR = None

# Edge type constants (matching sage-core/src/topology/topology_graph.rs)
EDGE_CONTROL = 0
EDGE_MESSAGE = 1
EDGE_STATE = 2

# Prefix of phases/learn.py EMPTY_STEP_SENTINEL. An output starting with this
# means the upstream node exited its step budget without producing content —
# forwarding it to downstream nodes only teaches them that predecessors
# failed, then they cascade the same sentinel. We drop such outputs from
# predecessor context entirely (without fabricating a replacement prompt).
_SENTINEL_PREFIX = "[sage: agent exited after"
_BUDGET_EXCEEDED_RESULT = "[sage: budget exceeded]"


@dataclass(frozen=True)
class _NodeStartEvent:
    node_idx: int
    role: str


@dataclass(frozen=True)
class _NodeDoneEvent:
    node_idx: int
    role: str
    output: str
    latency_ms: float
    model_id: str


@dataclass(frozen=True)
class _ControllerDecisionEvent:
    node_idx: int
    action: str
    reason: str = ""


@dataclass(frozen=True)
class _RerouteEvent:
    reason: str = ""


@dataclass(frozen=True)
class _BudgetExceededEvent:
    reason: str = "budget exceeded"


@dataclass(frozen=True)
class _TopologyDoneEvent:
    final_output: str
    node_count: int


_RunEvent = (
    _NodeStartEvent
    | _NodeDoneEvent
    | _ControllerDecisionEvent
    | _RerouteEvent
    | _BudgetExceededEvent
    | _TopologyDoneEvent
)

# Roles recognized as "planner" for the optional planner-output injection
# experiment. Kept in sync with topology/role_prompts.py _PLANNER aliases.
_PLANNER_ROLE_KEYWORDS = ("planner", "input_processor", "decomposer")

# Max characters of planner output injected into downstream system prompt.
# Keeps the prompt bounded regardless of how verbose the planner is.
_PLANNER_INJECTION_BUDGET = 2000


def _is_sentinel(output: str) -> bool:
    """True if output is the EMPTY_STEP_SENTINEL string from phases/learn.py."""
    return isinstance(output, str) and output.strip().startswith(_SENTINEL_PREFIX)


def _is_planner_role(role: str) -> bool:
    """True if a node role is a planner/decomposer variant."""
    if not isinstance(role, str):
        return False
    rl = role.lower()
    return any(kw in rl for kw in _PLANNER_ROLE_KEYWORDS)


def _edge_type_name(edge_type: Any) -> str:
    if edge_type == EDGE_CONTROL:
        return "control"
    if edge_type == EDGE_MESSAGE:
        return "message"
    if edge_type == EDGE_STATE:
        return "state"
    return str(edge_type).strip().lower()


def _partition_incoming_edges(
    graph: Any,
    node_idx: int,
    *,
    legacy_mode: bool,
) -> tuple[list[int], list[int], list[int]]:
    """Return incoming (control_preds, message_preds, state_preds) via get_edges()."""
    try:
        edges = graph.get_edges()
    except AttributeError as exc:
        if legacy_mode:
            log.warning(
                "topology.runner: graph.get_edges() unavailable in legacy mode; "
                "falling back to get_predecessors() for message context",
            )
            try:
                return [], list(graph.get_predecessors(node_idx)), []
            except (AttributeError, Exception):
                return [], [], []
        raise ValueError("StateCore requires graph.get_edges() for edge partitioning") from exc

    control_preds: list[int] = []
    message_preds: list[int] = []
    state_preds: list[int] = []
    for edge in edges:
        try:
            src, dst, edge_type = edge
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid edge tuple for node {node_idx}: {edge!r}") from exc
        if int(dst) != node_idx:
            continue

        edge_type_name = _edge_type_name(edge_type)
        src_idx = int(src)
        if edge_type_name == "control":
            control_preds.append(src_idx)
            if legacy_mode:
                message_preds.append(src_idx)
        elif edge_type_name == "message":
            message_preds.append(src_idx)
        elif edge_type_name == "state":
            state_preds.append(src_idx)
            if legacy_mode:
                message_preds.append(src_idx)
        elif legacy_mode:
            log.warning(
                "topology.runner: unknown edge type %r for %s->%s in legacy mode; "
                "treating as message context",
                edge_type,
                src,
                dst,
            )
            message_preds.append(src_idx)
        else:
            raise ValueError(
                f"unknown edge type {edge_type!r} for incoming edge {src}->{dst}"
            )
    return control_preds, message_preds, state_preds


class TopologyRunner:
    """Execute a TopologyGraph as a real multi-agent system.

    Single-use: each instance runs one topology execution. Do not call
    ``run()`` more than once (``_node_outputs`` is not reset between runs).

    Parameters
    ----------
    graph : TopologyGraph or compatible stub
        Must have ``node_count()``, ``get_node(idx)``.
    executor : TopologyExecutor or compatible stub
        Must have ``next_ready(graph)``, ``mark_completed(idx)``, ``is_done()``.
    llm_provider : LLMProvider
        The LLM provider for generating responses per node.
    llm_config : LLMConfig, optional
        Optional LLMConfig override.
    controller : TopologyController, optional
        Runtime adaptation controller (Phase C). If None, behaves as Phase B
        (no adaptation). When provided, ``evaluate_and_decide()`` is called
        after each node to trigger upgrade_model, spawn_subagent, reroute or
        prune actions.
    """

    def __init__(
        self,
        graph: Any,
        executor: Any,
        llm_provider: LLMProvider,
        llm_config: LLMConfig | None = None,
        *,
        provider_pool: Any | None = None,
        controller: Any | None = None,
        axis_hint: str = "",
        approval_callback: Callable | None = None,
        harness_config: Any | None = None,
        agent_loop_factory: Any | None = None,
        cost_tracker: Any | None = None,
        assigner: Any | None = None,
        task_domain: str = "",
        budget_usd: float = 0.0,
        event_log: RuntimeEventLog | None = None,
        run_frame_builder: _RunFrameBuilder | None = None,
    ) -> None:
        self.graph = graph
        self.executor = executor
        self._llm = llm_provider
        self._config = llm_config
        self._provider_pool = provider_pool
        self._controller = controller
        self._axis_hint = axis_hint
        self._approval_callback = approval_callback  # HITL: async fn(decision) -> bool
        self._node_outputs: dict[int, str] = {}
        self._node_exec_count: dict[int, int] = {}  # track per-node execution count
        # Aggregate tool-use telemetry summed across all per-node AgentLoops
        # executed by this runner. Surfaced to pipeline ctx so bench manifests
        # reflect real multi-agent behavior, not per-node zeros.
        self.tool_call_count: int = 0
        self.tool_turn_count: int = 0
        self.executed_commands: list[str] = []
        # Per-run cost aggregate. Before Apr 18 2026 per-node loops each had
        # their own total_cost_usd but nothing rolled them up, so benches read
        # system.agent_loop.total_cost_usd (the single top-level loop) and
        # saw 0 whenever topology ran in multi-node mode. Aggregating here
        # matches the tool_call_count pattern above and feeds the pipeline ctx.
        self.total_cost_usd: float = 0.0
        self._node_costs: dict[int, float] = {}
        self._cost_tracker = cost_tracker
        self._assigner = assigner
        self._task_domain = task_domain
        self._budget_usd = float(budget_usd or 0.0)
        self._event_log = event_log
        self._run_frame_builder = run_frame_builder
        self._run_frame_node_runs: dict[int, str] = {}
        self._node_event_costs: dict[int, float] = {}
        self._statecore_enabled = os.environ.get("SAGE_STATECORE") == "1"
        self._node_state_deltas: dict[int, StateDelta] = {}
        self._node_state_frames: dict[int, StateFrame] = {}
        self._node_state_apply_results: dict[int, StateApplyResult] = {}

        # Meta-Harness (arXiv 2603.28052): optional harness config overlay.
        # Loaded from config/harness.json at boot. Overrides context budget,
        # predecessor format, similarity threshold, system prompt templates,
        # and debate rounds — WITHOUT replacing any methods.
        self._harness = harness_config
        self._agent_loop_factory = agent_loop_factory
        self._max_rounds = (
            harness_config.execution.max_debate_rounds
            if harness_config else 3
        )

    def _is_over_budget(self) -> bool:
        return bool(self._cost_tracker is not None and self._cost_tracker.is_over_budget)

    def _record_spend_for_node(self, node_idx: int) -> None:
        if self._cost_tracker is None:
            return
        node_cost = float(self._node_costs.pop(node_idx, 0.0) or 0.0)
        self._node_event_costs[node_idx] = node_cost
        self._cost_tracker.record_spend(node_cost)

    def _remaining_budget_usd(self) -> float:
        """Available budget for the current run, in USD.

        Returns the conservative MIN of:
        - cost_tracker.remaining (task-level budget authority — exposes inf
          for unlimited, finite values that hit 0.0 on exhaustion)
        - budget_usd - total_cost_usd (runner-internal accounting)

        Returns float("inf") only when neither bound is configured.

        cost_tracker.remaining == 0.0 is the correct fail-closed signal,
        not "unknown" — assign_single_node treats budget_usd as a hard
        filter and will refuse paid models when 0.0 is passed. That is the
        intended behavior on a fully-spent budget. Never coerce 0.0 to inf.

        Never pass 0.0 as "unlimited" to assign_single_node; it treats 0.0
        as a hard filter that rejects every paid model. Unbounded mode uses
        float("inf").
        """
        tracker_remaining: float | None = None
        if self._cost_tracker is not None and hasattr(self._cost_tracker, "remaining"):
            try:
                tracker_remaining = float(self._cost_tracker.remaining)
            except (AttributeError, TypeError, ValueError):
                tracker_remaining = None

        internal_remaining = float("inf")
        if self._budget_usd > 0:
            internal_remaining = max(self._budget_usd - self.total_cost_usd, 0.0)

        if tracker_remaining is not None:
            return min(tracker_remaining, internal_remaining)
        return internal_remaining

    @staticmethod
    def _response_cost_usd(response: Any) -> float:
        usage = getattr(response, "usage", None) or {}
        if isinstance(usage, dict):
            raw_cost = usage.get("cost_usd", 0.0)
        else:
            raw_cost = getattr(usage, "cost_usd", 0.0)
        try:
            return max(0.0, float(raw_cost or 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _context_budget_per_predecessor(self, n_predecessors: int, node_idx: int = 0) -> int:
        """Compute per-predecessor character budget based on model context window.

        Uses the receiving node's model context_window (from ModelCard, e.g.
        128K for GPT-5.4, 1M for Gemini), NOT config.max_tokens which is the
        output token limit. Reserves 30% for system prompt + task text.

        Based on TalkHier (arXiv 2502.11098): structured communication carrying
        full intermediate outputs improves accuracy over truncated handoffs.
        """
        context_window = 131072  # safe default (128K tokens)
        # Try to read real context_window from the node's assigned model
        try:
            node = self.graph.get_node(node_idx)
            model_id = getattr(node, "model_id", "")
            if model_id and self._provider_pool:
                _, resolved_config = self._provider_pool.resolve(model_id)
                cw = getattr(resolved_config, "context_window", 0)
                if cw and cw > 0:
                    context_window = cw
        except (AttributeError, RuntimeError):
            pass  # graph or pool unavailable — use default
        # 70% of context for predecessor outputs, ~4 chars per token
        _budget_ratio = self._harness.context.budget_ratio if self._harness else 0.7
        _chars_per_token = self._harness.context.chars_per_token if self._harness else 4
        _floor = self._harness.context.budget_floor_chars if self._harness else 1000
        available_chars = int(context_window * _budget_ratio * _chars_per_token)
        budget = available_chars // max(n_predecessors, 1)
        return max(budget, _floor)  # floor at 1000 chars

    def _truncate_output(self, output: str, budget: int) -> str:
        """Truncate output to budget, appending '...' if cut."""
        if len(output) <= budget:
            return output
        return output[:budget] + "..."

    def _format_predecessor_context(
        self,
        node_idx: int,
        predecessor_indices: list[int],
    ) -> str:
        budget = self._context_budget_per_predecessor(len(predecessor_indices), node_idx)

        parts_with_roles: list[tuple[str, str]] = []
        for idx in predecessor_indices:
            output = self._node_outputs.get(idx)
            if not output:
                continue
            # Drop EMPTY_STEP_SENTINEL — forwarding "agent exited after N steps
            # with no content" to a downstream synthesizer teaches it that its
            # inputs failed, and it replies with the same sentinel (cascade
            # observed on smoke v4: 5/10 tasks ended with SENTINEL patches).
            # We strip at the source; no fabricated replacement — if all
            # predecessors are sentinels, the downstream node sees empty
            # context and falls back to the task prompt alone, which is
            # strictly better than being told "everyone failed".
            if _is_sentinel(output):
                log.info(
                    "topology.runner: dropped sentinel output from predecessor %d (role=%s)",
                    idx,
                    getattr(self.graph.get_node(idx), "role", f"node-{idx}"),
                )
                continue
            node = self.graph.get_node(idx)
            role = getattr(node, "role", f"node-{idx}")
            truncated = self._truncate_output(output, budget)
            parts_with_roles.append((truncated, role))

        # Similarity gate: deduplicate near-identical predecessor outputs
        # Saves tokens when parallel workers produce similar answers (S2-MAD)
        _sim_threshold = self._harness.context.similarity_threshold if self._harness else 0.90
        deduplicated = self._deduplicate_context(parts_with_roles, _sim_threshold)

        # Format with harness template if available
        _fmt = self._harness.context.predecessor_format if self._harness else "[{role}]: {text}"
        _sep = self._harness.context.predecessor_separator if self._harness else "\n\n"
        formatted = _sep.join(
            _fmt.format(role=role, text=text, node_idx=0, model_id="")
            for text, role in deduplicated
        )

        # Wrap with injection template if harness provides one
        if self._harness and formatted:
            return self._harness.context.injection_template.format(
                context=formatted,
                n_predecessors=len(predecessor_indices),
                task_preview="",
            )
        if formatted:
            return formatted
        # All predecessors were sentinels or empty (dropped by _is_sentinel
        # above). Returning "" would leave the downstream node with only
        # its own task prompt — it tool-explores from nothing and burns
        # its step budget. Instead, emit an explicit cold-start note so
        # the agent knows to short-circuit to a direct attempt rather
        # than chasing context that isn't coming. Fix for the sentinel
        # cascade observed on astropy-14995 (docs/audits/2026-04-18-*).
        if predecessor_indices:
            return (
                "[system]: upstream nodes did not produce usable output "
                "(all predecessors were step-budget sentinels). Work "
                "directly from the task description; do not wait for "
                "upstream context."
            )
        return ""

    def _gather_predecessor_context(self, node_idx: int) -> str:
        """Collect outputs from direct predecessors of node_idx only.

        Uses Rust TopologyGraph.get_predecessors() for correct DAG traversal.
        Falls back to all completed nodes if get_predecessors unavailable.
        Deduplicates near-identical outputs (S2-MAD arXiv 2502.04790).
        """
        try:
            predecessor_indices = self.graph.get_predecessors(node_idx)
        except (AttributeError, Exception):
            return self._gather_all_context()
        return self._format_predecessor_context(node_idx, list(predecessor_indices))

    def _partition_incoming_edges(self, node_idx: int) -> tuple[list[int], list[int], list[int]]:
        legacy_mode = not self._statecore_enabled
        try:
            return _partition_incoming_edges(
                self.graph,
                node_idx,
                legacy_mode=legacy_mode,
            )
        except ValueError as exc:
            if self._statecore_enabled and self._event_log is not None:
                self._event_log.emit_failure(
                    kind="statecore_edge_type",
                    error_type=type(exc).__name__,
                    message=str(exc),
                    node_id=str(node_idx),
                )
            raise

    def _predecessors_by_channel(self, node_idx: int) -> dict[str, tuple[str, ...]]:
        control_preds, message_preds, state_preds = self._partition_incoming_edges(node_idx)
        return {
            "control": tuple(str(idx) for idx in control_preds),
            "message": tuple(str(idx) for idx in message_preds),
            "state": tuple(str(idx) for idx in state_preds),
        }

    def _state_task_id(self) -> str:
        if self._event_log is not None:
            return self._event_log.run_id
        return self._runtime_topology_id() or "topology"

    def _base_state_frame_for(self, state_preds: list[int]) -> StateFrame:
        for pred_idx in sorted(state_preds, key=lambda value: str(value)):
            frame = self._node_state_frames.get(pred_idx)
            if frame is not None:
                return frame
        return StateFrame(task_id=self._state_task_id())

    def _emit_state_applied(
        self,
        *,
        node_idx: int,
        state_preds: list[int],
        result: StateApplyResult,
    ) -> None:
        seq = None
        if self._event_log is not None:
            seq = self._event_log.emit_state_applied(
                target_node_id=str(node_idx),
                source_node_ids=tuple(str(idx) for idx in state_preds),
                before_version=result.before_version,
                after_version=result.after.version,
                delta_count=len(state_preds),
                conflict_count=len(result.conflicts),
                applied=result.applied,
                invalidated_assumption_ids=tuple(result.after.invalidated_assumptions),
            )
        if self._run_frame_builder is not None:
            state_delta = (
                self._node_state_deltas.get(state_preds[0], StateDelta())
                if len(state_preds) == 1
                else StateDelta()
            )
            self._run_frame_builder.record_state_applied(
                seq=seq,
                target_node_id=str(node_idx),
                before_version=result.before_version,
                after_version=result.after.version,
                state_delta=state_delta,
                state_frame=result.after,
                source_node_ids=tuple(str(idx) for idx in state_preds),
                delta_count=len(state_preds),
                conflict_count=len(result.conflicts),
                applied=result.applied,
            )

    def _assemble_node_inputs(self, node_idx: int) -> tuple[str, StateFrame | None, bool]:
        """Return message context, state frame, and control readiness for a node."""
        if not self._statecore_enabled:
            return self._gather_predecessor_context(node_idx), None, True

        _control_preds, message_preds, state_preds = self._partition_incoming_edges(node_idx)
        message_text = self._format_predecessor_context(node_idx, message_preds)
        state_frame: StateFrame | None = None

        if state_preds:
            base_frame = self._base_state_frame_for(state_preds)
            deltas = tuple(
                (str(src), self._node_state_deltas.get(src, StateDelta()))
                for src in state_preds
            )
            result = apply_deltas(base_frame, deltas)
            self._node_state_apply_results[node_idx] = result
            self._node_state_frames[node_idx] = result.after
            self._emit_state_applied(
                node_idx=node_idx,
                state_preds=state_preds,
                result=result,
            )
            state_frame = result.after
        else:
            self._node_state_apply_results.pop(node_idx, None)

        return message_text, state_frame, True

    def _render_state_frame(self, state_frame: StateFrame | None) -> str:
        if state_frame is None:
            return ""
        import json

        lines = [
            "StateCore frame:",
            f"task_id: {state_frame.task_id}",
            f"version: {state_frame.version}",
            f"confidence: {state_frame.confidence}",
        ]
        if state_frame.objective:
            lines.append(f"objective: {state_frame.objective}")
        if state_frame.constraints:
            lines.append("constraints:")
            lines.extend(f"- {value}" for value in state_frame.constraints)
        if state_frame.assumptions:
            lines.append("assumptions:")
            lines.extend(f"- {value}" for value in state_frame.assumptions)
        if state_frame.invalidated_assumptions:
            lines.append("invalidated_assumptions:")
            lines.extend(f"- {value}" for value in state_frame.invalidated_assumptions)
        if state_frame.entities:
            lines.append("entities:")
            for entity_id, fields in sorted(state_frame.entities.items()):
                payload = json.dumps(dict(fields), sort_keys=True, default=str)
                lines.append(f"- {entity_id}: {payload}")
        if state_frame.decisions:
            lines.append("decisions:")
            for decision in state_frame.decisions:
                lines.append(f"- {json.dumps(dict(decision), sort_keys=True, default=str)}")
        if state_frame.tool_facts:
            lines.append("tool_facts:")
            for fact in state_frame.tool_facts:
                lines.append(f"- {json.dumps(dict(fact), sort_keys=True, default=str)}")
        if state_frame.open_questions:
            lines.append("open_questions:")
            lines.extend(f"- {value}" for value in state_frame.open_questions)
        return "\n".join(lines)

    def _prepare_statecore_node(
        self,
        node_idx: int,
        open_nodes: dict[int, str],
    ) -> tuple[str, StateFrame | None, bool]:
        message_text, state_frame, control_ready = self._assemble_node_inputs(node_idx)
        apply_result = self._node_state_apply_results.get(node_idx)
        if apply_result is not None and not apply_result.applied:
            self._runtime_emit_failure(
                kind="state_conflict",
                error_type="StateConflict",
                message="; ".join(apply_result.conflicts),
                open_nodes=open_nodes,
                node_idx=node_idx,
            )
            try:
                self.executor.mark_skipped(node_idx)
            except (AttributeError, Exception):
                pass
            try:
                self.executor.mark_completed(node_idx)
            except (AttributeError, Exception):
                pass
            return message_text, state_frame, False
        return message_text, state_frame, control_ready

    def _maybe_planner_injection(self, node_idx: int, system_prompt: str) -> str:
        """Optionally prepend upstream planner output to this node's system_prompt.

        Default ON since 2026-04-18 (audit docs/audits/2026-04-18-astropy-14995).
        Set SAGE_PLANNER_INJECTION=0 to disable. MASS (arXiv 2502.02533): the
        structured decomposition plan is higher-signal for downstream nodes
        than the raw predecessor context mixed with other outputs. Still
        emitted via predecessor context too — this only adds explicit
        section at the top of the system prompt for nodes downstream of a
        planner.

        The function is self-gated: it only injects if a planner exists
        among predecessors AND that planner produced non-sentinel output,
        so the default-ON flip is scoped to sequential-like topologies.

        No-op if:
        - Flag is set to "0"
        - Current node IS a planner (skip self-injection)
        - No planner found among predecessors
        - Planner output is a sentinel
        """
        if os.environ.get("SAGE_PLANNER_INJECTION", "1") == "0":
            return system_prompt

        current = self.graph.get_node(node_idx)
        current_role = getattr(current, "role", "")
        if _is_planner_role(current_role):
            return system_prompt

        if self._statecore_enabled:
            _control_preds, predecessors, _state_preds = self._partition_incoming_edges(node_idx)
        else:
            try:
                predecessors = self.graph.get_predecessors(node_idx)
            except (AttributeError, Exception):
                return system_prompt

        for pred_idx in predecessors:
            pred_node = self.graph.get_node(pred_idx)
            pred_role = getattr(pred_node, "role", "")
            if not _is_planner_role(pred_role):
                continue
            pred_output = self._node_outputs.get(pred_idx, "")
            if not pred_output or _is_sentinel(pred_output):
                continue
            # Bound the injection; planner output can be long
            truncated = pred_output[:_PLANNER_INJECTION_BUDGET]
            if len(pred_output) > _PLANNER_INJECTION_BUDGET:
                truncated += "\n... [truncated]"
            log.info(
                "topology.runner: injecting planner output (%d chars, role=%s) into node %d system prompt",
                len(truncated), pred_role, node_idx,
            )
            return (
                f"## Upstream plan (from {pred_role}):\n{truncated}\n\n"
                f"## Your role\n{system_prompt}"
            )
        return system_prompt

    def _gather_all_context(self) -> str:
        """Fallback: all completed nodes (legacy behavior)."""
        n_completed = len(self._node_outputs)
        budget = self._context_budget_per_predecessor(n_completed)

        parts_with_roles: list[tuple[str, str]] = []
        for idx in sorted(self._node_outputs.keys()):
            output = self._node_outputs[idx]
            if not output:
                continue
            # Same sentinel-strip logic as _gather_predecessor_context above.
            if _is_sentinel(output):
                continue
            node = self.graph.get_node(idx)
            role = getattr(node, "role", f"node-{idx}")
            truncated = self._truncate_output(output, budget)
            parts_with_roles.append((truncated, role))

        deduplicated = self._deduplicate_context(parts_with_roles)
        if deduplicated:
            return "\n\n".join(f"[{role}]: {text}" for text, role in deduplicated)
        # All completed outputs were sentinels (see D2 fix in
        # _gather_predecessor_context above). Same cold-start note.
        if self._node_outputs:
            return (
                "[system]: upstream nodes did not produce usable output "
                "(all predecessors were step-budget sentinels). Work "
                "directly from the task description; do not wait for "
                "upstream context."
            )
        return ""

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two embedding vectors (pure Python)."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a < 1e-15 or norm_b < 1e-15:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _deduplicate_jaccard(
        parts: list[tuple[str, str]],
        threshold: float = 0.85,
    ) -> list[tuple[str, str]]:
        """Fallback: Jaccard word dedup when embeddings unavailable."""
        if len(parts) <= 1:
            return parts
        deduplicated: list[tuple[str, str]] = []
        for text_i, role_i in parts:
            words_i = set(text_i.lower().split())
            is_duplicate = False
            for j, (text_j, _) in enumerate(deduplicated):
                words_j = set(text_j.lower().split())
                if words_i and words_j:
                    jaccard = len(words_i & words_j) / len(words_i | words_j)
                    if jaccard > threshold:
                        if len(text_i) < len(text_j):
                            deduplicated[j] = (text_i, role_i)
                        is_duplicate = True
                        break
            if not is_duplicate:
                deduplicated.append((text_i, role_i))
        return deduplicated

    @classmethod
    def _deduplicate_context(
        cls,
        parts: list[tuple[str, str]],
        threshold: float = 0.90,
    ) -> list[tuple[str, str]]:
        """Remove near-duplicate predecessor outputs via semantic similarity.

        Uses cosine similarity on arctic-embed-m embeddings (768-dim) when
        available. Falls back to Jaccard word similarity otherwise.

        Tie-breaker: keep **shortest** (penalize verbosity, not reward it).

        Based on S2-MAD (arXiv 2502.04790): -94% tokens, <2% perf loss.
        """
        if len(parts) <= 1:
            return parts

        # Try semantic similarity (cosine on embeddings)
        try:
            from sage.memory.embedder import Embedder
            emb = Embedder()
            if not emb.is_semantic:
                return cls._deduplicate_jaccard(parts)
            texts = [t for t, _ in parts]
            vectors = emb.embed_batch(texts)
        except (ImportError, RuntimeError, AttributeError):
            return cls._deduplicate_jaccard(parts)

        deduplicated: list[tuple[str, str, list[float]]] = []
        for (text_i, role_i), vec_i in zip(parts, vectors):
            is_duplicate = False
            for j, (text_j, _, vec_j) in enumerate(deduplicated):
                sim = cls._cosine_sim(vec_i, vec_j)
                if sim > threshold:
                    # Keep the SHORTER one (penalize verbosity)
                    if len(text_i) < len(text_j):
                        deduplicated[j] = (text_i, role_i, vec_i)
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduplicated.append((text_i, role_i, vec_i))

        return [(t, r) for t, r, _ in deduplicated]

    async def _execute_node_via_agent_loop(
        self,
        node_idx: int,
        task: str,
        context_override: str | None = None,
        state_frame: StateFrame | None = None,
    ) -> str:
        """Execute an LLM node via per-node AgentLoop (Phase 2).

        Creates an independent AgentLoop instance for this node with:
        - Role-filtered tools (H6)
        - Validation level from system classification (H6)
        - Skip routing (H1) and topology (H4) flags
        - Predecessor context in user message (H7)
        """
        node = self.graph.get_node(node_idx)
        role = getattr(node, "role", f"node-{node_idx}")
        caps = getattr(node, "required_capabilities", [])

        # Build system prompt (same logic as _execute_node)
        custom_prompt = getattr(node, "prompt", "")
        if custom_prompt:
            system_prompt = custom_prompt
        else:
            # New (2026-04-17): try the per-role prompt registry first. The
            # Rust template factories (sequential / parallel / robust) build
            # nodes with empty `prompt` fields for planner/coder/worker, so
            # the previous fallback — "You are acting as: {role}." — was
            # what every non-synthesizer agent actually saw. Smoke evidence:
            # docs/benchmarks/2026-04-17-swebench-smoke-debug.md.
            from sage.topology.role_prompts import get_role_prompt
            _role_prompt = get_role_prompt(role)
            if _role_prompt:
                system_prompt = _role_prompt
            else:
                _default_tmpl = (
                    self._harness.prompts.default_template if self._harness
                    else "You are acting as: {role}."
                )
                system_prompt = _default_tmpl.format(
                    role=role, capabilities=", ".join(caps) if caps else "",
                    task_preview=task[:200], n_predecessors=0,
                )
            if caps:
                _cap_tmpl = (
                    self._harness.prompts.capability_template if self._harness
                    else " Your capabilities: {capabilities}."
                )
                system_prompt += _cap_tmpl.format(capabilities=", ".join(caps))

        if self._harness:
            if self._harness.prompts.global_prefix:
                system_prompt = self._harness.prompts.global_prefix + "\n" + system_prompt
            if self._harness.prompts.global_suffix:
                system_prompt = system_prompt + "\n" + self._harness.prompts.global_suffix

        # Optional: prepend upstream planner output (MASS arXiv 2502.02533).
        # Gated by SAGE_PLANNER_INJECTION=1 — off by default so the base
        # behavior is unchanged.
        system_prompt = self._maybe_planner_injection(node_idx, system_prompt)

        # Resolve per-node model
        node_model_id = getattr(node, "model_id", "")
        if node_model_id and self._provider_pool:
            provider, config = self._provider_pool.resolve(node_model_id)
        else:
            provider, config = self._llm, self._config

        # Create per-node AgentLoop (H8: independent instance).
        # D1 fix (2026-04-18 audit): pass the node's own system tier,
        # not the outer task system. Sequential template declares
        # system=1/2/1 on nodes (templates.rs:36,44,54), but the
        # partial created in pipeline.py bound system_level=ctx.system
        # which is the outer task tier. That pushed S1 planner/synthesizer
        # into a 20-step S3 budget, which explains the 20-tool-call
        # sentinel on astropy-14995. Partial kwargs are overridable.
        _factory_kwargs: dict[str, Any] = dict(
            node_role=role,
            node_name=f"node-{node_idx}-{role}",
            llm_provider=provider,
            llm_config=config,
            system_prompt=system_prompt,
        )
        _node_system = int(getattr(node, "system", 0) or 0)
        if _node_system > 0:
            _factory_kwargs["system_level"] = _node_system
        # D6 audit fix (2026-04-18): forward drift classifications from
        # DriftMonitor (monitoring/drift.py) to ProviderPool so a
        # SWITCH_MODEL-worthy drift signal actually trips the circuit
        # breaker for the offending provider. Before this wiring the
        # drift action was log-only — classifications had zero effect
        # on subsequent model resolutions.
        _pool_for_drift = self._provider_pool
        if _pool_for_drift is not None and hasattr(_pool_for_drift, "record_failure"):
            _node_model_id = getattr(node, "model_id", "") or "default"
            def _on_drift(
                provider_hint: str,
                action: str,
                details: dict[str, Any],
                _pool: Any = _pool_for_drift,
                _model: str = _node_model_id,
            ) -> None:
                if action not in ("SWITCH_MODEL", "RESET_AGENT"):
                    return
                _key = (provider_hint or _model or "unknown")
                try:
                    _pool.record_failure(
                        _key,
                        RuntimeError(
                            f"drift_{action.lower()} score={details.get('latency', '?')}"
                        ),
                    )
                except Exception:  # noqa: BLE001
                    pass
            _factory_kwargs["on_drift"] = _on_drift
        assert self._agent_loop_factory is not None, (
            "TopologyRunner requires agent_loop_factory at construction"
        )
        loop = self._agent_loop_factory(**_factory_kwargs)

        # Build task with predecessor context (H7)
        if context_override is not None:
            context = context_override
        elif self._statecore_enabled:
            context, state_frame, _control_ready = self._assemble_node_inputs(node_idx)
        else:
            context = self._gather_predecessor_context(node_idx)
        state_block = self._render_state_frame(state_frame)
        sections: list[str] = []
        if context:
            sections.append(f"## Previous agent output:\n{context}")
        if state_block:
            sections.append(f"## StateCore frame:\n{state_block}")
        sections.append(f"## Task:\n{task}" if sections else task)
        full_task = "\n\n".join(sections)

        # Execute. Before Apr 18 2026 the agent-loop path had no provider
        # circuit-breaker wiring: if the per-node loop raised because of
        # a rate-limit or timeout, the exception propagated untouched and
        # `_provider_pool` never learned the provider was sick, so the
        # next node on the same provider just hit the same wall. The
        # direct _execute_node path already records failure/success on
        # `_provider_pool` — this block mirrors it for AgentLoop. P1.2
        # of the 2026-04-18 mega-plan.
        provider_name = getattr(config, "provider", "unknown")
        from sage.observability.spans import sage_span
        _node_name = (
            getattr(node, "name", None)
            or getattr(node, "role", None)
            or f"node_{node_idx}"
        )
        _node_span_attrs: dict[str, Any] = {
            "sage.node.name": _node_name,
            "sage.node.index": node_idx,
        }
        try:
            with sage_span(
                f"sage.node.{_node_name}",
                op="invoke_agent",
                **_node_span_attrs,
            ):
                result = await loop.run(full_task)
        except (RuntimeError, TimeoutError, asyncio.TimeoutError, ConnectionError) as exc:
            if self._provider_pool and hasattr(self._provider_pool, "record_failure"):
                try:
                    self._provider_pool.record_failure(provider_name, exc)
                except Exception:  # noqa: BLE001 — never let telemetry mask the real error
                    pass
            raise
        else:
            if self._provider_pool and hasattr(self._provider_pool, "record_success"):
                try:
                    self._provider_pool.record_success(provider_name)
                except Exception:  # noqa: BLE001
                    pass
        self._node_outputs[node_idx] = result
        # Aggregate tool-use telemetry — per-node counters are local to each
        # AgentLoop; without this rollup the pipeline ctx sees zero even
        # when nodes did call tools.
        self.tool_call_count += int(getattr(loop, "tool_call_count", 0) or 0)
        self.tool_turn_count += int(getattr(loop, "tool_turn_count", 0) or 0)
        node_cost = float(getattr(loop, "total_cost_usd", 0.0) or 0.0)
        self.total_cost_usd += node_cost
        self._node_costs[node_idx] = self._node_costs.get(node_idx, 0.0) + node_cost
        node_commands = list(getattr(loop, "executed_commands", []) or [])
        if node_commands:
            self.executed_commands.extend(f"[{role}] {c}" for c in node_commands)
        log.info(
            "[TopologyRunner] node %d (%s) completed via agent_loop, output %d chars, tool_calls=%d",
            node_idx, role, len(result), int(getattr(loop, "tool_call_count", 0) or 0),
        )

        return result

    async def _execute_code_node(
        self,
        node_idx: int,
        task: str,
        context_override: str | None = None,
        state_frame: StateFrame | None = None,
    ) -> str:
        """Execute a code node in sandbox (HyEvo v^Code deterministic execution).

        Code nodes run synthesized Python in a restricted sandbox instead of
        calling an LLM. This offloads deterministic work (validation, parsing,
        computation) from expensive LLM inference.
        """
        import json
        import time

        node = self.graph.get_node(node_idx)
        role = getattr(node, "role", f"node-{node_idx}")
        code_spec = getattr(node, "code_spec", "") or getattr(node, "prompt", "")

        if not code_spec:
            log.error("Code node %d (%s) has no code_spec", node_idx, role)
            return f"ERROR: code node {node_idx} has no code_spec"

        if context_override is not None:
            context = context_override
        elif self._statecore_enabled:
            context, state_frame, _control_ready = self._assemble_node_inputs(node_idx)
        else:
            context = self._gather_predecessor_context(node_idx)

        t0 = time.monotonic()

        # Build a self-contained script that receives task+context via globals
        wrapped_code = (
            f"_TASK = {json.dumps(task[:2000])}\n"
            f"_CONTEXT = {json.dumps(context[:5000])}\n"
            f"_STATECORE = {json.dumps(self._render_state_frame(state_frame)[:5000])}\n"
            f"{code_spec}\n"
        )

        unsafe_raw_exec = os.environ.get("SAGE_UNSAFE_RAW_EXEC") == "1"
        sandbox_isolated_available = _SANDBOX_IMPORT_OK and BWRAP_AVAILABLE

        if sandbox_isolated_available:
            stdout, stderr, exit_code = execute_isolated(wrapped_code, timeout=30)
        elif unsafe_raw_exec:
            log.warning(
                "[TopologyRunner] _execute_code_node falling back to raw subprocess "
                "(SAGE_UNSAFE_RAW_EXEC=1; isolated_executor %s). "
                "DO NOT USE IN PRODUCTION.",
                "missing"
                if not _SANDBOX_IMPORT_OK
                else "imported but BWRAP_AVAILABLE=False",
            )
            try:
                proc = subprocess.run(
                    [PYTHON, "-c", wrapped_code],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                stdout = proc.stdout
                stderr = proc.stderr
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                stdout = ""
                stderr = "TIMEOUT"
                exit_code = -1
            except (OSError, ValueError) as exc:
                stdout = ""
                stderr = str(exc)
                exit_code = -1
        else:
            from sage.sandbox.errors import SandboxUnavailable

            if _SANDBOX_IMPORT_ERR is not None:
                raise SandboxUnavailable(
                    "sage.sandbox.isolated_executor unavailable; refusing to execute "
                    "topology code node without sandbox. Set SAGE_UNSAFE_RAW_EXEC=1 "
                    "only for local/dev unsafe raw execution."
                ) from _SANDBOX_IMPORT_ERR
            raise SandboxUnavailable(
                "sage.sandbox.isolated_executor imported but BWRAP_AVAILABLE=False "
                "(no bwrap binary on this platform). Refusing to execute topology "
                "code node without isolation. Set SAGE_UNSAFE_RAW_EXEC=1 only for "
                "local/dev unsafe raw execution."
            )

        output = stdout

        latency_ms = (time.monotonic() - t0) * 1000

        if stderr and exit_code != 0:
            log.warning(
                "Code node %d (%s) failed (exit=%d, %.0fms): %s",
                node_idx, role, exit_code, latency_ms, stderr[:200],
            )
        else:
            log.info(
                "Code node %d (%s) completed (%.0fms, %d chars output)",
                node_idx, role, latency_ms, len(output),
            )

        self._node_outputs[node_idx] = output
        return output

    async def _execute_solver_node(
        self,
        node_idx: int,
        task: str,
        context_override: str | None = None,
        state_frame: StateFrame | None = None,
    ) -> str:
        """Execute a formal solver node — try Rust solver, fall back to LLM.

        Hybrid approach (MALT, arXiv 2412.01928):
        1. Parse equations from formalizer output
        2. Solve via Rust (exact, sub-ms, deterministic)
        3. If solver fails → fall back to LLM chain-of-thought on the
           original task (the LLM can reason through what it can't formalize)

        This gives us the best of both: exact answers when formalization
        works, LLM reasoning when it doesn't.
        """
        import time as _time

        node = self.graph.get_node(node_idx)
        role = getattr(node, "role", f"node-{node_idx}")

        if context_override is not None:
            context = context_override
        elif self._statecore_enabled:
            context, state_frame, _control_ready = self._assemble_node_inputs(node_idx)
        else:
            context = self._gather_predecessor_context(node_idx)

        t0 = _time.monotonic()

        # ── Phase 1: Parse equations from formalizer output ────────────
        equations = []
        answer_var = None
        source = context if context else task

        for line in source.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("```") or line.startswith("-") or line.startswith("*"):
                continue
            if "=" in line and not line.startswith("="):
                parts = line.split("=", 1)
                var = parts[0].strip().lower().replace(" ", "_")
                expr = parts[1].strip()
                if var in ("answer", "the_answer", "final_answer", "result"):
                    answer_var = expr.strip().lower().replace(" ", "_")
                else:
                    equations.append((var, expr))

        # ── Phase 2: Try Rust solver ───────────────────────────────────
        solver_answer = None
        n_resolved = 0
        if equations:
            try:
                from sage_core import SmtVerifier
                solved = SmtVerifier.solve_equations(equations)
                n_resolved = len(solved)
                if solved:
                    # Find the answer variable
                    if answer_var and answer_var in solved:
                        solver_answer = str(solved[answer_var])
                    else:
                        target = self._infer_answer_variable(task, solved)
                        if target:
                            solver_answer = str(solved[target])
                        elif equations:
                            last_var = equations[-1][0]
                            if last_var in solved:
                                solver_answer = str(solved[last_var])
            except (ImportError, RuntimeError) as exc:
                log.warning("Solver node %d: Rust solver error: %s", node_idx, exc)

        solve_ms = (_time.monotonic() - t0) * 1000

        # ── Phase 3: Decide — use solver or fall back to LLM ──────────
        # Solver succeeded if: we got equations, resolved most of them,
        # and found a numeric answer.
        solver_ok = (
            solver_answer is not None
            and n_resolved >= len(equations) * 0.7  # resolved ≥70% of vars
        )

        if solver_ok:
            assert solver_answer is not None  # narrowed by solver_ok above
            output = solver_answer
            log.info(
                "Solver node %d (%s): Rust solved %d/%d vars, answer=%s (%.1fms)",
                node_idx, role, n_resolved, len(equations), output, solve_ms,
            )
        else:
            # Fall back to LLM chain-of-thought on the original task.
            # The formalizer tried but the solver couldn't handle it —
            # let a strong LLM reason through the problem directly.
            log.info(
                "Solver node %d (%s): Rust solved %d/%d vars (%.1fms) — "
                "falling back to LLM chain-of-thought",
                node_idx, role, n_resolved, len(equations), solve_ms,
            )
            try:
                from sage.llm.base import Message, Role
                messages = [
                    Message(
                        role=Role.SYSTEM,
                        content=(
                            "Solve this math problem step by step. "
                            "Verify each intermediate result. "
                            "Give your final answer as a single number only."
                        ),
                    ),
                    Message(role=Role.USER, content=task),
                ]
                response = await self._llm.generate(
                    messages=messages, config=self._config,
                )
                output = response.content or ""
            except Exception as exc:
                log.warning("Solver node %d: LLM fallback failed: %s", node_idx, exc)
                output = solver_answer or ""

        self._node_outputs[node_idx] = output
        return output

    @staticmethod
    def _infer_answer_variable(
        question: str, solved: dict[str, int],
    ) -> str | None:
        """Infer which solved variable the question asks about.

        Uses word-overlap scoring between the question and each solved
        variable name. The variable with the highest overlap wins.
        No regex — pure set intersection on normalized words.
        """
        # Normalize question into a set of lowercase word tokens
        q_lower = question.lower().replace("'s", " ").replace("'", " ")
        # Strip punctuation
        q_clean = "".join(c if c.isalnum() or c == " " else " " for c in q_lower)
        q_words = set(q_clean.split())

        # Remove common stop words that add noise
        stop = {
            "the", "a", "an", "is", "are", "was", "were", "how", "many",
            "much", "does", "do", "did", "have", "has", "had", "what",
            "each", "of", "in", "to", "for", "and", "or", "that", "this",
            "number", "total", "give", "your", "final", "answer", "only",
            "as", "step", "by", "think", "verify", "result", "intermediate",
        }
        q_content = q_words - stop

        if not q_content:
            return None

        # Score each solved variable by word overlap with the question
        best_var: str | None = None
        best_score = 0
        for var_name in solved:
            var_words = set(var_name.lower().split("_"))
            overlap = len(q_content & var_words)
            if overlap > best_score:
                best_score = overlap
                best_var = var_name

        return best_var if best_score >= 2 else None

    async def _capability_aware_fallback_generate(
        self,
        *,
        node_idx: int,
        messages: list[Message],
        original_config: Any,
    ) -> tuple[str, float] | None:
        """Retry a failed node through ModelAssigner + ProviderPool.

        Replaces the legacy connector-order fallback. Candidates must satisfy
        the graph node's capability requirements, remaining budget, explicit
        model exclusions, and the provider pool's runtime availability guard.
        """
        if self._assigner is None or self._provider_pool is None:
            return None

        node = self.graph.get_node(node_idx)
        failed_model_id = (
            getattr(node, "model_id", "") or getattr(original_config, "model", "") or ""
        )
        excluded: set[str] = {failed_model_id} if failed_model_id else set()
        original_model_id = failed_model_id

        while True:
            try:
                fallback_model_id = self._assigner.assign_single_node(
                    self.graph,
                    node_idx,
                    task_domain=self._task_domain,
                    budget_usd=self._remaining_budget_usd(),
                    exclude_model_ids=sorted(excluded) or None,
                    task_system=getattr(node, "system", None),
                )
            except ValueError:
                if original_model_id and hasattr(self.graph, "set_node_model_id"):
                    try:
                        self.graph.set_node_model_id(node_idx, original_model_id)
                    except Exception:  # noqa: BLE001
                        pass
                return None

            if (
                hasattr(self._provider_pool, "is_model_available")
                and not self._provider_pool.is_model_available(fallback_model_id)
            ):
                excluded.add(fallback_model_id)
                continue

            try:
                fallback_provider, fallback_config = self._provider_pool.resolve(
                    fallback_model_id,
                )
            except (RuntimeError, ValueError, AttributeError):
                excluded.add(fallback_model_id)
                continue

            try:
                response = await asyncio.wait_for(
                    fallback_provider.generate(messages=messages, config=fallback_config),
                    timeout=60.0,
                )
            except (RuntimeError, TimeoutError, asyncio.TimeoutError, ConnectionError) as exc:
                fallback_provider_name = getattr(fallback_config, "provider", "unknown")
                if hasattr(self._provider_pool, "record_failure"):
                    try:
                        self._provider_pool.record_failure(fallback_provider_name, exc)
                    except Exception:  # noqa: BLE001
                        pass
                excluded.add(fallback_model_id)
                continue

            output = response.content or ""
            node_cost = self._response_cost_usd(response)

            fallback_provider_name = getattr(fallback_config, "provider", "unknown")
            if hasattr(self._provider_pool, "record_success"):
                try:
                    self._provider_pool.record_success(fallback_provider_name)
                except Exception:  # noqa: BLE001
                    pass

            return output, node_cost

    async def _execute_node(
        self,
        node_idx: int,
        task: str,
        context_override: str | None = None,
        state_frame: StateFrame | None = None,
    ) -> str:
        """Execute a single topology node — LLM call or code sandbox.

        HyEvo hybrid dispatch (arXiv 2603.19639):
          - node_type="llm" → LLM inference via ProviderPool
          - node_type="code" → deterministic sandbox execution

        Parameters
        ----------
        context_override : str, optional
            Pre-captured context snapshot. Used by parallel batches to avoid
            race conditions on ``_node_outputs`` during ``asyncio.gather``.

        Sets ``sage_recurse_origin_node`` to ``node_idx`` around the
        dispatch so the sage_recurse tool's budget gate (Task D) can
        debit the right originating node.
        """
        token = sage_recurse_origin_node.set(node_idx)
        try:
            if self._is_over_budget():
                return _BUDGET_EXCEEDED_RESULT

            node = self.graph.get_node(node_idx)

            # HyEvo code node dispatch: deterministic sandbox execution
            node_type = getattr(node, "node_type", "llm")
            if node_type == "code":
                result = await self._execute_code_node(
                    node_idx,
                    task,
                    context_override,
                    state_frame,
                )
                self._record_spend_for_node(node_idx)
                return result

            # Formal solver node: parse equations from predecessor, solve via Rust
            if node_type == "solver" or getattr(node, "role", "") == "solver":
                result = await self._execute_solver_node(
                    node_idx,
                    task,
                    context_override,
                    state_frame,
                )
                self._record_spend_for_node(node_idx)
                return result

            # Phase 2: LLM nodes use agent_loop when factory available
            if self._agent_loop_factory:
                result = await self._execute_node_via_agent_loop(
                    node_idx,
                    task,
                    context_override,
                    state_frame,
                )
                self._record_spend_for_node(node_idx)
                return result

            role = getattr(node, "role", f"node-{node_idx}")
            caps = getattr(node, "required_capabilities", [])

            # Use custom prompt if available, otherwise generate from role
            custom_prompt = getattr(node, "prompt", "")
            if custom_prompt:
                system_prompt = custom_prompt
            else:
                # Meta-Harness: configurable default template
                _default_tmpl = (
                    self._harness.prompts.default_template if self._harness
                    else "You are acting as: {role}."
                )
                system_prompt = _default_tmpl.format(
                    role=role, capabilities=", ".join(caps) if caps else "",
                    task_preview=task[:200], n_predecessors=0,
                )
                if caps:
                    _cap_tmpl = (
                        self._harness.prompts.capability_template if self._harness
                        else " Your capabilities: {capabilities}."
                    )
                    system_prompt += _cap_tmpl.format(capabilities=", ".join(caps))

            # Meta-Harness: global prefix/suffix applied to ALL system prompts
            if self._harness:
                if self._harness.prompts.global_prefix:
                    system_prompt = self._harness.prompts.global_prefix + "\n" + system_prompt
                if self._harness.prompts.global_suffix:
                    system_prompt = system_prompt + "\n" + self._harness.prompts.global_suffix

            messages: list[Message] = [
                Message(role=Role.SYSTEM, content=system_prompt),
            ]

            if context_override is not None:
                context = context_override
            elif self._statecore_enabled:
                context, state_frame, _control_ready = self._assemble_node_inputs(node_idx)
            else:
                context = self._gather_predecessor_context(node_idx)
            state_block = self._render_state_frame(state_frame)
            if state_block:
                messages.append(
                    Message(
                        role=Role.SYSTEM,
                        content=state_block,
                    )
                )
            if context:
                messages.append(Message(
                    role=Role.SYSTEM,
                    content=f"Context from previous agents:\n{context}",
                ))

            messages.append(Message(role=Role.USER, content=task))

            # Resolve per-node model if ProviderPool available
            node_model_id = getattr(node, "model_id", "")
            if node_model_id and self._provider_pool:
                provider, config = self._provider_pool.resolve(node_model_id)
            else:
                provider, config = self._llm, self._config

            # Context gate: if payload exceeds model window, compress via summarization
            # Uses context_window (input limit, e.g. 128K) NOT max_tokens (output limit, e.g. 8K)
            total_chars = sum(len(m.content) for m in messages)
            context_window = getattr(config, 'context_window', 0) or 128000
            estimated_tokens = total_chars // 4
            if estimated_tokens > context_window * 0.85:
                # Compress the context message to fit
                context_msg = next(
                    (m for m in messages if m.content.startswith("Context from previous")),
                    None,
                )
                if context_msg:
                    log.warning(
                        "Context overflow for node %d (%s): %d tokens > %d * 0.85, compressing",
                        node_idx, role, estimated_tokens, context_window,
                    )
                    try:
                        summary_msgs = [
                            Message(role=Role.SYSTEM, content="Summarize concisely. Preserve all key facts, numbers, code, and conclusions."),
                            Message(role=Role.USER, content=context_msg.content[:context_window * 2]),
                        ]
                        summary_resp = await asyncio.wait_for(
                            provider.generate(messages=summary_msgs, config=config),
                            timeout=30.0,
                        )
                        context_msg.content = f"Context (summarized):\n{summary_resp.content or ''}"
                    except (RuntimeError, TimeoutError, asyncio.TimeoutError) as exc:
                        # Compression failed — hard truncate as last resort
                        max_chars = int(context_window * 0.6 * 4)
                        context_msg.content = context_msg.content[:max_chars] + "\n[truncated]"
                        log.error("Context compression failed for node %d: %s", node_idx, exc)

            # Per-node resilience: timeout + retry with fallback provider
            output = ""
            try:
                response = await asyncio.wait_for(
                    provider.generate(messages=messages, config=config),
                    timeout=60.0,  # 60s per node, not per topology
                )
                output = response.content or ""
                node_cost = self._response_cost_usd(response)
                self.total_cost_usd += node_cost
                self._node_costs[node_idx] = self._node_costs.get(node_idx, 0.0) + node_cost
                # Record success in circuit breaker
                provider_name = getattr(config, "provider", "unknown")
                if self._provider_pool and hasattr(self._provider_pool, "record_success"):
                    self._provider_pool.record_success(provider_name)
            except (RuntimeError, TimeoutError, asyncio.TimeoutError, ConnectionError) as exc:
                provider_name = getattr(config, "provider", "unknown")
                # Record failure in circuit breaker
                if self._provider_pool and hasattr(self._provider_pool, "record_failure"):
                    try:
                        self._provider_pool.record_failure(provider_name, exc)
                    except Exception:  # noqa: BLE001 - telemetry must not mask the real error
                        pass
                log.warning(
                    "[TopologyRunner] node %d (%s) failed with %s provider: %s - "
                    "attempting capability-aware fallback",
                    node_idx, role, provider_name, str(exc)[:150],
                )
                # cgpro 2026-04-28 R3 verify "best patch": always attempt the
                # capability-aware fallback (no `provider is not self._llm`
                # guard). When assigner+pool are wired, even a default-provider
                # failure deserves a retry on a different model. When they're
                # not wired, the helper returns None and we raise.
                fallback_result = await self._capability_aware_fallback_generate(
                    node_idx=node_idx,
                    messages=messages,
                    original_config=config,
                )
                if fallback_result is None:
                    log.warning(
                        "[TopologyRunner] node %d (%s) capability-aware fallback exhausted; raising",
                        node_idx,
                        role,
                    )
                    raise
                output, fallback_cost = fallback_result
                self.total_cost_usd += fallback_cost
                self._node_costs[node_idx] = (
                    self._node_costs.get(node_idx, 0.0) + fallback_cost
                )
                log.info(
                    "[TopologyRunner] node %d (%s) succeeded via capability-aware fallback",
                    node_idx,
                    role,
                )
            self._node_outputs[node_idx] = output
            log.info(
                "[TopologyRunner] node %d (%s) completed, output %d chars",
                node_idx,
                role,
                len(output),
            )
            self._record_spend_for_node(node_idx)
            return output
        finally:
            sage_recurse_origin_node.reset(token)

    async def _retry_with_upgrade(self, node_idx: int, decision: Any, task: str) -> str:
        """Model upgrade: re-resolve provider via ProviderPool and retry node.

        The controller already called assign_single_node on the topology to
        update the node's model_id. Re-executing the node picks up the new
        model automatically via ProviderPool.resolve().
        """
        if decision.new_model_id:
            try:
                self.graph.set_node_model_id(node_idx, decision.new_model_id)
                log.info("Node %d model upgraded to %s", node_idx, decision.new_model_id)
            except (AttributeError, Exception) as exc:
                log.warning("Could not set model_id on node %d: %s", node_idx, exc)
        return await self._execute_node(node_idx, task)

    async def _spawn_sub(self, node_idx: int, decision: Any, task: str) -> None:
        """Sub-agent spawn: run emergent sub-task and inject result into node output."""
        sub_task = decision.reason
        if not sub_task:
            return
        try:
            from sage.llm.base import Message, Role  # local re-import for clarity
            provider = self._llm
            config = self._config
            if self._provider_pool:
                node = self.graph.get_node(node_idx) if hasattr(self.graph, "get_node") else None
                model_id = getattr(node, "model_id", "") if node else ""
                if model_id:
                    provider, config = self._provider_pool.resolve(model_id)
            response = await provider.generate(
                messages=[Message(role=Role.USER, content=sub_task)],
                config=config,
            )
            sub_result = response.content or ""
            # Inject into node outputs
            existing = self._node_outputs.get(node_idx, "")
            self._node_outputs[node_idx] = f"{existing}\n[Sub-agent]: {sub_result}"
        except (RuntimeError, TimeoutError, ValueError) as exc:
            log.warning("Sub-agent spawn failed: %s", exc)

    async def _apply_controller_decision(
        self,
        *,
        node_idx: int,
        result: str,
        task: str,
        latency_ms: float,
        parallel_outputs: list[str] | None,
    ) -> tuple[str, Any | None]:
        """Run the controller for one node and apply its decision.

        Calls self._controller.evaluate_and_decide ONCE. Handles all 5
        actions: upgrade_model, spawn_subagent, reroute_topology, prune_node,
        open_gate. Returns (possibly-updated result, raw decision or None).
        Caller is responsible for yielding the _ControllerDecisionEvent and
        deciding whether to early-return on reroute (look at decision.action).
        Does NOT mark the node completed -- caller does that.
        """
        if not self._controller:
            return result, None

        node = self.graph.get_node(node_idx)
        node_ctx = {
            "node_idx": node_idx,
            "latency_ms": latency_ms,
            "model_id": getattr(node, "model_id", ""),
            "output_length": len(result),
            "axis_hint": self._axis_hint,
        }
        decision = self._controller.evaluate_and_decide(
            node_idx,
            result,
            task,
            self.graph,
            node_ctx,
            parallel_outputs=parallel_outputs,
        )

        if (
            self._approval_callback
            and decision.action in ("upgrade_model", "reroute_topology", "open_gate")
        ):
            try:
                approved = await self._approval_callback(decision)
                if not approved:
                    log.info("HITL rejected %s for node %d", decision.action, node_idx)
                    decision = type(decision)(action="continue", target_node=node_idx)
            except (RuntimeError, TimeoutError, asyncio.TimeoutError) as exc:
                log.warning("HITL callback failed: %s, proceeding", exc)

        if decision.action == "upgrade_model":
            result = await self._retry_with_upgrade(node_idx, decision, task)
            self._node_outputs[node_idx] = result
        elif decision.action == "spawn_subagent":
            await self._spawn_sub(node_idx, decision, task)
        elif decision.action == "reroute_topology":
            pass
        elif decision.action == "prune_node":
            try:
                self.executor.mark_skipped(decision.target_node)
            except (AttributeError, Exception):
                pass  # Executor may not support skip
            log.info("Node %d pruned by controller", decision.target_node)
        elif decision.action == "open_gate":
            target = decision.gate_target
            source = decision.gate_source
            if target is not None and source is not None:
                count = self._node_exec_count.get(target, 1)
                if count < self._max_rounds:
                    self.executor.open_gate(self.graph, source, target)
                    self.executor.reset_node(target)
                    self._node_exec_count[target] = count + 1
                    log.info(
                        "Multi-turn: reopened gate %d->%d (round %d/%d)",
                        source,
                        target,
                        count + 1,
                        self._max_rounds,
                    )
                else:
                    log.info(
                        "Multi-turn: max rounds reached for node %d (%d/%d)",
                        target,
                        count,
                        self._max_rounds,
                    )

        return result, decision

    def _runtime_topology_id(self) -> str:
        return (
            str(getattr(self.graph, "id", "") or "")
            or str(getattr(self.graph, "topology_id", "") or "")
            or str(getattr(self.graph, "template_type", "") or "")
        )

    def _runtime_provider_id_for_node(self, node: Any) -> str:
        model_id = getattr(node, "model_id", "") or ""
        if self._provider_pool is not None and hasattr(self._provider_pool, "infer_provider"):
            try:
                provider_id = self._provider_pool.infer_provider(model_id)
                if provider_id:
                    return str(provider_id)
            except (AttributeError, RuntimeError, ValueError, TypeError):
                pass
        config_provider = getattr(self._config, "provider", "") if self._config is not None else ""
        return str(config_provider or "")

    def _runtime_predecessor_ids(self, node_idx: int) -> tuple[str, ...]:
        try:
            predecessors = self.graph.get_predecessors(node_idx)
        except (AttributeError, Exception):
            predecessors = []
        return tuple(str(idx) for idx in predecessors)

    def _runtime_edge_ids(self, node_idx: int) -> tuple[str, ...]:
        return tuple(f"{pred}->{node_idx}" for pred in self._runtime_predecessor_ids(node_idx))

    def _runtime_node_cost_usd(self, node_idx: int) -> float:
        return float(
            self._node_event_costs.get(
                node_idx,
                self._node_costs.get(node_idx, 0.0),
            )
            or 0.0
        )

    def _runtime_emit_node_started(
        self,
        node_idx: int,
        role: str,
        open_nodes: dict[int, str],
    ) -> None:
        node = self.graph.get_node(node_idx)
        open_nodes[node_idx] = role
        predecessors_by_channel = (
            self._predecessors_by_channel(node_idx)
            if self._statecore_enabled
            else None
        )
        predecessor_ids = self._runtime_predecessor_ids(node_idx)
        model_id = getattr(node, "model_id", "") or ""
        provider_id = self._runtime_provider_id_for_node(node)
        seq = None
        if self._event_log is not None:
            seq = self._event_log.emit_node_started(
                topology_id=self._runtime_topology_id(),
                node_id=str(node_idx),
                node_role=role,
                attempt=self._node_exec_count.get(node_idx, 0) + 1,
                model_id=model_id,
                provider_id=provider_id,
                predecessor_ids=predecessor_ids,
                edge_ids=self._runtime_edge_ids(node_idx),
                predecessors_by_channel=predecessors_by_channel,
            )
        if self._run_frame_builder is not None:
            node_run_id = self._run_frame_builder.record_node_started(
                seq=seq,
                node_id=str(node_idx),
                provider_id=provider_id,
                model_id=model_id,
                predecessor_ids=predecessor_ids,
                predecessors_by_channel=predecessors_by_channel,
            )
            self._run_frame_node_runs[node_idx] = node_run_id

    def _runtime_emit_node_completed(
        self,
        node_idx: int,
        role: str,
        output: str,
        latency_ms: float,
        open_nodes: dict[int, str],
    ) -> None:
        node = self.graph.get_node(node_idx)
        cost_usd = self._runtime_node_cost_usd(node_idx)
        model_id = getattr(node, "model_id", "") or ""
        provider_id = self._runtime_provider_id_for_node(node)
        seq = None
        if self._event_log is not None:
            seq = self._event_log.emit_node_completed(
                node_id=str(node_idx),
                node_role=role,
                output=output,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                model_id=model_id,
                provider_id=provider_id,
            )
        if self._run_frame_builder is not None:
            node_run_id = self._run_frame_node_runs.get(node_idx)
            if node_run_id is not None:
                self._run_frame_builder.record_node_completed(
                    seq=seq,
                    node_run_id=node_run_id,
                    output=output,
                    latency_ms=latency_ms,
                    cost_usd=cost_usd,
                )
                self._run_frame_node_runs.pop(node_idx, None)
        open_nodes.pop(node_idx, None)

    def _runtime_emit_controller_decision(self, node_idx: int, decision: Any) -> None:
        target = getattr(decision, "target_node", "")
        gate_source = getattr(decision, "gate_source", None)
        gate_target = getattr(decision, "gate_target", None)
        action = str(getattr(decision, "action", "continue") or "continue")
        target_node_id = "" if target is None else str(target)
        gate_source_id = None if gate_source is None else str(gate_source)
        gate_target_id = None if gate_target is None else str(gate_target)
        reason = getattr(decision, "reason", "") or ""
        seq = None
        if self._event_log is not None:
            seq = self._event_log.emit_controller_decision(
                node_id=str(node_idx),
                action=action,
                target_node_id=target_node_id,
                gate_source_id=gate_source_id,
                gate_target_id=gate_target_id,
                reason=reason,
            )
        if self._run_frame_builder is not None:
            self._run_frame_builder.record_controller_decision(
                seq=seq,
                node_run_id=self._run_frame_node_runs.get(node_idx),
                action=action,
                target_node_id=target_node_id,
                gate_source_id=gate_source_id,
                gate_target_id=gate_target_id,
                reason=reason,
            )

    def _runtime_emit_failure(
        self,
        *,
        kind: str,
        error_type: str,
        message: str,
        open_nodes: dict[int, str],
        node_idx: int | None = None,
    ) -> None:
        if node_idx is None:
            targets = list(open_nodes)
        else:
            targets = [node_idx] if node_idx in open_nodes else []
        if not targets:
            seq = None
            if self._event_log is not None:
                seq = self._event_log.emit_failure(
                    kind=kind,
                    error_type=error_type,
                    message=message,
                    node_id="" if node_idx is None else str(node_idx),
                )
            if self._run_frame_builder is not None:
                node_run_id = (
                    self._run_frame_node_runs.get(node_idx)
                    if node_idx is not None
                    else None
                )
                self._run_frame_builder.record_failure(
                    seq=seq,
                    node_run_id=node_run_id,
                    kind=kind,
                    error_type=error_type,
                    message=message,
                )
            return
        for target in targets:
            seq = None
            if self._event_log is not None:
                seq = self._event_log.emit_failure(
                    kind=kind,
                    error_type=error_type,
                    message=message,
                    node_id=str(target),
                )
            if self._run_frame_builder is not None:
                self._run_frame_builder.record_failure(
                    seq=seq,
                    node_run_id=self._run_frame_node_runs.get(target),
                    kind=kind,
                    error_type=error_type,
                    message=message,
                )
                self._run_frame_node_runs.pop(target, None)
            open_nodes.pop(target, None)

    def _runtime_emit_budget_exceeded(self, open_nodes: dict[int, str]) -> None:
        remaining = self._remaining_budget_usd()
        if remaining == float("inf"):
            remaining = 0.0
        budget_limit = float(self._budget_usd or 0.0)
        cost_so_far = float(self.total_cost_usd or 0.0)
        seq = None
        if self._event_log is not None:
            seq = self._event_log.emit_budget(
                kind="exceeded",
                budget_limit_usd=budget_limit,
                budget_remaining_usd=float(remaining),
                cost_so_far_usd=cost_so_far,
            )
        if self._run_frame_builder is not None:
            self._run_frame_builder.record_budget(
                seq=seq,
                kind="exceeded",
                budget_limit_usd=budget_limit,
                budget_remaining_usd=float(remaining),
                cost_so_far_usd=cost_so_far,
            )
        self._runtime_emit_failure(
            kind="budget_exceeded",
            error_type="BudgetExceeded",
            message="budget exceeded",
            open_nodes=open_nodes,
        )

    async def _run_core(self, task: str) -> AsyncIterator[_RunEvent]:
        """Central executor loop yielding private RunEvents.

        All 3 public methods (run, run_traced, run_stream) consume this
        generator and translate events to their respective return shapes.
        """
        import time as _time

        last_output = ""
        nodes_executed = 0
        open_nodes: dict[int, str] = {}

        while not self.executor.is_done():
            if self._is_over_budget():
                self._runtime_emit_budget_exceeded(open_nodes)
                yield _BudgetExceededEvent()
                return

            ready = self.executor.next_ready(self.graph)
            if not ready:
                break

            if len(ready) == 1:
                node_idx = ready[0]
                node = self.graph.get_node(node_idx)
                role = getattr(node, "role", f"node-{node_idx}")
                context_override: str | None = None
                state_frame: StateFrame | None = None
                if self._statecore_enabled:
                    context_override, state_frame, should_execute = (
                        self._prepare_statecore_node(node_idx, open_nodes)
                    )
                    if not should_execute:
                        continue
                self._runtime_emit_node_started(node_idx, role, open_nodes)
                yield _NodeStartEvent(node_idx=node_idx, role=role)

                t0 = _time.monotonic()
                try:
                    result = await self._execute_node(
                        node_idx,
                        task,
                        context_override=context_override,
                        state_frame=state_frame,
                    )
                except Exception as exc:
                    self._runtime_emit_failure(
                        kind="provider_error",
                        error_type=type(exc).__name__,
                        message=str(exc),
                        open_nodes=open_nodes,
                        node_idx=node_idx,
                    )
                    raise
                latency_ms = (_time.monotonic() - t0) * 1000

                if result == _BUDGET_EXCEEDED_RESULT:
                    self._runtime_emit_budget_exceeded(open_nodes)
                    yield _BudgetExceededEvent()
                    return

                if self._is_over_budget():
                    self.executor.mark_completed(node_idx)
                    self._node_exec_count[node_idx] = (
                        self._node_exec_count.get(node_idx, 0) + 1
                    )
                    self._runtime_emit_budget_exceeded(open_nodes)
                    yield _BudgetExceededEvent()
                    return

                result, decision = await self._apply_controller_decision(
                    node_idx=node_idx,
                    result=result,
                    task=task,
                    latency_ms=latency_ms,
                    parallel_outputs=None,
                )
                if decision is not None:
                    self._runtime_emit_controller_decision(node_idx, decision)
                    yield _ControllerDecisionEvent(
                        node_idx=node_idx,
                        action=decision.action,
                        reason=getattr(decision, "reason", "") or "",
                    )
                    if decision.action == "reroute_topology":
                        self._runtime_emit_failure(
                            kind="controller_reroute",
                            error_type="ControllerReroute",
                            message=getattr(decision, "reason", "") or "controller reroute",
                            open_nodes=open_nodes,
                        )
                        yield _RerouteEvent(reason=getattr(decision, "reason", "") or "")
                        return

                self.executor.mark_completed(node_idx)
                self._node_exec_count[node_idx] = (
                    self._node_exec_count.get(node_idx, 0) + 1
                )
                self._node_state_deltas.setdefault(node_idx, StateDelta())
                last_output = self._node_outputs.get(node_idx, result)
                nodes_executed += 1
                self._runtime_emit_node_completed(
                    node_idx,
                    role,
                    last_output,
                    latency_ms,
                    open_nodes,
                )
                yield _NodeDoneEvent(
                    node_idx=node_idx,
                    role=role,
                    output=last_output,
                    latency_ms=latency_ms,
                    model_id=getattr(node, "model_id", "") or "",
                )
            else:
                prepared_inputs: dict[int, tuple[str | None, StateFrame | None]] = {}
                if self._statecore_enabled:
                    ready_to_execute: list[int] = []
                    for idx in ready:
                        context_override, state_frame, should_execute = (
                            self._prepare_statecore_node(idx, open_nodes)
                        )
                        if should_execute:
                            ready_to_execute.append(idx)
                            prepared_inputs[idx] = (context_override, state_frame)
                    if not ready_to_execute:
                        continue
                    for idx in ready_to_execute:
                        node = self.graph.get_node(idx)
                        role = getattr(node, "role", f"node-{idx}")
                        self._runtime_emit_node_started(idx, role, open_nodes)
                        yield _NodeStartEvent(
                            node_idx=idx,
                            role=role,
                        )
                    coros = [
                        self._execute_node(
                            idx,
                            task,
                            context_override=prepared_inputs[idx][0],
                            state_frame=prepared_inputs[idx][1],
                        )
                        for idx in ready_to_execute
                    ]
                else:
                    ready_to_execute = list(ready)
                    for idx in ready_to_execute:
                        node = self.graph.get_node(idx)
                        role = getattr(node, "role", f"node-{idx}")
                        self._runtime_emit_node_started(idx, role, open_nodes)
                        yield _NodeStartEvent(
                            node_idx=idx,
                            role=role,
                        )

                    ctx_snapshot = self._gather_all_context()
                    coros = [
                        self._execute_node(idx, task, context_override=ctx_snapshot)
                        for idx in ready_to_execute
                    ]
                t0_par = _time.monotonic()
                results = await asyncio.gather(*coros, return_exceptions=True)
                par_latency_ms = (_time.monotonic() - t0_par) * 1000

                first_exc: BaseException | None = None
                for idx, parallel_result in zip(ready_to_execute, results):
                    if isinstance(parallel_result, BaseException):
                        if first_exc is None:
                            first_exc = parallel_result
                        self._runtime_emit_failure(
                            kind="provider_error",
                            error_type=type(parallel_result).__name__,
                            message=str(parallel_result),
                            open_nodes=open_nodes,
                            node_idx=idx,
                        )
                if first_exc is not None:
                    for idx, parallel_result in zip(ready_to_execute, results):
                        if isinstance(parallel_result, BaseException):
                            continue
                        node = self.graph.get_node(idx)
                        role = getattr(node, "role", f"node-{idx}")
                        output = self._node_outputs.get(idx, str(parallel_result))
                        self._runtime_emit_node_completed(
                            idx,
                            role,
                            output,
                            par_latency_ms,
                            open_nodes,
                        )
                    raise first_exc

                if _BUDGET_EXCEEDED_RESULT in results:
                    self._runtime_emit_budget_exceeded(open_nodes)
                    yield _BudgetExceededEvent()
                    return

                if self._is_over_budget():
                    for idx in ready_to_execute:
                        self.executor.mark_completed(idx)
                        self._node_exec_count[idx] = self._node_exec_count.get(idx, 0) + 1
                    self._runtime_emit_budget_exceeded(open_nodes)
                    yield _BudgetExceededEvent()
                    return

                parallel_outputs = [str(parallel_result) for parallel_result in results]
                updated_results: list[str] = []
                reroute_reason: str | None = None
                for idx, parallel_result in zip(ready_to_execute, parallel_outputs):
                    updated_result, decision = await self._apply_controller_decision(
                        node_idx=idx,
                        result=parallel_result,
                        task=task,
                        latency_ms=par_latency_ms,
                        parallel_outputs=parallel_outputs,
                    )
                    updated_results.append(updated_result)
                    if decision is not None:
                        self._runtime_emit_controller_decision(idx, decision)
                        yield _ControllerDecisionEvent(
                            node_idx=idx,
                            action=decision.action,
                            reason=getattr(decision, "reason", "") or "",
                        )
                        if decision.action == "reroute_topology" and reroute_reason is None:
                            reroute_reason = getattr(decision, "reason", "") or ""

                if reroute_reason is not None:
                    self._runtime_emit_failure(
                        kind="controller_reroute",
                        error_type="ControllerReroute",
                        message=reroute_reason or "controller reroute",
                        open_nodes=open_nodes,
                    )
                    yield _RerouteEvent(reason=reroute_reason)
                    return

                for idx, output in zip(ready_to_execute, updated_results):
                    self.executor.mark_completed(idx)
                    self._node_exec_count[idx] = self._node_exec_count.get(idx, 0) + 1
                    self._node_state_deltas.setdefault(idx, StateDelta())
                    node = self.graph.get_node(idx)
                    last_output = self._node_outputs.get(idx, output)
                    nodes_executed += 1
                    role = getattr(node, "role", f"node-{idx}")
                    self._runtime_emit_node_completed(
                        idx,
                        role,
                        last_output,
                        par_latency_ms,
                        open_nodes,
                    )
                    yield _NodeDoneEvent(
                        node_idx=idx,
                        role=role,
                        output=last_output,
                        latency_ms=par_latency_ms,
                        model_id=getattr(node, "model_id", "") or "",
                    )

        yield _TopologyDoneEvent(final_output=last_output, node_count=nodes_executed)

    async def run(self, task: str) -> str:
        """Execute the full topology, returning the final node's output.

        For parallel batches, ``last_output`` is the last node in executor
        order. Topologies that need aggregation should include an explicit
        aggregator node in a subsequent batch.

        If a controller is attached and decides ``reroute_topology``, this
        method returns the special sentinel ``"__REROUTE__"`` so the caller
        (Pipeline Stage 4) can handle the reroute.
        """
        final = ""
        async for event in self._run_core(task):
            if isinstance(event, _BudgetExceededEvent):
                return _BUDGET_EXCEEDED_RESULT
            if isinstance(event, _RerouteEvent):
                return "__REROUTE__"
            if isinstance(event, _TopologyDoneEvent):
                final = event.final_output
        return final

    async def run_traced(self, task: str) -> list[dict]:
        """Execute topology and return per-node traces for GiGPO step rewards.

        Returns a list of dicts, one per executed node:
            [{"node_idx": 0, "role": "coder", "output": "...", "latency": 1.2}, ...]

        Uses the same execution logic as run() (ProviderPool, controller, fallback)
        but captures per-node metadata instead of just the final output.
        """
        traces: list[dict] = []
        async for event in self._run_core(task):
            if isinstance(event, _NodeDoneEvent):
                traces.append(
                    {
                        "node_idx": event.node_idx,
                        "role": event.role,
                        "output": event.output,
                        "latency": event.latency_ms / 1000.0,
                        "model_id": event.model_id,
                    }
                )
        return traces

    async def run_stream(self, task: str):
        """Execute topology, yielding per-node events as an async generator.

        Yields dicts with event types:
        - {"type": "node_start", "node_idx": int, "role": str}
        - {"type": "node_done", "node_idx": int, "role": str, "output": str, "latency_ms": float}
        - {"type": "topology_done", "final_output": str, "node_count": int}
        - {"type": "topology_reroute", "reason": str}  (on controller reroute)

        Enables real-time UI updates (LangGraph-style streaming) and is the
        foundation for HITL interrupt/resume (Patch 6).
        """
        async for event in self._run_core(task):
            if isinstance(event, _NodeStartEvent):
                yield {
                    "type": "node_start",
                    "node_idx": event.node_idx,
                    "role": event.role,
                }
            elif isinstance(event, _NodeDoneEvent):
                yield {
                    "type": "node_done",
                    "node_idx": event.node_idx,
                    "role": event.role,
                    "output": event.output,
                    "latency_ms": event.latency_ms,
                }
            elif isinstance(event, _RerouteEvent):
                yield {"type": "topology_reroute", "reason": event.reason}
                return
            elif isinstance(event, _BudgetExceededEvent):
                yield {
                    "type": "topology_done",
                    "final_output": _BUDGET_EXCEEDED_RESULT,
                    "node_count": 0,
                }
                return
            elif isinstance(event, _TopologyDoneEvent):
                yield {
                    "type": "topology_done",
                    "final_output": event.final_output,
                    "node_count": event.node_count,
                }
        return
