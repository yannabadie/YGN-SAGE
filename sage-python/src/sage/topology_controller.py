"""TopologyController — runtime adaptation for Pipeline Stage 4.

Evaluates node output quality after each execution step and decides
whether to continue, upgrade model, prune node, reroute topology,
or spawn a sub-agent. Research basis: AgentDropout (ACL 2025),
AdaptOrch (arXiv 2026), OpenSage (ICML), Self-Regulation (arXiv).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class AdaptationDecision:
    """Decision from TopologyController after evaluating a node's output."""
    action: str  # "continue", "upgrade_model", "prune_node", "reroute_topology", "spawn_subagent", "open_gate"
    target_node: int | None = None
    reason: str = ""
    new_model_id: str | None = None
    invariant_feedback: str | None = None  # clause-level from OxiZ
    gate_source: int | None = None  # for open_gate: source node of back-edge
    gate_target: int | None = None  # for open_gate: target node to re-execute


# Regex for detecting structured reasoning content
_STRUCTURED_CONTENT = re.compile(r'<think>|```|assert\s|def\s+test_|proof:|invariant:', re.IGNORECASE)
_TASK_TOKEN = re.compile(r"[a-zA-Z_]{4,}")
_ERROR_OUTPUT = re.compile(
    r"^\s*(error|exception|traceback|timeout|failed)\b|"
    r"\b(traceback|stack trace|timed out|no output|failed with)\b",
    re.IGNORECASE,
)


class TopologyController:
    """Runtime adaptation controller for Pipeline Stage 4.

    Thresholds calibrated on TopologyBench results (March 2026).
    """

    THETA_GOOD = 0.7
    THETA_CRITICAL = 0.3
    THETA_CONSISTENCY = 0.5
    THETA_PRUNE = 0.2
    MAX_RETRIES = 2
    MAX_REROUTES = 1
    MAX_GATE_TURNS = 2  # Multi-turn refinement limit (MALT arXiv 2412.01928)
    MAX_SPAWNS = 3

    def __init__(
        self,
        assigner: Any = None,
        quality_estimator: Any = None,
        prm: Any = None,
        policy_verifier: Any = None,
        embedder: Any = None,
        event_bus: Any = None,
    ) -> None:
        self._assigner = assigner
        self._qe = quality_estimator
        self._prm = prm
        self._pv = policy_verifier
        self._embedder = embedder
        self._event_bus = event_bus
        self._node_retries: dict[int, int] = {}
        self._node_qualities: dict[int, float] = {}
        self._gate_loops: dict[int, int] = {}  # Multi-turn refinement tracker
        self._reroute_count = 0
        self._spawn_count = 0
        self._abstain_count = 0

    def quality_stats(self) -> dict:
        """Return quality tracking stats for diagnostics."""
        return {
            "abstain_count": self._abstain_count,
            "node_qualities": dict(self._node_qualities),
            "reroute_count": self._reroute_count,
            "spawn_count": self._spawn_count,
        }

    def _emit(self, event_type: str, data: dict) -> None:
        if self._event_bus and hasattr(self._event_bus, 'emit'):
            try:
                from sage.agent_loop import AgentEvent
                self._event_bus.emit(AgentEvent(phase="PIPELINE", data={"stage": event_type, **data}))
            except Exception:
                pass

    def evaluate_and_decide(
        self,
        node_idx: int,
        result: str | None = None,
        task: str = "",
        topology: Any = None,
        ctx: Any = None,
        parallel_outputs: list[str] | None = None,
        *,
        output: str | None = None,
    ) -> AdaptationDecision:
        """Core decision logic — called after each node execution.

        Args:
            node_idx: index of the node that just completed
            result: the node's output text
            task: the original task string
            topology: TopologyGraph instance
            ctx: PipelineContext
            parallel_outputs: outputs from sibling parallel nodes (if any)
        """
        result = result if result is not None else (output or "")

        # Early exit: max reroute hit
        if self._reroute_count >= self.MAX_REROUTES:
            if parallel_outputs and self.compute_consistency_score(parallel_outputs) < self.THETA_CONSISTENCY:
                self._emit("MAX_REROUTE_HIT", {"node": node_idx, "reroute_count": self._reroute_count})
                log.warning("Max reroute limit reached (count=%d), forcing continue", self._reroute_count)

        # Empty/error outputs are structural failures, not quality failures.
        if self._is_empty_or_error(result):
            if self._reroute_count < self.MAX_REROUTES:
                self._reroute_count += 1
                reason = "empty output" if not result.strip() else "error-like output"
                self._emit("REROUTE_TOPOLOGY", {"node": node_idx, "reason": reason})
                return AdaptationDecision(
                    action="reroute_topology",
                    target_node=node_idx,
                    reason=reason,
                )
            return AdaptationDecision(
                action="continue",
                target_node=node_idx,
                reason="reroute budget exhausted",
            )

        # Compute quality (formal estimator when available, heuristic fallback otherwise).
        _abstain_before = self._abstain_count
        quality = self._compute_quality(node_idx, result, task, ctx)
        _quality_is_known = self._abstain_count == _abstain_before  # True if estimator gave a real score
        self._node_qualities[node_idx] = quality

        # Depth axis: verify arithmetic before quality cascade (MASPRM arXiv 2510.24803)
        axis_hint = ""
        axis_hint = str(self._ctx_value(ctx, "axis_hint", "") or "")
        if axis_hint == 'depth':
            verified, feedback = self._verify_arithmetic(result)
            if not verified:
                retries_d = self._node_retries.get(node_idx, 0)
                retry_limit = self._max_retries_for_node(topology, node_idx)
                if retries_d < retry_limit:
                    self._node_retries[node_idx] = retries_d + 1
                    return AdaptationDecision(
                        action="upgrade_model",
                        target_node=node_idx,
                        reason=f"arithmetic verification failed: {feedback}",
                        invariant_feedback=feedback,
                    )

        # Decision cascade
        # 1. Good quality -> continue
        if quality >= self.THETA_GOOD:
            return AdaptationDecision(action="continue", target_node=node_idx)

        # 1.5. Debate refinement is gated by disagreement between peer debaters.
        if self.THETA_CRITICAL <= quality < self.THETA_GOOD:
            gate_decision = self._open_gate(node_idx, topology, parallel_outputs)
            if gate_decision is not None:
                return gate_decision

        # 2. Critical quality -> upgrade model
        retries = self._node_retries.get(node_idx, 0)
        retry_limit = self._max_retries_for_node(topology, node_idx)
        if quality < self.THETA_CRITICAL and retries < retry_limit:
            self._node_retries[node_idx] = retries + 1
            # Get invariant feedback for S3 nodes if available
            feedback = self._get_invariant_feedback(result, topology, node_idx)
            new_model_id = self._resolve_upgrade_model(node_idx, task, topology, ctx)
            return AdaptationDecision(
                action="upgrade_model",
                target_node=node_idx,
                reason=f"quality={quality:.2f} < {self.THETA_CRITICAL}",
                invariant_feedback=feedback,
                new_model_id=new_model_id,
            )

        # 3. Parallel inconsistency -> reroute topology
        if parallel_outputs and self._reroute_count < self.MAX_REROUTES and not self._is_debate_topology(topology):
            consistency = self.compute_consistency_score(parallel_outputs)
            if consistency < self.THETA_CONSISTENCY:
                self._reroute_count += 1
                self._emit("REROUTE_TOPOLOGY", {"consistency": consistency, "node": node_idx})
                return AdaptationDecision(
                    action="reroute_topology",
                    target_node=node_idx,
                    reason=f"consistency={consistency:.2f} < {self.THETA_CONSISTENCY}",
                )

        # 4. Low importance -> prune (only when quality is KNOWN, not abstained)
        if parallel_outputs and _quality_is_known and not self._is_debate_topology(topology):
            importance = self.compute_importance_score(node_idx, result, parallel_outputs)
            if importance < self.THETA_PRUNE:
                self._emit("PRUNE_NODE", {"node": node_idx, "importance": importance})
                return AdaptationDecision(
                    action="prune_node",
                    target_node=node_idx,
                    reason=f"importance={importance:.2f} < {self.THETA_PRUNE}",
                )

        # 5. Emergent sub-task detected -> spawn
        emergent = self._detect_emergent_subtask(result)
        if emergent and self._spawn_count < self.MAX_SPAWNS:
            self._spawn_count += 1
            self._emit("SPAWN_SUBAGENT", {"node": node_idx, "subtask": emergent[:100]})
            return AdaptationDecision(
                action="spawn_subagent",
                target_node=node_idx,
                reason=emergent,
            )

        # 6. Default: continue (accept imperfect result)
        return AdaptationDecision(action="continue", target_node=node_idx)

    def _compute_quality(self, node_idx: int, result: str, task: str, ctx: Any) -> float:
        """Formal quality estimate with heuristic fallback and optional PRM blend."""
        base_score: float | None = None
        if self._qe:
            try:
                latency = float(self._ctx_value(ctx, "latency_ms", 0.0) or 0.0)
                base_score = self._qe.estimate(task, result, latency)
            except Exception:
                pass

        if base_score is None:
            self._abstain_count += 1
            base_score = self._heuristic_quality(result, task)
            log.debug(
                "QualityEstimator abstained for node %d, using heuristic fallback=%.3f",
                node_idx,
                base_score,
            )
        quality = float(base_score)

        # PRM only for structured content (guard: -1.0 on plain text)
        if self._prm and _STRUCTURED_CONTENT.search(result):
            try:
                r_path, _ = self._prm.calculate_r_path(result)
                if r_path >= 0.0:  # valid PRM score
                    return 0.8 * quality + 0.2 * r_path
            except Exception as exc:
                log.debug("PRM scoring failed: %s", exc)

        return quality

    @staticmethod
    def _ctx_value(ctx: Any, key: str, default: Any = None) -> Any:
        if ctx is None:
            return default
        if isinstance(ctx, dict):
            return ctx.get(key, default)
        if hasattr(ctx, "get") and callable(ctx.get):
            try:
                return ctx.get(key, default)
            except TypeError:
                pass
        return getattr(ctx, key, default)

    @staticmethod
    def _coerce_float(value: Any, default: float) -> float:
        if isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return default
        return default

    @staticmethod
    def _coerce_str(value: Any, default: str = "") -> str:
        return value if isinstance(value, str) else default

    @staticmethod
    def _is_empty_or_error(result: str) -> bool:
        stripped = result.strip()
        if not stripped:
            return True
        return bool(_ERROR_OUTPUT.search(stripped))

    @staticmethod
    def _is_debate_topology(topology: Any) -> bool:
        return str(getattr(topology, "template_type", "") or "") == "debate"

    def _heuristic_quality(self, result: str, task: str) -> float:
        """Cheap structural fallback when formal quality signals are unavailable."""
        stripped = result.strip()
        if not stripped:
            return 0.0
        if _ERROR_OUTPUT.search(stripped):
            return 0.0

        words = stripped.split()
        length_score = min(len(words) / 96.0, 1.0)
        structure_score = 1.0 if _STRUCTURED_CONTENT.search(stripped) else 0.35
        if "\n" in stripped or re.search(r"(^|\n)([-*]|\d+\.)\s", stripped):
            structure_score = max(structure_score, 0.7)
        elif re.search(r"[.:;]", stripped):
            structure_score = max(structure_score, 0.55)

        task_terms = set(_TASK_TOKEN.findall(task.lower()))
        result_terms = set(_TASK_TOKEN.findall(stripped.lower()))
        overlap = len(task_terms & result_terms)
        coverage_score = min(overlap / 4.0, 1.0) if task_terms else 0.5

        quality = 0.45 * length_score + 0.35 * structure_score + 0.20 * coverage_score
        return max(0.0, min(1.0, quality))

    def _max_retries_for_node(self, topology: Any, node_idx: int) -> int:
        node = topology.get_node(node_idx) if topology is not None and hasattr(topology, "get_node") else None
        node_limit = int(getattr(node, "max_retries", 0) or self.MAX_RETRIES)
        return max(0, min(node_limit, self.MAX_RETRIES))

    def _resolve_upgrade_model(self, node_idx: int, task: str, topology: Any, ctx: Any) -> str | None:
        node = topology.get_node(node_idx) if topology is not None and hasattr(topology, "get_node") else None
        if node is None:
            return None

        current_model_id = self._coerce_str(getattr(node, "model_id", ""), "")
        task_domain = self._infer_task_domain(task, node, ctx)
        budget_usd = self._coerce_float(
            self._ctx_value(ctx, "budget_usd", None)
            or self._ctx_value(ctx, "budget", None)
            or getattr(node, "max_cost_usd", 1.0)
            or 1.0,
            1.0,
        )

        if self._assigner and hasattr(self._assigner, "assign_single_node") and hasattr(topology, "set_node_model_id"):
            excluded = [current_model_id] if current_model_id else None
            try:
                return self._assigner.assign_single_node(
                    topology,
                    node_idx,
                    task_domain,
                    budget_usd,
                    excluded,
                )
            except TypeError:
                try:
                    return self._assigner.assign_single_node(
                        topology,
                        node_idx,
                        task_domain,
                        budget_usd,
                    )
                except Exception as exc:
                    log.debug("assign_single_node retry without exclusion failed: %s", exc)
            except Exception as exc:
                log.debug("assign_single_node failed for node %d: %s", node_idx, exc)

        return self._resolve_fallback_model(node, current_model_id)

    def _resolve_fallback_model(self, node: Any, current_model_id: str) -> str | None:
        fallback = self._coerce_str(getattr(node, "fallback_tier", ""), "")
        if not fallback:
            return None
        try:
            from pathlib import Path

            from sage.llm.model_card import CognitiveSystem
            from sage.llm.model_registry import ModelCardCatalog

            tier_to_cs = {
                "reasoner": CognitiveSystem.S3,
                "fast": CognitiveSystem.S2,
                "budget": CognitiveSystem.S1,
            }
            catalog_path = Path(__file__).resolve().parents[2] / "config" / "cards.toml"
            catalog = ModelCardCatalog.from_toml_file(str(catalog_path))
            candidates = [
                card for card in catalog.select_for_system(tier_to_cs.get(fallback, CognitiveSystem.S2))
                if card.id != current_model_id
            ]
            return candidates[0].id if candidates else None
        except Exception as exc:
            log.debug("fallback_tier resolution failed: %s", exc)
            return None

    def _infer_task_domain(self, task: str, node: Any, ctx: Any) -> str:
        domain = self._ctx_value(ctx, "domain", None)
        if isinstance(domain, str) and domain:
            return domain

        required_caps = getattr(node, "required_capabilities", [])
        if not isinstance(required_caps, (list, tuple, set)):
            required_caps = []
        required = {str(cap).lower() for cap in required_caps}
        role = self._coerce_str(getattr(node, "role", ""), "").lower()
        task_lower = task.lower()

        if {"code", "python", "json"} & required or any(tok in task_lower for tok in ("code", "python", "function", "bug")):
            return "code"
        if {"reasoning", "evaluation", "formal"} & required or any(tok in role for tok in ("judge", "review", "reason", "formal")):
            return "reasoning"
        if "math" in required or "equation" in task_lower or "calculate" in task_lower:
            return "math"
        return "general"

    def _open_gate(
        self,
        node_idx: int,
        topology: Any,
        parallel_outputs: list[str] | None,
    ) -> AdaptationDecision | None:
        """Open another debate round when peer outputs still materially disagree."""
        if not self._is_debate_topology(topology) or not parallel_outputs or len(parallel_outputs) < 2:
            return None

        gate_turns = self._gate_loops.get(node_idx, 0)
        if gate_turns >= self.MAX_GATE_TURNS:
            return None

        consistency = self.compute_consistency_score(parallel_outputs)
        if consistency >= self.THETA_CONSISTENCY:
            return None

        try:
            predecessors = topology.get_predecessors(node_idx)
        except Exception:
            predecessors = []
        if not predecessors:
            return None

        source = predecessors[0]
        self._gate_loops[node_idx] = gate_turns + 1
        self._emit(
            "OPEN_GATE",
            {
                "node": node_idx,
                "turn": gate_turns + 1,
                "consistency": consistency,
            },
        )
        return AdaptationDecision(
            action="open_gate",
            target_node=node_idx,
            gate_source=source,
            gate_target=node_idx,
            reason=(
                f"debate disagreement (consistency={consistency:.2f}, "
                f"turn {gate_turns + 1}/{self.MAX_GATE_TURNS})"
            ),
        )

    def compute_consistency_score(self, outputs: list[str]) -> float:
        """Mean pairwise cosine similarity of parallel outputs."""
        try:
            from sage.consistency import consistency_score
            return consistency_score(outputs, embedder=self._embedder)
        except ImportError:
            return 1.0  # no consistency module -> assume consistent

    def compute_importance_score(self, node_idx: int, result: str, all_outputs: list[str]) -> float:
        """Semantic importance: 1 - mean_similarity(this_node, others).

        High similarity to existing outputs = low marginal value = low importance.
        """
        if not all_outputs or len(all_outputs) <= 1:
            return 1.0  # single node = always important

        # Exclude by identity (not value) — if two workers return identical text,
        # we must still keep both in other_outputs for correct importance scoring.
        other_outputs = [o for o in all_outputs if o is not result]
        if not other_outputs:
            return 1.0

        try:
            from sage.consistency import consistency_score
            similarity = consistency_score([result] + other_outputs, embedder=self._embedder)
            return max(0.0, 1.0 - similarity)  # high similarity = low importance
        except ImportError:
            return 0.5  # default: assume moderate importance

    def _get_invariant_feedback(self, result: str, topology: Any, node_idx: int) -> str | None:
        """Get clause-level feedback from OxiZ for S3 nodes."""
        node = topology.get_node(node_idx) if hasattr(topology, 'get_node') else None
        if not node or getattr(node, 'system', 1) < 3:
            return None  # Only for S3 nodes
        try:
            from sage_core import SmtVerifier
            verifier = SmtVerifier()
            # Try to verify any assertions in the result
            # This is a lightweight check — not full PRM
            result_feedback = verifier.verify_invariant_with_feedback("true", result[:500])
            if hasattr(result_feedback, 'feedback') and result_feedback.feedback:
                return result_feedback.feedback
        except (ImportError, Exception):
            pass
        return None

    def _verify_arithmetic(self, result: str) -> tuple[bool, str]:
        """Verify arithmetic equations in result text.

        Catches obvious calculation errors like "5 + 3 = 9" without needing
        full OxiZ. Based on MASPRM (arXiv 2510.24803) per-step verification.
        """
        try:
            equations = re.findall(r'(\d+)\s*([+\-*/])\s*(\d+)\s*=\s*(\d+)', result)
            for a_s, op, b_s, c_s in equations:
                a, b, c = int(a_s), int(b_s), int(c_s)
                ops = {'+': a + b, '-': a - b, '*': a * b}
                if op == '/' and b != 0:
                    ops['/'] = a // b
                expected = ops.get(op)
                if expected is not None and expected != c:
                    return False, f"{a}{op}{b}={c} should be {expected}"
            return True, ""
        except Exception:
            return True, ""  # On error, don't block

    @staticmethod
    def _detect_emergent_subtask(result: str) -> str | None:
        """Detect emergent sub-tasks from node output."""
        patterns = [
            r"(?:need to also|additionally|we should also|another step would be)\s+(.{10,200})",
            r"(?:TODO|FIXME|NOTE):\s+(.{10,200})",
            r"(?:this requires|prerequisite:)\s+(.{10,200})",
        ]
        for pattern in patterns:
            match = re.search(pattern, result, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
