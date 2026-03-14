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
    action: str  # "continue", "upgrade_model", "prune_node", "reroute_topology", "spawn_subagent"
    target_node: int | None = None
    reason: str = ""
    new_model_id: str | None = None
    invariant_feedback: str | None = None  # clause-level from OxiZ


# Regex for detecting structured reasoning content
_STRUCTURED_CONTENT = re.compile(r'<think>|```|assert\s|def\s+test_|proof:|invariant:', re.IGNORECASE)


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
        self._reroute_count = 0
        self._spawn_count = 0

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
        result: str,
        task: str,
        topology: Any,
        ctx: Any,
        parallel_outputs: list[str] | None = None,
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
        # Early exit: max reroute hit
        if self._reroute_count >= self.MAX_REROUTES:
            if parallel_outputs and self.compute_consistency_score(parallel_outputs) < self.THETA_CONSISTENCY:
                self._emit("MAX_REROUTE_HIT", {"node": node_idx, "reroute_count": self._reroute_count})
                log.warning("Max reroute limit reached (count=%d), forcing continue", self._reroute_count)

        # Compute quality (80% heuristic + 20% PRM if structured content)
        quality = self._compute_quality(node_idx, result, task, ctx)
        self._node_qualities[node_idx] = quality

        # Decision cascade
        # 1. Good quality -> continue
        if quality >= self.THETA_GOOD:
            return AdaptationDecision(action="continue", target_node=node_idx)

        # 2. Critical quality -> upgrade model
        retries = self._node_retries.get(node_idx, 0)
        if quality < self.THETA_CRITICAL and retries < self.MAX_RETRIES:
            self._node_retries[node_idx] = retries + 1
            # Get invariant feedback for S3 nodes if available
            feedback = self._get_invariant_feedback(result, topology, node_idx)
            return AdaptationDecision(
                action="upgrade_model",
                target_node=node_idx,
                reason=f"quality={quality:.2f} < {self.THETA_CRITICAL}",
                invariant_feedback=feedback,
            )

        # 3. Parallel inconsistency -> reroute topology
        if parallel_outputs and self._reroute_count < self.MAX_REROUTES:
            consistency = self.compute_consistency_score(parallel_outputs)
            if consistency < self.THETA_CONSISTENCY:
                self._reroute_count += 1
                self._emit("REROUTE_TOPOLOGY", {"consistency": consistency, "node": node_idx})
                return AdaptationDecision(
                    action="reroute_topology",
                    target_node=node_idx,
                    reason=f"consistency={consistency:.2f} < {self.THETA_CONSISTENCY}",
                )

        # 4. Low importance -> prune
        if parallel_outputs:
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
        """80% QualityEstimator + 20% PRM (if structured content detected)."""
        heuristic = 0.5
        if self._qe:
            try:
                latency = getattr(ctx, 'latency_ms', 0.0)
                heuristic = self._qe.estimate(task, result, latency)
            except Exception:
                pass

        # PRM only for structured content (guard: -1.0 on plain text)
        if self._prm and _STRUCTURED_CONTENT.search(result):
            try:
                r_path, _ = self._prm.calculate_r_path(result)
                if r_path >= 0.0:  # valid PRM score
                    return 0.8 * heuristic + 0.2 * r_path
            except Exception as exc:
                log.debug("PRM scoring failed: %s", exc)

        return heuristic

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

        other_outputs = [o for i, o in enumerate(all_outputs) if o != result]
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
