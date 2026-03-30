"""Extended drift monitor — 12-dimension Agent Stability Index.

Adds 9 new dimensions to the base DriftMonitor's 3 signals (latency, error, cost):

  4. Semantic drift — embedding distance between consecutive outputs
  5. Behavioral consistency — action sequence variance (Levenshtein)
  6. Topic coherence — keyword overlap between task and response
  7. Reasoning depth — chain-of-thought length trend
  8. Memory utilization — S-MMU retrieval hit rate trend
  9. Tool diversity — Shannon entropy of tool usage distribution
 10. Output stability — coefficient of variation of response lengths
 11. Confidence trend — write gate confidence trend
 12. Coordination stability — sub-agent spawn/complete ratio

Research basis:
- Agent Drift (2601.04170): 12-dimension ASI, episodic consolidation mitigates drift
- Behavioral Consistency (2602.11619): action variance predicts failure (80-92% vs 25-60%)
"""
from __future__ import annotations

import math
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from sage.constants import (
    ASI_WEIGHT_SEMANTIC,
    ASI_WEIGHT_BEHAVIORAL,
    ASI_WEIGHT_TOPIC,
    ASI_WEIGHT_REASONING_DEPTH,
    ASI_WEIGHT_MEMORY_UTIL,
    ASI_WEIGHT_TOOL_DIVERSITY,
    ASI_WEIGHT_OUTPUT_STABILITY,
    ASI_WEIGHT_CONFIDENCE_TREND,
    ASI_WEIGHT_COORDINATION,
    ASI_BEHAVIORAL_WINDOW,
    DRIFT_ACTION_CONTINUE,
    DRIFT_ACTION_SWITCH,
)
from sage.monitoring.drift import DriftMonitor, DriftReport

log = logging.getLogger(__name__)


# ── Behavior Tracker ──────────────────────────────────────────────────────────

class BehaviorTracker:
    """Tracks action sequence consistency across steps.

    Research: arXiv 2602.11619 — consistent agents (score > 0.7) achieve
    80-92% accuracy vs 25-60% for inconsistent agents.
    """

    def __init__(self, window: int = ASI_BEHAVIORAL_WINDOW) -> None:
        self._action_sequences: deque[list[str]] = deque(maxlen=window)

    def record_actions(self, actions: list[str]) -> None:
        """Record the action sequence from one step."""
        self._action_sequences.append(actions)

    def consistency_score(self) -> float:
        """Compute behavioral consistency (1.0 = perfectly consistent, 0.0 = chaotic).

        Uses normalized Levenshtein distance between consecutive action sequences.
        """
        if len(self._action_sequences) < 2:
            return 1.0  # Not enough data → assume stable

        distances = []
        seqs = list(self._action_sequences)
        for i in range(len(seqs) - 1):
            d = _normalized_levenshtein(seqs[i], seqs[i + 1])
            distances.append(d)

        mean_distance = sum(distances) / len(distances)
        return max(0.0, 1.0 - mean_distance)

    @property
    def sequence_count(self) -> int:
        return len(self._action_sequences)


def _normalized_levenshtein(a: list[str], b: list[str]) -> float:
    """Normalized Levenshtein distance between two string lists. Returns 0-1."""
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    # Simple edit distance via DP
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(dp[j], dp[j - 1], prev)
            prev = temp
    return dp[n] / max_len


# ── Extended Drift Monitor ────────────────────────────────────────────────────

@dataclass
class ExtendedDriftReport(DriftReport):
    """DriftReport with 12-dimension breakdown."""

    asi_score: float = 0.0
    extended_signals: dict[str, float] = field(default_factory=dict)


class ExtendedDriftMonitor:
    """12-dimension Agent Stability Index (ASI) drift monitor.

    Wraps the base DriftMonitor (3 legacy signals) and adds 9 new
    dimensions from event metadata.
    """

    def __init__(self) -> None:
        self._base = DriftMonitor()
        self._behavior_tracker = BehaviorTracker()
        self._prev_embedding: list[float] | None = None
        self._output_lengths: deque[int] = deque(maxlen=50)
        self._confidence_values: deque[float] = deque(maxlen=50)

    @property
    def behavior_tracker(self) -> BehaviorTracker:
        return self._behavior_tracker

    def analyze(self, events: list[Any], task: str = "") -> ExtendedDriftReport:
        """Compute 12-dimension ASI from events and accumulated state."""
        # Base 3-signal analysis
        base_report = self._base.analyze(events)

        if not events or len(events) < 2:
            return ExtendedDriftReport(
                drift_score=base_report.drift_score,
                action=base_report.action,
                details=base_report.details,
                asi_score=0.0,
            )

        # Extract extended signals from event metadata
        signals: dict[str, float] = {}

        # Signal 4: Semantic drift (embedding distance between consecutive outputs)
        signals["semantic"] = self._semantic_drift(events)

        # Signal 5: Behavioral consistency
        signals["behavioral"] = 1.0 - self._behavior_tracker.consistency_score()

        # Signal 6: Topic coherence (keyword overlap between task and response)
        signals["topic"] = self._topic_drift(events, task)

        # Signal 7: Reasoning depth trend
        signals["reasoning_depth"] = self._reasoning_depth_trend(events)

        # Signal 8: Memory utilization
        signals["memory_util"] = self._memory_util_drift(events)

        # Signal 9: Tool diversity (Shannon entropy)
        signals["tool_diversity"] = self._tool_diversity(events)

        # Signal 10: Output length stability (coefficient of variation)
        signals["output_stability"] = self._output_stability(events)

        # Signal 11: Confidence trend
        signals["confidence_trend"] = self._confidence_trend(events)

        # Signal 12: Coordination stability
        signals["coordination"] = self._coordination_drift(events)

        # Composite ASI score (weighted sum of extended signals)
        asi_extended = (
            ASI_WEIGHT_SEMANTIC * signals["semantic"]
            + ASI_WEIGHT_BEHAVIORAL * signals["behavioral"]
            + ASI_WEIGHT_TOPIC * signals["topic"]
            + ASI_WEIGHT_REASONING_DEPTH * signals["reasoning_depth"]
            + ASI_WEIGHT_MEMORY_UTIL * signals["memory_util"]
            + ASI_WEIGHT_TOOL_DIVERSITY * signals["tool_diversity"]
            + ASI_WEIGHT_OUTPUT_STABILITY * signals["output_stability"]
            + ASI_WEIGHT_CONFIDENCE_TREND * signals["confidence_trend"]
            + ASI_WEIGHT_COORDINATION * signals["coordination"]
        )

        # Combined score: base (3 signals) + extended (9 signals)
        # Base report already has catastrophic factor applied
        combined = base_report.drift_score * 0.62 + asi_extended
        combined = min(1.0, max(0.0, combined))

        if combined > DRIFT_ACTION_SWITCH:
            action = "RESET_AGENT"
        elif combined > DRIFT_ACTION_CONTINUE:
            action = "SWITCH_MODEL"
        else:
            action = "CONTINUE"

        # Merge all signal details
        all_details = dict(base_report.details or {})
        all_details.update({f"asi_{k}": round(v, 4) for k, v in signals.items()})

        return ExtendedDriftReport(
            drift_score=round(combined, 3),
            action=action,
            details=all_details,
            asi_score=round(asi_extended, 4),
            extended_signals=signals,
        )

    # ── Signal extractors ─────────────────────────────────────────────────

    def _semantic_drift(self, events: list[Any]) -> float:
        """Embedding distance between consecutive outputs."""
        embeddings = []
        for e in events:
            meta = getattr(e, "meta", {}) or {}
            emb = meta.get("embedding")
            if emb and isinstance(emb, (list, tuple)) and len(emb) > 0:
                embeddings.append(emb)

        if len(embeddings) < 2:
            return 0.0

        distances = []
        for i in range(len(embeddings) - 1):
            d = 1.0 - _cosine_similarity(embeddings[i], embeddings[i + 1])
            distances.append(max(0.0, d))

        return min(1.0, sum(distances) / len(distances))

    @staticmethod
    def _topic_drift(events: list[Any], task: str) -> float:
        """Low keyword overlap between task and recent outputs = drift."""
        if not task:
            return 0.0
        import re
        task_words = set(re.findall(r'\b[a-z]{3,}\b', task.lower()))
        if not task_words:
            return 0.0

        overlaps = []
        for e in events[-5:]:
            content = getattr(e, "content", "") or ""
            output_words = set(re.findall(r'\b[a-z]{3,}\b', content.lower()))
            if output_words:
                overlap = len(task_words & output_words) / len(task_words)
                overlaps.append(overlap)

        if not overlaps:
            return 0.0
        # Low overlap = high drift
        return max(0.0, 1.0 - (sum(overlaps) / len(overlaps)))

    @staticmethod
    def _reasoning_depth_trend(events: list[Any]) -> float:
        """Declining reasoning depth (content length) indicates drift."""
        lengths = []
        for e in events:
            content = getattr(e, "content", "") or ""
            lengths.append(len(content))

        if len(lengths) < 4:
            return 0.0

        mid = len(lengths) // 2
        first_avg = sum(lengths[:mid]) / max(mid, 1)
        second_avg = sum(lengths[mid:]) / max(len(lengths) - mid, 1)

        if first_avg <= 0:
            return 0.0
        # Declining length = drift
        ratio = second_avg / first_avg
        if ratio >= 1.0:
            return 0.0
        return min(1.0, 1.0 - ratio)

    @staticmethod
    def _memory_util_drift(events: list[Any]) -> float:
        """Declining memory retrieval hit rate indicates drift."""
        hits = []
        for e in events:
            meta = getattr(e, "meta", {}) or {}
            hit = meta.get("smmu_hit")
            if hit is not None:
                hits.append(1.0 if hit else 0.0)

        if len(hits) < 4:
            return 0.0

        mid = len(hits) // 2
        first_rate = sum(hits[:mid]) / max(mid, 1)
        second_rate = sum(hits[mid:]) / max(len(hits) - mid, 1)

        # Declining hit rate = drift
        decline = max(0.0, first_rate - second_rate)
        return min(1.0, decline * 2.0)  # Scale: 50% decline -> 1.0

    @staticmethod
    def _tool_diversity(events: list[Any]) -> float:
        """Low tool usage entropy (using same tool repeatedly) may indicate stuck agent."""
        tool_counts: dict[str, int] = {}
        for e in events:
            meta = getattr(e, "meta", {}) or {}
            tool = meta.get("tool")
            if tool:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1

        if not tool_counts:
            return 0.0

        total = sum(tool_counts.values())
        entropy = 0.0
        for count in tool_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize: max entropy = log2(n_tools)
        max_entropy = math.log2(len(tool_counts)) if len(tool_counts) > 1 else 1.0
        normalized = entropy / max_entropy if max_entropy > 0 else 0.0

        # Low diversity = drift (agent is stuck on one tool)
        return max(0.0, 1.0 - normalized) if len(tool_counts) > 1 else 0.0

    def _output_stability(self, events: list[Any]) -> float:
        """High coefficient of variation in output lengths = instability."""
        for e in events:
            content = getattr(e, "content", "") or ""
            self._output_lengths.append(len(content))

        if len(self._output_lengths) < 3:
            return 0.0

        lengths = list(self._output_lengths)
        mean = sum(lengths) / len(lengths)
        if mean <= 0:
            return 0.0
        variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
        cv = math.sqrt(variance) / mean

        # CV > 1.0 is very unstable
        return min(1.0, cv)

    def _confidence_trend(self, events: list[Any]) -> float:
        """Declining write gate confidence indicates quality drift."""
        for e in events:
            meta = getattr(e, "meta", {}) or {}
            conf = meta.get("write_confidence")
            if conf is not None:
                self._confidence_values.append(float(conf))

        if len(self._confidence_values) < 4:
            return 0.0

        vals = list(self._confidence_values)
        mid = len(vals) // 2
        first_avg = sum(vals[:mid]) / max(mid, 1)
        second_avg = sum(vals[mid:]) / max(len(vals) - mid, 1)

        decline = max(0.0, first_avg - second_avg)
        return min(1.0, decline * 3.0)  # Scale: 33% decline -> 1.0

    @staticmethod
    def _coordination_drift(events: list[Any]) -> float:
        """Imbalanced sub-agent spawn/complete ratio = coordination drift."""
        spawns = 0
        completes = 0
        for e in events:
            meta = getattr(e, "meta", {}) or {}
            if meta.get("sub_agent_spawn"):
                spawns += 1
            if meta.get("sub_agent_complete"):
                completes += 1

        if spawns == 0:
            return 0.0

        ratio = completes / spawns
        # Perfect = 1.0 ratio. Drift = deviation from 1.0
        return min(1.0, abs(1.0 - ratio))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1e-9
    norm_b = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (norm_a * norm_b)
