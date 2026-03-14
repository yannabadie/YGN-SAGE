"""Shadow routing — dual Rust/Python routing with divergence tracking.

When both Rust and Python routers are available, runs both on every task.
Logs divergences (system mismatch, model/tier mismatch) to structured JSONL.
Tracks divergence rate for Phase 5 gate: < 5% divergence on 1000+ traces.
Zero-overhead when only one router is available.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sage.constants import (
    SHADOW_SOFT_TRACES,
    SHADOW_SOFT_DIVERGENCE,
    SHADOW_HARD_TRACES,
    SHADOW_HARD_DIVERGENCE,
    SHADOW_MAX_TRACE_BYTES,
    DEFAULT_BUDGET_USD,
)

_log = logging.getLogger("sage.routing.shadow")

# Default trace file location
_DEFAULT_TRACE_PATH = Path.home() / ".sage" / "shadow_traces.jsonl"


class ShadowRouter:
    """Dual-path shadow router for Rust/Python routing comparison.

    Runs both routers on every task when both are present. Returns the Rust
    decision as primary (production path). Compares system numbers and logs
    divergences to a JSONL trace file for Phase 5 deletion gate.

    When only one router is available, acts as a zero-overhead passthrough.

    Parameters
    ----------
    rust_router:
        Rust SystemRouter instance (from sage_core), or None.
    python_metacognition:
        Python AdaptiveRouter or ComplexityRouter instance, or None.
    trace_path:
        Path for JSONL divergence trace file. Defaults to ~/.sage/shadow_traces.jsonl.
    """

    def __init__(
        self,
        rust_router: Any | None = None,
        python_metacognition: Any | None = None,
        trace_path: Path | None = None,
    ) -> None:
        self._rust_router = rust_router
        self._python_metacognition = python_metacognition
        self._trace_path = trace_path or _DEFAULT_TRACE_PATH
        self.stats: dict[str, int] = {
            "total_comparisons": 0,
            "system_mismatches": 0,
        }

    @property
    def _shadow_active(self) -> bool:
        """True when both routers are present (shadow comparison enabled)."""
        return self._rust_router is not None and self._python_metacognition is not None

    async def route(self, task: str, budget: float = DEFAULT_BUDGET_USD) -> Any:
        """Route a task, running both routers in shadow mode when available.

        Returns the primary decision:
        - Rust decision when Rust router is present (production path).
        - Python decision when only Python router is present (fallback).

        Parameters
        ----------
        task:
            The task string to route.
        budget:
            Budget in USD for the Rust router.

        Returns
        -------
        The routing decision from the primary router.
        """
        # Case 1: Both routers -- shadow comparison
        if self._shadow_active:
            return await self._route_shadow(task, budget)

        # Case 2: Rust only -- no comparison
        if self._rust_router is not None:
            return self._rust_router.route(task, budget)

        # Case 3: Python only -- passthrough
        if self._python_metacognition is not None:
            profile = await self._python_metacognition.assess_complexity_async(task)
            return self._python_metacognition.route(profile)

        # Case 4: Neither router (should not happen in practice)
        raise RuntimeError("ShadowRouter: no router available")

    async def _route_shadow(self, task: str, budget: float) -> Any:
        """Run both routers and compare. Returns Rust decision as primary."""
        # Always run Rust (primary path -- must not fail)
        rust_decision = self._rust_router.route(task, budget)

        # Run Python in shadow (failures must not affect primary path)
        python_decision = None
        try:
            profile = await self._python_metacognition.assess_complexity_async(task)
            python_decision = self._python_metacognition.route(profile)
        except Exception as exc:
            _log.warning("Shadow: Python router failed (%s), skipping comparison", exc)

        # Compare only if Python succeeded
        if python_decision is not None:
            rust_system = int(rust_decision.system)
            python_system = int(python_decision.system)
            rust_model = getattr(rust_decision, "model_id", "")
            python_tier = getattr(python_decision, "llm_tier", "")
            match = rust_system == python_system

            # Update stats
            self.stats["total_comparisons"] += 1
            if not match:
                self.stats["system_mismatches"] += 1

            # Write trace record
            self._write_trace(
                task=task,
                rust_system=rust_system,
                python_system=python_system,
                rust_model=rust_model,
                python_tier=python_tier,
                match=match,
            )

            if not match:
                _log.info(
                    "Shadow divergence: task=%r rust=S%d python=S%d (rate=%.2f%%, n=%d)",
                    task[:60], rust_system, python_system,
                    self.divergence_rate() * 100,
                    self.stats["total_comparisons"],
                )

        return rust_decision

    # Maximum trace file size before rotation
    _MAX_TRACE_BYTES = SHADOW_MAX_TRACE_BYTES

    def _write_trace(
        self,
        task: str,
        rust_system: int,
        python_system: int,
        rust_model: str,
        python_tier: str,
        match: bool,
    ) -> None:
        """Append a single trace record to the JSONL file."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_preview": task[:60],
            "rust_system": rust_system,
            "python_system": python_system,
            "rust_model": rust_model,
            "python_tier": python_tier,
            "match": match,
        }
        try:
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)
            # Rotate if file exceeds max size
            if self._trace_path.exists() and self._trace_path.stat().st_size > self._MAX_TRACE_BYTES:
                bak = self._trace_path.with_suffix(".jsonl.bak")
                self._trace_path.rename(bak)
                _log.info("Shadow: rotated trace file to %s", bak)
            with open(self._trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            _log.warning("Shadow: failed to write trace (%s)", exc)

    def divergence_rate(self) -> float:
        """Fraction of comparisons where Rust and Python routers disagreed.

        Returns 0.0 if no comparisons have been made.
        """
        total = self.stats["total_comparisons"]
        if total == 0:
            return 0.0
        return self.stats["system_mismatches"] / total

    @property
    def total(self) -> int:
        """Total number of shadow comparisons made."""
        return self.stats["total_comparisons"]

    def is_phase5_soft_ready(self) -> bool:
        """Soft gate: 500 traces, <10% divergence. Can start preferring Rust router."""
        return self.total >= SHADOW_SOFT_TRACES and self.divergence_rate() < SHADOW_SOFT_DIVERGENCE

    def is_phase5_hard_ready(self) -> bool:
        """Hard gate: 1000 traces, <5% divergence. Safe to delete Python router."""
        return self.total >= SHADOW_HARD_TRACES and self.divergence_rate() < SHADOW_HARD_DIVERGENCE

    def load_existing_traces(self) -> None:
        """Load trace counts from existing JSONL file for cross-session gate continuity."""
        if not self._trace_path.exists():
            return
        try:
            total = 0
            mismatches = 0
            with open(self._trace_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    total += 1
                    try:
                        record = json.loads(line)
                        if not record.get("match", True):
                            mismatches += 1
                    except json.JSONDecodeError:
                        continue
            self.stats["total_comparisons"] = total
            self.stats["system_mismatches"] = mismatches
            _log.info(
                "Shadow: loaded %d existing traces (%.1f%% divergence)",
                total, self.divergence_rate() * 100,
            )
        except Exception as exc:
            _log.warning("Shadow: failed to load existing traces (%s)", exc)

    def is_phase5_ready(self) -> bool:
        """Check if divergence is low enough to safely remove the Python router.

        Alias for is_phase5_hard_ready() for backward compatibility.

        Phase 5 gate requires:
        - At least 1000 shadow comparisons
        - Strictly less than 5% divergence rate
        """
        return self.is_phase5_hard_ready()
