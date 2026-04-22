"""Benchmark data structures and report generation."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _discover_git_sha() -> str:
    """P2.8: auto-detect current HEAD for benchmark attribution.

    Returns the 40-char SHA of HEAD or empty string if git isn't available
    or this isn't a repo. Never raises. Called at report-build time so the
    sha reflects the code that actually produced the numbers.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return ""


def _discover_feature_flags() -> list[str]:
    """P2.8: probe which sage_core features were compiled in.

    Feature flags affect the truth of a benchmark number (e.g. SMT off
    means the QualityLabeler ran in heuristic-only mode, which changes
    learning updates). We record which features are available at run
    time so reports survive post-hoc auditing.
    """
    flags: list[str] = []
    try:
        import sage_core

        for probe in (
            "WasmSandbox",     # sandbox feature
            "ToolExecutor",    # tool-executor feature
            "QualityLabeler",  # smt feature (requires oxiz)
            "KnnRouter",       # base
            "TopologyEngine",  # base
        ):
            if hasattr(sage_core, probe):
                flags.append(probe.lower())
    except ImportError:
        flags.append("no_sage_core")
    # Audit-important env vars that materially change runtime behaviour
    for env in ("SAGE_UNSAFE_RAW_EXEC", "SAGE_CHAT_ALLOW_BASH", "SAGE_ENABLE_PATH6"):
        val = os.environ.get(env, "").strip().lower()
        if val in {"1", "true", "yes", "on"}:
            flags.append(f"env:{env.lower()}=on")
    return flags


@dataclass
class TaskResult:
    """Result of a single benchmark task."""

    task_id: str
    passed: bool
    system_used: int = 0        # 1, 2, or 3
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    sandbox_executions: int = 0
    memory_events: int = 0
    escalations: int = 0
    z3_checks: int = 0
    tokens_used: int = 0
    error: str = ""


@dataclass
class BenchReport:
    """Aggregated benchmark report."""

    benchmark: str
    total: int
    passed: int
    failed: int
    errors: int
    pass_rate: float
    avg_latency_ms: float
    avg_cost_usd: float
    routing_breakdown: dict[str, int]   # {"S1": n, "S2": n, "S3": n}
    results: list[TaskResult]
    model_config: dict[str, Any] = field(default_factory=dict)
    model: str = "unknown"
    provider: str = ""
    git_sha: str = ""
    feature_flags: list[str] = field(default_factory=list)
    timestamp: str = ""
    temperature: float = 0.0

    @staticmethod
    def from_results(
        benchmark: str,
        results: list[TaskResult],
        model_config: dict[str, Any] | None = None,
    ) -> BenchReport:
        """Build a report by aggregating a list of TaskResult objects."""
        total = len(results)
        if total == 0:
            return BenchReport(
                benchmark=benchmark,
                total=0,
                passed=0,
                failed=0,
                errors=0,
                pass_rate=0.0,
                avg_latency_ms=0.0,
                avg_cost_usd=0.0,
                routing_breakdown={"S1": 0, "S2": 0, "S3": 0},
                results=[],
                model_config=model_config or {},
                model=model_config.get("model", "unknown") if model_config else "unknown",
                provider=model_config.get("provider", "") if model_config else "",
                git_sha=_discover_git_sha(),
                feature_flags=_discover_feature_flags(),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        errors = sum(1 for r in results if r.error)
        pass_rate = passed / total
        avg_latency = sum(r.latency_ms for r in results) / total
        avg_cost = sum(r.cost_usd for r in results) / total

        breakdown: dict[str, int] = {"S1": 0, "S2": 0, "S3": 0}
        for r in results:
            key = f"S{r.system_used}"
            if key in breakdown:
                breakdown[key] += 1

        return BenchReport(
            benchmark=benchmark,
            total=total,
            passed=passed,
            failed=failed,
            errors=errors,
            pass_rate=pass_rate,
            avg_latency_ms=avg_latency,
            avg_cost_usd=avg_cost,
            routing_breakdown=breakdown,
            results=results,
            model_config=model_config or {},
            model=model_config.get("model", "unknown") if model_config else "unknown",
            provider=model_config.get("provider", "") if model_config else "",
            # P2.8 (2026-04-22 audit remediation): auto-capture reproducibility
            # metadata at report-build time. git_sha + feature_flags make every
            # bench number post-hoc auditable — external reviewers can re-check
            # which code + which compiled features produced the result.
            git_sha=_discover_git_sha(),
            feature_flags=_discover_feature_flags(),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


class BenchmarkRunner:
    """Thin orchestrator for running benchmark suites."""

    def __init__(self) -> None:
        self._suites: dict[str, Any] = {}

    def register(self, name: str, suite: Any) -> None:
        """Register a benchmark suite by name."""
        self._suites[name] = suite

    async def run(self, name: str) -> BenchReport:
        """Run a registered benchmark suite and return the report."""
        suite = self._suites[name]
        return await suite.run()

    @property
    def available(self) -> list[str]:
        return list(self._suites.keys())
