"""Benchmark data structures and report generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


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
