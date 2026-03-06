"""Benchmark truth pack — machine-auditable per-task traces."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, timeout=5
        ).strip()[:12]
    except Exception:
        return "unknown"


@dataclass
class TaskTrace:
    task_id: str
    passed: bool
    latency_ms: float
    cost_usd: float
    model: str = ""
    routing: str = ""
    error: str = ""
    seed: int = 0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v or isinstance(v, bool)}


@dataclass
class BenchmarkManifest:
    benchmark: str
    model: str
    git_sha: str = field(default_factory=_git_sha)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    traces: list[TaskTrace] = field(default_factory=list)

    def add(self, trace: TaskTrace) -> None:
        self.traces.append(trace)

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(t.to_dict()) for t in self.traces)

    def summary(self) -> dict:
        passed = sum(1 for t in self.traces if t.passed)
        return {
            "benchmark": self.benchmark,
            "model": self.model,
            "git_sha": self.git_sha,
            "total": len(self.traces),
            "passed": passed,
            "pass_rate": round(passed / max(len(self.traces), 1), 4),
            "avg_latency_ms": round(
                sum(t.latency_ms for t in self.traces) / max(len(self.traces), 1), 1
            ),
        }
