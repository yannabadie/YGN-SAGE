"""Benchmark truth pack — machine-auditable per-task traces.

Every benchmark run must record provenance (Audit5 §2):
model, provider, git_sha, feature_flags at minimum.
"""
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


def _detect_feature_flags() -> list[str]:
    """Detect which sage_core features are available at runtime."""
    flags: list[str] = []
    try:
        import sage_core
        flags.append("sage_core")
        if hasattr(sage_core, "SmtVerifier"):
            flags.append("smt")
        if hasattr(sage_core, "ToolExecutor"):
            flags.append("tool-executor")
        if hasattr(sage_core, "RustEmbedder"):
            flags.append("onnx")
        if hasattr(sage_core, "WasmSandbox"):
            flags.append("sandbox")
    except ImportError:
        pass
    return flags


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
    provider: str = ""
    git_sha: str = field(default_factory=_git_sha)
    feature_flags: list[str] = field(default_factory=_detect_feature_flags)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    traces: list[TaskTrace] = field(default_factory=list)

    def add(self, trace: TaskTrace) -> None:
        self.traces.append(trace)

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(t.to_dict()) for t in self.traces)

    def summary(self) -> dict:
        passed = sum(1 for t in self.traces if t.passed)
        total_cost = sum(t.cost_usd for t in self.traces)
        return {
            "benchmark": self.benchmark,
            "model": self.model,
            "provider": self.provider,
            "git_sha": self.git_sha,
            "feature_flags": self.feature_flags,
            "total": len(self.traces),
            "passed": passed,
            "pass_rate": round(passed / max(len(self.traces), 1), 4),
            "avg_latency_ms": round(
                sum(t.latency_ms for t in self.traces) / max(len(self.traces), 1), 1
            ),
            "total_cost_usd": round(total_cost, 6),
        }
