"""
SageCandidate: base contract every Meta-Harness candidate must satisfy.

Upstream text_classification uses `MemorySystem` as the candidate
interface; terminal_bench_2 uses `AgentHarness`. For YGN-SAGE the
natural boundary is `AgentSystem` (the `system.run(task)` entry-point
reachable from `sage.boot.boot_agent_system`).

Candidates override `build_system()` to return an AgentSystem configured
with whatever variations the proposer wants to explore: custom topology,
alternate memory tier, new agent_loop_factory, different tool registry,
modified provider routing, etc.

Mutable axes (proposer CAN change):
- The returned system's pipeline, topology engine, memory tiers, provider
  pool wiring, agent_loop_factory, tool_registry composition
- Runtime parameters passed through `system.run(task, **kwargs)`
- Pre- and post-processing around the system call (wrappers)

Out of scope (fixed by evaluation harness):
- The base model tier for S3 reasoning (cost budget enforced at bench level)
- The benchmark dataset and scoring metric
- The SAGE core (sage-core Rust crate) — candidates work with the compiled
  binary as-is; Rust code changes need a separate engineering cycle
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SageCandidate(ABC):
    """A single harness candidate evaluated by Meta-Harness.

    Concrete subclasses live under `agents/<id>.py` and are imported by
    `meta_harness.py` at evaluation time. The outer loop calls
    `candidate.build_system()`, then evaluates the returned system
    against the fixed benchmark via `benchmark.py`.
    """

    name: str = "unnamed"
    hypothesis: str = ""
    axis: str = ""  # e.g. "topology", "memory", "routing", "tools"

    @abstractmethod
    def build_system(self, hints: dict[str, Any] | None = None) -> Any:
        """Return a configured SAGE AgentSystem.

        The returned object must expose:
          - `async run(task: str, **kwargs) -> str` — the standard
            single-entry-point call
          - `pipeline` attribute (optional but preferred) — reachable
            CognitiveOrchestrationPipeline
          - `_last_execution_path` (optional) — for trace logging

        Parameters
        ----------
        hints : dict, optional
            Evaluation-time hints. Benchmark passes things like
            `{"seed": 42, "task_domain": "code", "max_cost_usd": 10.0}`.
            Candidates may ignore unknown hints.
        """
        raise NotImplementedError

    def describe(self) -> dict[str, str]:
        """Self-description for logs and leaderboard."""
        return {
            "name": self.name,
            "hypothesis": self.hypothesis,
            "axis": self.axis,
        }


def load_candidate(module_name: str) -> SageCandidate:
    """Import a candidate module and return its SageCandidate instance.

    Convention: the module exposes a top-level symbol `CANDIDATE` that is
    an instance of SageCandidate. If not found, we look for the first
    subclass of SageCandidate defined in the module and instantiate it.
    """
    import importlib
    mod = importlib.import_module(module_name)
    candidate = getattr(mod, "CANDIDATE", None)
    if isinstance(candidate, SageCandidate):
        return candidate
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, type) and issubclass(obj, SageCandidate) and obj is not SageCandidate:
            return obj()
    raise ImportError(
        f"module {module_name!r} defines no SageCandidate (no CANDIDATE symbol "
        f"and no SageCandidate subclass)",
    )
