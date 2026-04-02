"""Meta-Harness search loop: propose -> evaluate -> log -> repeat.

Implements the outer loop from Lee et al. (arXiv 2603.28052):
1. Agent reads filesystem with all prior candidates' code, scores, traces
2. Proposes a new HarnessConfig
3. Evaluates on search-set (MASBENCH subset)
4. Logs everything to filesystem
5. Repeat

The proposer is Claude Code (invoked by the user), not an automated LLM call.
This matches Meta-Harness's design: the proposer needs filesystem access
and selective inspection via grep/cat, not a single-pass prompt.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sage.meta_harness.config import HarnessConfig

log = logging.getLogger(__name__)

DEFAULT_WORKSPACE = Path.home() / ".sage-meta-harness"


@dataclass
class CandidateResult:
    """Evaluation result for a single candidate harness."""

    candidate_id: str
    config: HarnessConfig
    scores: dict[str, float]
    aggregate_score: float
    aggregate_pass_rate: float
    token_usage: int
    total_latency_ms: float
    evaluated_at: str
    traces: list[dict[str, Any]] = field(default_factory=list)


class MetaHarnessLoop:
    """Outer-loop system that searches over harness configurations.

    Filesystem layout:
        workspace/
        +-- baseline/
        |   +-- config.json
        |   +-- scores.json
        +-- candidates/
        |   +-- 001/
        |   |   +-- config.json     # HarnessConfig
        |   |   +-- scores.json     # Per-task scores
        |   |   +-- traces.jsonl    # Full execution traces
        |   |   +-- diff.json       # Diff vs parent
        |   |   +-- proposal.md     # Proposer reasoning
        |   +-- ...
        +-- leaderboard.json
        +-- PROPOSER_INSTRUCTIONS.md
    """

    def __init__(self, workspace: Path | None = None) -> None:
        self.workspace = workspace or DEFAULT_WORKSPACE
        self.candidates_dir = self.workspace / "candidates"
        self.baseline_dir = self.workspace / "baseline"
        self.leaderboard_path = self.workspace / "leaderboard.json"

    def init_workspace(self) -> None:
        """Create workspace with baseline config and proposer instructions."""
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.candidates_dir.mkdir(exist_ok=True)
        self.baseline_dir.mkdir(exist_ok=True)

        baseline = HarnessConfig(id="baseline", description="SAGE defaults (hardcoded)")
        baseline.save(self.baseline_dir / "config.json")

        self._save_leaderboard([])
        self._write_proposer_instructions()

        log.info("Meta-Harness workspace initialized at %s", self.workspace)

    def next_candidate_id(self) -> str:
        """Get next sequential candidate ID."""
        existing = sorted(self.candidates_dir.iterdir()) if self.candidates_dir.exists() else []
        nums = []
        for d in existing:
            if d.is_dir():
                try:
                    nums.append(int(d.name))
                except ValueError:
                    pass
        return f"{max(nums, default=0) + 1:03d}"

    # ── Evaluation ──────────────────────────────────────────────────────

    async def evaluate_candidate(
        self,
        candidate_id: str,
        bench_type: str = "masbench",
        axis: str = "depth",
        limit: int = 20,
    ) -> CandidateResult:
        """Evaluate a candidate harness on the search set."""
        candidate_dir = self.candidates_dir / candidate_id
        config_path = candidate_dir / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"No config at {config_path}")

        config = HarnessConfig.load(config_path)
        log.info("Evaluating candidate %s: %s", candidate_id, config.description)

        # Boot SAGE
        from sage.boot import boot_agent_system
        system = await boot_agent_system()

        # Patch pipeline
        from sage.meta_harness.patcher import HarnessPatcher
        patcher = HarnessPatcher(config)
        if system.pipeline:
            patcher.patch_pipeline(system.pipeline)

        # Run benchmark with tracing
        t0 = time.monotonic()
        scores: dict[str, float] = {}
        traces: list[dict[str, Any]] = []

        if bench_type == "masbench":
            await self._eval_masbench(system, config, patcher, axis, limit, scores, traces)
        elif bench_type == "bigcodebench":
            await self._eval_bigcodebench(system, config, patcher, limit, scores, traces)
        else:
            raise ValueError(f"Unknown bench_type: {bench_type}")

        total_latency_ms = (time.monotonic() - t0) * 1000

        # Aggregates
        vals = list(scores.values())
        aggregate_score = sum(vals) / len(vals) if vals else 0.0
        pass_rate = sum(1 for s in vals if s > 0.5) / len(vals) if vals else 0.0

        result = CandidateResult(
            candidate_id=candidate_id,
            config=config,
            scores=scores,
            aggregate_score=aggregate_score,
            aggregate_pass_rate=pass_rate,
            token_usage=0,
            total_latency_ms=total_latency_ms,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            traces=traces,
        )

        # Save to filesystem
        self._save_results(candidate_dir, result)
        self._update_leaderboard(result)
        self._save_diff(candidate_dir, config)

        log.info(
            "Candidate %s: score=%.3f pass_rate=%.1f%% latency=%.0fms",
            candidate_id, aggregate_score, pass_rate * 100, total_latency_ms,
        )
        return result

    async def _eval_masbench(
        self, system: Any, config: HarnessConfig, patcher: Any,
        axis: str, limit: int,
        scores: dict[str, float], traces: list[dict[str, Any]],
    ) -> None:
        """Run MASBENCH with harness-patched pipeline."""
        from sage.bench.masbench import MASBenchmark

        bench = MASBenchmark(system=system, axis=axis)

        # Wrap pipeline.run to capture traces
        if system.pipeline:
            original_run = system.pipeline.run

            async def _traced_run(task: str, budget_usd: float = 10.0) -> str:
                result = await original_run(task, budget_usd)
                traces.append({
                    "task": task[:300],
                    "result_preview": (result or "")[:500],
                    "result_length": len(result) if result else 0,
                    "config_id": config.id,
                    "timestamp": time.time(),
                })
                return result

            system.pipeline.run = _traced_run

        report = await bench.run(limit=limit)

        for r in report.results:
            scores[r.task_id] = 1.0 if r.passed else 0.0
            # Enrich trace with per-task metadata
            traces.append({
                "type": "task_result",
                "task_id": r.task_id,
                "passed": r.passed,
                "system_used": r.system_used,
                "latency_ms": r.latency_ms,
                "cost_usd": r.cost_usd,
                "error": r.error,
            })

        if system.pipeline and hasattr(system.pipeline, 'run'):
            system.pipeline.run = original_run  # type: ignore[possibly-undefined]

    async def _eval_bigcodebench(
        self, system: Any, config: HarnessConfig, patcher: Any,
        limit: int,
        scores: dict[str, float], traces: list[dict[str, Any]],
    ) -> None:
        """Run BigCodeBench with harness-patched pipeline."""
        from sage.bench.bigcodebench_bench import BigCodeBenchRunner

        runner = BigCodeBenchRunner(system=system, subset="hard", split="instruct")
        report = await runner.run(limit=limit)

        for r in report.results:
            scores[r.task_id] = 1.0 if r.passed else 0.0
            traces.append({
                "type": "task_result",
                "task_id": r.task_id,
                "passed": r.passed,
                "latency_ms": r.latency_ms,
                "error": r.error,
            })

    # ── Evaluate baseline ───────────────────────────────────────────────

    async def evaluate_baseline(
        self, bench_type: str = "masbench", axis: str = "depth", limit: int = 20,
    ) -> CandidateResult:
        """Evaluate the baseline (unmodified SAGE) and store results."""
        baseline_config = HarnessConfig.load(self.baseline_dir / "config.json")
        baseline_config.id = "baseline"

        # Temporarily copy baseline config to candidates dir
        baseline_candidate = self.candidates_dir / "baseline"
        baseline_candidate.mkdir(exist_ok=True)
        baseline_config.save(baseline_candidate / "config.json")

        return await self.evaluate_candidate(
            "baseline", bench_type=bench_type, axis=axis, limit=limit,
        )

    # ── Filesystem operations ───────────────────────────────────────────

    def _save_results(self, candidate_dir: Path, result: CandidateResult) -> None:
        candidate_dir.mkdir(parents=True, exist_ok=True)

        (candidate_dir / "scores.json").write_text(
            json.dumps({
                "aggregate_score": result.aggregate_score,
                "aggregate_pass_rate": result.aggregate_pass_rate,
                "token_usage": result.token_usage,
                "total_latency_ms": result.total_latency_ms,
                "evaluated_at": result.evaluated_at,
                "per_task": result.scores,
            }, indent=2),
            encoding="utf-8",
        )

        with (candidate_dir / "traces.jsonl").open("w", encoding="utf-8") as f:
            for trace in result.traces:
                f.write(json.dumps(trace, default=str) + "\n")

    def _save_diff(self, candidate_dir: Path, config: HarnessConfig) -> None:
        if not config.parent_id:
            return
        if config.parent_id == "baseline":
            parent_path = self.baseline_dir / "config.json"
        else:
            parent_path = self.candidates_dir / config.parent_id / "config.json"
        if parent_path.exists():
            parent = HarnessConfig.load(parent_path)
            diff = config.diff(parent)
            (candidate_dir / "diff.json").write_text(
                json.dumps(diff, indent=2, default=str), encoding="utf-8",
            )

    def _update_leaderboard(self, result: CandidateResult) -> None:
        lb = self._load_leaderboard()
        entry = {
            "candidate_id": result.candidate_id,
            "description": result.config.description,
            "aggregate_score": result.aggregate_score,
            "aggregate_pass_rate": result.aggregate_pass_rate,
            "token_usage": result.token_usage,
            "total_latency_ms": result.total_latency_ms,
            "evaluated_at": result.evaluated_at,
            "parent_id": result.config.parent_id,
        }
        lb = [e for e in lb if e["candidate_id"] != result.candidate_id]
        lb.append(entry)
        lb.sort(key=lambda x: x["aggregate_score"], reverse=True)
        self._save_leaderboard(lb)

    def _load_leaderboard(self) -> list[dict[str, Any]]:
        if self.leaderboard_path.exists():
            return json.loads(self.leaderboard_path.read_text(encoding="utf-8"))
        return []

    def _save_leaderboard(self, lb: list[dict[str, Any]]) -> None:
        self.leaderboard_path.write_text(
            json.dumps(lb, indent=2, default=str), encoding="utf-8",
        )

    def status(self) -> str:
        """Human-readable leaderboard."""
        lb = self._load_leaderboard()
        if not lb:
            return "No candidates evaluated yet.\nRun: python -m sage.meta_harness evaluate <id>"

        lines = [
            "=" * 72,
            f" {'ID':>7}  {'Description':<36} {'Score':>6}  {'Pass%':>6}  {'Latency':>8}",
            "-" * 72,
        ]
        for e in lb:
            lines.append(
                f" {e['candidate_id']:>7}  "
                f"{e['description'][:36]:<36} "
                f"{e['aggregate_score']:6.3f}  "
                f"{e['aggregate_pass_rate']*100:5.1f}%  "
                f"{e['total_latency_ms']:7.0f}ms"
            )
        lines.append("=" * 72)
        return "\n".join(lines)

    def best_config(self) -> HarnessConfig | None:
        """Return the best-scoring HarnessConfig."""
        lb = self._load_leaderboard()
        if not lb:
            return None
        best_id = lb[0]["candidate_id"]
        if best_id == "baseline":
            path = self.baseline_dir / "config.json"
        else:
            path = self.candidates_dir / best_id / "config.json"
        return HarnessConfig.load(path) if path.exists() else None

    # ── Proposer instructions ───────────────────────────────────────────

    def _write_proposer_instructions(self) -> None:
        instructions = r'''# Meta-Harness Proposer Instructions (for Claude Code)

You are the proposer in a Meta-Harness search loop (Lee et al., arXiv 2603.28052).
Goal: propose HarnessConfig candidates that improve SAGE's benchmark scores.

## Quick Start

```bash
# 1. Diagnose: read leaderboard and traces
cat ~/.sage-meta-harness/leaderboard.json
cat ~/.sage-meta-harness/candidates/001/scores.json
grep -r '"passed": false' ~/.sage-meta-harness/candidates/*/traces.jsonl
cat ~/.sage-meta-harness/candidates/001/traces.jsonl | head -30

# 2. Generate template for next candidate
python -m sage.meta_harness propose

# 3. Edit the generated config.json based on diagnosis

# 4. Evaluate
python -m sage.meta_harness evaluate 002 --bench masbench --axis depth --limit 20

# 5. Check leaderboard
python -m sage.meta_harness status
```

## Search Space (what you can tune)

### context (ContextConfig)
| Parameter | Default | What it controls |
|-----------|---------|------------------|
| predecessor_format | `"[{role}]: {text}"` | How each predecessor output is formatted |
| predecessor_separator | `"\n\n"` | Separator between predecessor outputs |
| injection_template | `"Context from previous agents:\n{context}"` | System message wrapping all context |
| budget_ratio | 0.70 | Fraction of context window for predecessors |
| budget_floor_chars | 1000 | Minimum chars per predecessor |
| similarity_threshold | 0.90 | Cosine threshold for dedup gate |
| overflow_strategy | "summarize" | "summarize" / "truncate" / "hierarchical" |

### prompts (PromptConfig)
| Parameter | Default | What it controls |
|-----------|---------|------------------|
| default_template | `"You are acting as: {role}."` | System prompt when no custom prompt |
| capability_template | `" Your capabilities: {caps}."` | Appended when node has capabilities |
| global_prefix | `""` | Prepended to ALL system prompts |
| global_suffix | `""` | Appended to ALL system prompts |
| role_overrides | `{}` | Per-role custom prompts |

### execution (ExecutionConfig)
| Parameter | Default | What it controls |
|-----------|---------|------------------|
| quality_cascade_threshold | 0.30 | FrugalGPT retry trigger |
| cascade_budget_multiplier | 1.50 | Budget escalation on retry |
| node_timeout_s | 60.0 | Per-node timeout |
| max_debate_rounds | 3 | Max multi-turn debate iterations |
| compression_prompt | (summarize) | Prompt for context overflow compression |

### topology (TopologyConfig)
| Parameter | Default | What it controls |
|-----------|---------|------------------|
| omega_parallel_threshold | 3 | DAG parallelism threshold |
| delta_deep_threshold | 4 | DAG depth threshold |
| domain_template_overrides | {} | Force template per domain |

## Diagnosis Patterns

```bash
# Find which tasks consistently fail across candidates
for d in ~/.sage-meta-harness/candidates/*/; do
  echo "=== $(basename $d) ==="
  grep '"passed": false' "$d/traces.jsonl" | wc -l
done

# Compare two candidates
python -c "
from pathlib import Path
from sage.meta_harness.config import HarnessConfig
a = HarnessConfig.load(Path.home() / '.sage-meta-harness/candidates/001/config.json')
b = HarnessConfig.load(Path.home() / '.sage-meta-harness/candidates/002/config.json')
for k, (va, vb) in a.diff(b).items():
    print(f'{k}: {va} -> {vb}')
"

# Read error messages from failed tasks
grep '"error":' ~/.sage-meta-harness/candidates/001/traces.jsonl | grep -v '""'
```

## Key Principles

1. **Read traces before proposing** — don't guess from scores
2. **Change ONE thing per candidate** — isolate the variable
3. **Less context can be better** — Meta-Harness beat ACE with 4x fewer tokens
4. **Cross-model transfer** — good harnesses work across models
5. **Record your reasoning** in proposal.md
'''
        (self.workspace / "PROPOSER_INSTRUCTIONS.md").write_text(
            instructions, encoding="utf-8",
        )
