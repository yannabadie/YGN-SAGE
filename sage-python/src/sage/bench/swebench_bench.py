"""SWE-Bench evaluation adapter for YGN-SAGE.

Loads SWE-Bench instances (Lite: 300, Verified: 500) from HuggingFace,
feeds each issue to SAGE's agent loop to generate a patch, then evaluates
via the official swebench Docker-based harness.

Requirements:
    pip install swebench datasets docker

Evaluation flow:
    1. Load dataset from HuggingFace (princeton-nlp/SWE-bench_Lite or _Verified)
    2. For each instance: feed problem_statement to SAGE -> capture generated patch
    3. Write predictions in swebench JSONL format
    4. Build Docker images (swebench harness)
    5. Run evaluation in Docker containers (applies patch, runs tests)
    6. Grade results: resolved = all FAIL_TO_PASS tests now pass

Platform notes:
    - swebench's harness requires Linux `resource` module. On Windows, a stub
      is injected at import time. The actual evaluation runs inside Docker
      (Linux containers via Docker Desktop / WSL2), so this is safe.
    - Docker Desktop with WSL2 backend is REQUIRED on Windows.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import sys
import time
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sage.bench.runner import BenchReport, TaskResult
from sage.bench.truth_pack import BenchmarkManifest, TaskTrace

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Windows compatibility: stub the `resource` module before importing swebench
# ---------------------------------------------------------------------------
if platform.system() != "Linux" and "resource" not in sys.modules:
    _resource_stub = types.ModuleType("resource")
    _resource_stub.RLIMIT_NOFILE = 7  # type: ignore[attr-defined]
    _resource_stub.getrlimit = lambda _x: (1024, 1048576)  # type: ignore[attr-defined]
    _resource_stub.setrlimit = lambda _x, _y: None  # type: ignore[attr-defined]
    sys.modules["resource"] = _resource_stub

# ---------------------------------------------------------------------------
# Dataset names on HuggingFace
# ---------------------------------------------------------------------------
_DATASET_MAP = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "full": "princeton-nlp/SWE-bench",
}

# swebench prediction keys
_KEY_INSTANCE_ID = "instance_id"
_KEY_MODEL = "model_name_or_path"
_KEY_PREDICTION = "model_patch"


# ---------------------------------------------------------------------------
# Dataset loading (uses HuggingFace `datasets` directly)
# ---------------------------------------------------------------------------

def _ssl_bypass() -> None:
    """Disable SSL verification for corporate proxy environments."""
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ.setdefault("CURL_CA_BUNDLE", "")
    os.environ.setdefault("REQUESTS_CA_BUNDLE", "")


def load_swebench_dataset(
    dataset: str = "lite",
    split: str = "test",
) -> list[dict[str, Any]]:
    """Load a SWE-Bench dataset from HuggingFace.

    Args:
        dataset: "lite" (300), "verified" (500), or "full" (2294).
        split: HuggingFace split name (default: "test").

    Returns:
        List of instance dicts with keys: repo, instance_id, base_commit,
        patch, test_patch, problem_statement, hints_text, version,
        FAIL_TO_PASS, PASS_TO_PASS, etc.
    """
    _ssl_bypass()
    from datasets import load_dataset as hf_load

    hf_name = _DATASET_MAP.get(dataset, dataset)
    log.info("Loading %s (split=%s) from HuggingFace...", hf_name, split)
    ds = hf_load(hf_name, split=split)
    instances = [dict(row) for row in ds]
    log.info("Loaded %d instances from %s", len(instances), hf_name)
    return instances


# ---------------------------------------------------------------------------
# Prompt engineering for SWE-Bench tasks
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert software engineer. You will be given a GitHub issue and \
information about the repository. Your task is to write a patch (unified diff \
format) that resolves the issue.

Rules:
- Output ONLY the patch in unified diff format (starting with --- and +++).
- The patch must apply cleanly with `git apply`.
- Do NOT include any explanation, markdown fencing, or commentary.
- Focus on the minimal change needed to fix the issue.
- Preserve existing code style and conventions."""

_TASK_TEMPLATE = """\
Repository: {repo}
Version: {version}
Base commit: {base_commit}

## Issue Description

{problem_statement}

{hints_section}\
Write a unified diff patch that resolves this issue. Output ONLY the diff."""


def _build_task_prompt(instance: dict[str, Any]) -> str:
    """Build the task prompt for an SWE-Bench instance."""
    hints = instance.get("hints_text") or ""
    hints_section = ""
    if hints and hints.strip():
        hints_section = f"## Hints\n\n{hints.strip()}\n\n"

    return _TASK_TEMPLATE.format(
        repo=instance["repo"],
        version=instance.get("version", "unknown"),
        base_commit=instance["base_commit"],
        problem_statement=instance["problem_statement"],
        hints_section=hints_section,
    )


def _extract_patch(response: str) -> str:
    """Extract a unified diff patch from agent response.

    Handles various output formats:
    - Raw diff (starts with diff --git or ---)
    - Markdown code blocks (```diff ... ```)
    - Mixed text with embedded diff
    """
    if not response:
        return ""

    response = response.strip()

    # Case 1: Already a clean diff
    if response.startswith("diff --git") or response.startswith("---"):
        return response

    # Case 2: Markdown code block
    for marker in ["```diff", "```patch", "```"]:
        if marker in response:
            start = response.find(marker)
            start = response.find("\n", start) + 1
            end = response.find("```", start)
            if end > start:
                candidate = response[start:end].strip()
                if candidate and ("---" in candidate or "diff --git" in candidate):
                    return candidate

    # Case 3: Find diff content anywhere in the response
    lines = response.split("\n")
    diff_lines: list[str] = []
    in_diff = False
    for line in lines:
        if line.startswith("diff --git") or line.startswith("---"):
            in_diff = True
        if in_diff:
            diff_lines.append(line)
        # Stop if we hit a blank line after a diff section ends
        if in_diff and not line.strip() and diff_lines and not diff_lines[-1].startswith(
            ("diff", "---", "+++", "@@", " ", "+", "-")
        ):
            break

    if diff_lines:
        return "\n".join(diff_lines).strip()

    # Fallback: return the entire response (swebench will reject if not a valid diff)
    return response


# ---------------------------------------------------------------------------
# SWE-Bench Adapter
# ---------------------------------------------------------------------------

@dataclass
class SWEBenchResult:
    """Per-instance evaluation result."""
    instance_id: str
    repo: str
    resolved: bool
    patch_generated: bool
    patch_applied: bool
    error: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    system_used: int = 0


class SWEBenchBench:
    """SWE-Bench evaluation adapter for YGN-SAGE.

    Generates patches via AgentSystem, then evaluates them using the
    official swebench Docker harness.

    Args:
        system: AgentSystem instance (from boot.py).
        event_bus: EventBus for progress events.
        dataset: "lite" (300), "verified" (500), or "full" (2294).
        timeout_per_task: Max seconds for agent to generate a patch.
        eval_timeout: Max seconds for Docker evaluation per instance.
        max_workers: Parallel Docker evaluation workers.
        run_id: Identifier for this evaluation run.
    """

    def __init__(
        self,
        system: Any = None,
        event_bus: Any = None,
        dataset: str = "lite",
        timeout_per_task: float = 120.0,
        eval_timeout: int = 300,
        max_workers: int = 4,
        run_id: str | None = None,
    ):
        if dataset not in _DATASET_MAP:
            raise ValueError(
                f"Unknown dataset '{dataset}'. Supported: {list(_DATASET_MAP.keys())}"
            )
        self.system = system
        self.event_bus = event_bus
        self.dataset = dataset
        self.timeout_per_task = timeout_per_task
        self.eval_timeout = eval_timeout
        self.max_workers = max_workers
        self.run_id = run_id or f"sage-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        self.manifest: BenchmarkManifest | None = None

    # ------------------------------------------------------------------
    # Phase 1: Generate patches
    # ------------------------------------------------------------------

    async def generate_patches(
        self,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Generate patches for SWE-Bench instances via AgentSystem.

        Returns list of prediction dicts in swebench format:
        {instance_id, model_name_or_path, model_patch, ...metadata}
        """
        if self.system is None:
            log.error("No AgentSystem configured")
            return []

        instances = load_swebench_dataset(self.dataset)
        if limit is not None:
            instances = instances[:limit]

        # Detect model info
        model_id = "unknown"
        provider_name = ""
        if hasattr(self.system, "agent_loop"):
            llm = getattr(self.system.agent_loop, "_llm", None)
            if llm:
                model_id = getattr(llm, "model_id", "unknown")
                provider_name = type(llm).__name__

        self.manifest = BenchmarkManifest(
            benchmark=f"swebench_{self.dataset}",
            model=model_id,
            provider=provider_name,
        )

        predictions: list[dict[str, Any]] = []

        for i, instance in enumerate(instances):
            instance_id = instance["instance_id"]
            task_prompt = _build_task_prompt(instance)
            t0 = time.perf_counter()
            error = ""
            system_used = 0
            patch = ""

            try:
                response = await asyncio.wait_for(
                    self.system.run(task_prompt),
                    timeout=self.timeout_per_task,
                )
                patch = _extract_patch(response)
                system_used = (
                    getattr(self.system.agent_loop, "_last_routing_system", 0)
                    or 2
                )
            except asyncio.TimeoutError:
                error = f"generation_timeout_{self.timeout_per_task:.0f}s"
                log.warning("[%s] Generation timed out", instance_id)
            except Exception as e:
                error = str(e)[:200]
                log.error("[%s] Generation failed: %s", instance_id, error)

            latency = (time.perf_counter() - t0) * 1000
            cost = getattr(
                getattr(self.system, "agent_loop", None), "total_cost_usd", 0.0
            )

            predictions.append({
                _KEY_INSTANCE_ID: instance_id,
                _KEY_MODEL: f"sage/{model_id}",
                _KEY_PREDICTION: patch,
                # Metadata (prefixed with _ to not interfere with swebench)
                "_latency_ms": round(latency, 1),
                "_cost_usd": round(cost, 6),
                "_system_used": system_used,
                "_error": error,
                "_repo": instance["repo"],
            })

            self.manifest.add(TaskTrace(
                task_id=instance_id,
                passed=False,  # Unknown until evaluation
                latency_ms=round(latency, 1),
                cost_usd=round(cost, 6),
                model=model_id,
                routing=f"S{system_used}",
                error=error[:200] if error else "",
            ))

            if self.event_bus:
                from sage.agent_loop import AgentEvent
                self.event_bus.emit(AgentEvent(
                    type="BENCH_RESULT",
                    step=i + 1,
                    timestamp=time.time(),
                    meta={
                        "benchmark": f"swebench_{self.dataset}",
                        "task_id": instance_id,
                        "system_used": system_used,
                        "latency_ms": round(latency, 1),
                        "patch_len": len(patch),
                        "progress": f"{i + 1}/{len(instances)}",
                    },
                ))

            has_patch = "PATCH" if patch else "EMPTY"
            status = has_patch if not error else f"ERR:{error[:30]}"
            print(
                f"  [{i + 1}/{len(instances)}] {instance_id}: "
                f"{status} ({latency:.0f}ms, {len(patch)} chars)",
                flush=True,
            )

        return predictions

    # ------------------------------------------------------------------
    # Phase 2: Write predictions file
    # ------------------------------------------------------------------

    def write_predictions(
        self,
        predictions: list[dict[str, Any]],
        path: str | Path,
    ) -> Path:
        """Write predictions in swebench JSONL format.

        Format per line: {instance_id, model_name_or_path, model_patch}
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for pred in predictions:
                entry = {
                    _KEY_INSTANCE_ID: pred[_KEY_INSTANCE_ID],
                    _KEY_MODEL: pred[_KEY_MODEL],
                    _KEY_PREDICTION: pred[_KEY_PREDICTION],
                }
                f.write(json.dumps(entry) + "\n")

        log.info("Wrote %d predictions to %s", len(predictions), path)
        return path

    # ------------------------------------------------------------------
    # Phase 3: Docker-based evaluation (official swebench harness)
    # ------------------------------------------------------------------

    def evaluate_with_harness(
        self,
        predictions_path: str | Path,
        dataset_name: str | None = None,
    ) -> dict[str, Any]:
        """Run official swebench Docker evaluation.

        This builds Docker images for each repo/version, applies the patch,
        runs the test suite, and grades results.

        Returns dict with:
            resolved_ids: list of resolved instance IDs
            completed_ids: list of completed (evaluated) instance IDs
            error_ids: list of instances that errored during evaluation
            resolved_rate: float
        """
        try:
            from swebench.harness.run_evaluation import run_instances, get_dataset_from_preds
            from swebench.harness.utils import get_predictions_from_file
            from swebench.harness.reporting import make_run_report
        except ImportError as e:
            return {
                "error": f"swebench harness not available: {e}",
                "resolved_ids": [],
                "resolved_rate": 0.0,
            }

        predictions_path = str(predictions_path)
        hf_name = dataset_name or _DATASET_MAP[self.dataset]

        # Load predictions
        predictions = get_predictions_from_file(
            predictions_path, hf_name, "test"
        )
        predictions_dict = {p[_KEY_INSTANCE_ID]: p for p in predictions}

        # Get dataset instances that have predictions
        dataset, remaining = get_dataset_from_preds(
            dataset_name=hf_name,
            split="test",
            instance_ids=list(predictions_dict.keys()),
            predictions=predictions_dict,
            run_id=self.run_id,
            rewrite_reports=False,
        )

        if not remaining:
            log.info("All instances already evaluated (or no predictions)")
        else:
            log.info(
                "Evaluating %d instances (%d already done)",
                len(remaining), len(dataset) - len(remaining),
            )

        if remaining:
            # Build images and run evaluation
            print(f"\n  Building Docker images and evaluating {len(remaining)} instances...")
            print(f"  Run ID: {self.run_id}")
            print(f"  Timeout per instance: {self.eval_timeout}s")
            print(f"  Max workers: {self.max_workers}")
            print()

            run_instances(
                predictions=predictions_dict,
                instances=remaining,
                cache_level="env",
                clean=False,
                force_rebuild=False,
                max_workers=self.max_workers,
                run_id=self.run_id,
                timeout=self.eval_timeout,
            )

        # Generate final report
        import docker
        client = docker.from_env()
        report_path = make_run_report(
            predictions=predictions_dict,
            full_dataset=dataset,
            run_id=self.run_id,
            client=client,
        )

        # Parse report
        if report_path and report_path.exists():
            report = json.loads(report_path.read_text())
        else:
            report = {}

        resolved_ids = report.get("resolved_ids", [])
        completed_ids = report.get("completed_ids", [])
        error_ids = report.get("error_ids", [])
        total = len(predictions)

        return {
            "resolved_ids": resolved_ids,
            "completed_ids": completed_ids,
            "error_ids": error_ids,
            "total": total,
            "resolved": len(resolved_ids),
            "resolved_rate": len(resolved_ids) / max(total, 1),
            "report_path": str(report_path) if report_path else None,
        }

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    async def run(self, limit: int | None = None) -> BenchReport:
        """Full pipeline: generate patches, evaluate with Docker harness.

        Returns a BenchReport compatible with the existing benchmark framework.
        """
        # Phase 1: Generate patches
        predictions = await self.generate_patches(limit=limit)
        if not predictions:
            return BenchReport.from_results(
                f"swebench_{self.dataset}", [],
                model_config={"model": self.manifest.model if self.manifest else "unknown"},
            )

        # Phase 2: Write predictions
        import tempfile
        out_dir = Path(tempfile.mkdtemp(prefix="sage_swebench_"))
        preds_path = self.write_predictions(predictions, out_dir / "predictions.jsonl")

        # Phase 3: Evaluate
        print(f"\n  Predictions saved to: {preds_path}")
        print(f"  Starting Docker evaluation...")

        eval_results = self.evaluate_with_harness(preds_path)

        if "error" in eval_results and eval_results.get("resolved_rate", 0) == 0:
            log.error("Docker evaluation failed: %s", eval_results["error"])
            print(f"\n  Docker evaluation failed: {eval_results['error']}")
            print(f"  Predictions are saved at: {preds_path}")
            print(f"  You can evaluate manually with:")
            print(f"    python -m swebench.harness.run_evaluation \\")
            print(f"      --predictions_path {preds_path} \\")
            print(f"      --dataset_name {_DATASET_MAP[self.dataset]} \\")
            print(f"      --run_id {self.run_id}")
            print()

            # Return generation-only results
            task_results = self._predictions_to_task_results(predictions, {})
            return BenchReport.from_results(
                f"swebench_{self.dataset}", task_results,
                model_config={"model": self.manifest.model if self.manifest else "unknown"},
            )

        # Phase 4: Build task results
        resolved_set = set(eval_results.get("resolved_ids", []))
        task_results = self._predictions_to_task_results(predictions, resolved_set)

        # Update manifest traces
        if self.manifest:
            for trace, result in zip(self.manifest.traces, task_results):
                trace.passed = result.passed

        resolved_count = sum(1 for r in task_results if r.passed)
        total = len(task_results)
        print(f"\n  SWE-Bench {self.dataset}: {resolved_count}/{total} resolved "
              f"({resolved_count/max(total,1):.1%})")

        return BenchReport.from_results(
            f"swebench_{self.dataset}", task_results,
            model_config={"model": self.manifest.model if self.manifest else "unknown"},
        )

    async def run_generate_only(self, limit: int | None = None) -> Path:
        """Generate patches and save predictions file (skip Docker evaluation).

        Useful when Docker is not available or for deferred evaluation on Linux.
        Returns path to the predictions JSONL file.
        """
        predictions = await self.generate_patches(limit=limit)
        if not predictions:
            raise RuntimeError("No predictions generated")

        import tempfile
        out_dir = Path(tempfile.mkdtemp(prefix="sage_swebench_"))
        preds_path = self.write_predictions(predictions, out_dir / "predictions.jsonl")

        # Also save the full predictions with metadata
        meta_path = out_dir / "predictions_meta.json"
        meta_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

        # Print summary
        patches_count = sum(1 for p in predictions if p[_KEY_PREDICTION])
        empty_count = sum(1 for p in predictions if not p[_KEY_PREDICTION])
        errors_count = sum(1 for p in predictions if p.get("_error"))

        print(f"\n  Generation complete:")
        print(f"    Total instances: {len(predictions)}")
        print(f"    Patches generated: {patches_count}")
        print(f"    Empty patches: {empty_count}")
        print(f"    Errors: {errors_count}")
        print(f"\n  Predictions: {preds_path}")
        print(f"  Metadata: {meta_path}")
        print(f"\n  To evaluate (requires Docker with Linux containers):")
        print(f"    python -m swebench.harness.run_evaluation \\")
        print(f"      --predictions_path {preds_path} \\")
        print(f"      --dataset_name {_DATASET_MAP[self.dataset]} \\")
        print(f"      --run_id {self.run_id}")
        print()

        return preds_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _predictions_to_task_results(
        self,
        predictions: list[dict[str, Any]],
        resolved_set: set[str],
    ) -> list[TaskResult]:
        """Convert predictions + evaluation results to TaskResult list."""
        results: list[TaskResult] = []
        for pred in predictions:
            instance_id = pred[_KEY_INSTANCE_ID]
            resolved = instance_id in resolved_set
            has_patch = bool(pred[_KEY_PREDICTION])
            error = pred.get("_error", "")
            if not has_patch and not error:
                error = "empty_patch"

            results.append(TaskResult(
                task_id=instance_id,
                passed=resolved,
                system_used=pred.get("_system_used", 0),
                latency_ms=pred.get("_latency_ms", 0.0),
                cost_usd=pred.get("_cost_usd", 0.0),
                error=error,
            ))
        return results


# ---------------------------------------------------------------------------
# Standalone evaluation (for pre-generated predictions)
# ---------------------------------------------------------------------------

def evaluate_predictions(
    predictions_path: str,
    dataset: str = "lite",
    run_id: str | None = None,
    timeout: int = 300,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Evaluate a pre-generated predictions file with the swebench harness.

    This is useful when predictions were generated on one machine (e.g., Windows)
    and evaluation needs to happen on another (e.g., Linux with Docker).

    Args:
        predictions_path: Path to JSONL predictions file.
        dataset: "lite", "verified", or "full".
        run_id: Optional run identifier.
        timeout: Timeout per instance in seconds.
        max_workers: Number of parallel evaluation workers.

    Returns:
        Evaluation results dict.
    """
    bench = SWEBenchBench(
        dataset=dataset,
        eval_timeout=timeout,
        max_workers=max_workers,
        run_id=run_id,
    )
    return bench.evaluate_with_harness(predictions_path)


# ---------------------------------------------------------------------------
# Quick dataset info
# ---------------------------------------------------------------------------

def dataset_info(dataset: str = "lite") -> dict[str, Any]:
    """Print summary information about a SWE-Bench dataset."""
    instances = load_swebench_dataset(dataset)

    repos = {}
    for inst in instances:
        repo = inst["repo"]
        repos[repo] = repos.get(repo, 0) + 1

    difficulties = {}
    for inst in instances:
        diff = inst.get("difficulty", "unknown")
        difficulties[diff] = difficulties.get(diff, 0) + 1

    return {
        "dataset": dataset,
        "hf_name": _DATASET_MAP.get(dataset, dataset),
        "total_instances": len(instances),
        "repos": dict(sorted(repos.items(), key=lambda x: -x[1])),
        "repo_count": len(repos),
        "difficulties": difficulties,
        "columns": list(instances[0].keys()) if instances else [],
    }
