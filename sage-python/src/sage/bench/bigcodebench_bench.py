"""BigCodeBench adapter: 1140 real-world coding tasks (ICLR '25).

Wraps the bigcodebench package to generate solutions via AgentSystem,
evaluate locally with unittest subprocess, and optionally run official CLI.

Install: pip install bigcodebench
Dataset: https://huggingface.co/datasets/bigcode/bigcodebench
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from sage._python import PYTHON
from sage.bench.humaneval import extract_code
from sage.bench.runner import BenchReport, TaskResult

log = logging.getLogger(__name__)


def _load_dataset(subset: str = "full") -> dict[str, dict[str, Any]]:
    """Load BigCodeBench dataset.

    Args:
        subset: "full" (1140 tasks) or "hard" (~150 tasks).
    """
    from bigcodebench.data import get_bigcodebench
    return get_bigcodebench(subset=subset)


class BigCodeBenchBench:
    """BigCodeBench adapter for SAGE.

    Args:
        system: AgentSystem to benchmark.
        event_bus: EventBus for BENCH_RESULT events.
        subset: "full" (1140) or "hard" (~150).
        split: "instruct" (NL) or "complete" (docstring).
        task_timeout: Max seconds for LLM generation.
        eval_timeout: Max seconds for test evaluation.
    """

    def __init__(
        self,
        system: Any = None,
        event_bus: Any = None,
        subset: str = "full",
        split: str = "instruct",
        task_timeout: float = 120.0,
        eval_timeout: float = 30.0,
    ):
        self.system = system
        self.event_bus = event_bus
        self.subset = subset
        self.split = split
        self.task_timeout = task_timeout
        self.eval_timeout = eval_timeout

    async def run(self, limit: int | None = None) -> BenchReport:
        """Run BigCodeBench benchmark."""
        problems = _load_dataset(self.subset)
        task_ids = list(problems.keys())
        if limit:
            task_ids = task_ids[:limit]

        results: list[TaskResult] = []
        self._predictions: list[dict[str, str]] = []
        passed_count = 0

        from sage.input.bcb import normalize_bcb

        for i, task_id in enumerate(task_ids):
            task = problems[task_id]
            # C4 (2026-04-22): pass TaskInput directly; AgentSystem.run()
            # dispatches to render_bcb_prompt internally.
            task_input = normalize_bcb(task, self.split)

            entry = task["entry_point"]
            tid = task_id

            t0 = time.time()
            solution = ""
            error = ""
            eval_stderr = ""

            # Per-task trace for full observability
            trace: dict = {}

            if self.system:
                try:
                    raw = await asyncio.wait_for(
                        self.system.run(task_input),
                        timeout=self.task_timeout,
                    )
                    solution = extract_code(raw, entry)
                    # Capture pipeline context if available
                    pipe = getattr(self.system, "pipeline", None)
                    ctx = getattr(pipe, "last_context", None) if pipe else None
                    if ctx:
                        trace = {
                            "system": ctx.system,
                            "domain": ctx.domain,
                            "topology_id": ctx.topology_id,
                            "topology_nodes": (
                                ctx.topology.node_count()
                                if ctx.topology and hasattr(ctx.topology, "node_count")
                                else 0
                            ),
                            "assignments": {str(k): v for k, v in ctx.assignments.items()},
                            "dag_features": (
                                {"omega": ctx.dag_features.omega, "delta": ctx.dag_features.delta, "gamma": ctx.dag_features.gamma}
                                if ctx.dag_features else None
                            ),
                            "pipeline_cost": ctx.cost,
                            "pipeline_latency_ms": ctx.latency_ms,
                        }
                except asyncio.TimeoutError:
                    error = "TIMEOUT"
                except Exception as exc:
                    error = str(exc)[:200]

            latency_ms = (time.time() - t0) * 1000

            if solution and not error:
                task_passed, eval_stderr = self._evaluate_solution_with_stderr(
                    solution=solution,
                    test_code=task["test"],
                    entry_point=entry,
                    task_id=tid,
                    timeout=self.eval_timeout,
                )

                # Escalation strategy (Conductor-inspired recursive self-invocation):
                # 1. Bypass already tried (fast single-agent) → failed
                # 2. Try reasoner repair (stronger model fixes syntax/logic)
                # 3. If still fails AND was bypassed → escalate with full topology
                was_bypassed = trace.get("topology_nodes", 0) == 0

                # Step 1: Reasoner repair (arXiv 2306.09896: 1 repair optimal)
                #
                # 2026-04-21 audit (docs/audits/2026-04-21-bcb-hard-failure-analysis.md,
                # top-1 lever). Prior prompt only gave the LLM the NL description
                # + last 500 chars of stderr. The repair stage fired 72/72 times
                # on non-API failures in the Apr-08 run and repaired 0 of them —
                # classic "not enough signal" bottleneck. Expected lift +5-8 pp.
                #
                # Enrichments (all already public for this instance — no leakage):
                #   - Function template (``task['code_prompt']``) — signature +
                #     docstring. Anchors the repair on the exact entry-point shape.
                #   - Acceptance tests (``task['test']``) — the tests that just
                #     failed. This is the contract, not the solution. Equivalent
                #     to what SWE-bench's agent discovers via ``grep test_*``;
                #     BCB has no tool loop, so we inject directly. Standard Aider
                #     / OpenHands / SWE-agent practice.
                #   - Stderr window raised 500 → 1500 chars (cap is 2000 since
                #     this morning's commit c6e40e9); keeps full tracebacks.
                #   - Entry-point explicitly named again to prevent LLM renaming.
                #
                # NOT included (would be overfitting / leakage):
                #   - ``task['canonical_solution']`` — ground-truth code. Excluded.
                if not task_passed and eval_stderr:
                    code_template = task.get("code_prompt", "") or task.get("prompt", "")
                    test_code = task.get("test", "")
                    # C4 (2026-04-22): `prompt` was renamed to `task_input`
                    # at the loop head. Rebuild the pre-C4 string on the fly
                    # so the AVR retry prompt stays byte-identical with what
                    # the commit 9eb05b0 smoke measured.
                    from sage.input.bcb import render_bcb_prompt
                    original_prompt = render_bcb_prompt(task_input)
                    retry_prompt = (
                        f"Your previous code for this task failed. Read the error, "
                        f"the function template, and the acceptance tests, then "
                        f"return a corrected implementation.\n\n"
                        f"## Error from the failing test run\n"
                        f"```\n{eval_stderr[-1500:]}\n```\n\n"
                        f"## Original task description\n"
                        f"{original_prompt}\n\n"
                        f"## Function template to complete\n"
                        f"```python\n{code_template}\n```\n\n"
                        f"## Acceptance tests (these are what just failed)\n"
                        f"```python\n{test_code}\n```\n\n"
                        f"Return ONLY the corrected Python code inside a "
                        f"```python fenced block. The function must be named "
                        f"exactly `{entry}`. Do not include the tests — only "
                        f"the implementation."
                    )
                    try:
                        raw = await asyncio.wait_for(
                            self._run_with_reasoner(retry_prompt),
                            timeout=self.task_timeout,
                        )
                        code = extract_code(raw, entry)
                        if code.strip():
                            solution = code
                            task_passed, eval_stderr = self._evaluate_solution_with_stderr(
                                solution=code,
                                test_code=task["test"],
                                entry_point=entry,
                                task_id=tid,
                                timeout=self.eval_timeout,
                            )
                            if task_passed:
                                log.info("  Repair succeeded for %s (reasoner tier)", tid)
                    except Exception:
                        pass

                # Step 2: Topology escalation (only if bypassed and still failing)
                if not task_passed and was_bypassed:
                    try:
                        pipe = getattr(self.system, "pipeline", None)
                        if pipe:
                            pipe._force_topology = True  # Override bypass
                        raw = await asyncio.wait_for(
                            self.system.run(task_input),
                            timeout=self.task_timeout,
                        )
                        if pipe:
                            pipe._force_topology = False
                        code = extract_code(raw, entry)
                        if code.strip():
                            solution = code
                            task_passed, eval_stderr = self._evaluate_solution_with_stderr(
                                solution=code,
                                test_code=task["test"],
                                entry_point=entry,
                                task_id=tid,
                                timeout=self.eval_timeout,
                            )
                            if task_passed:
                                log.info("  Topology escalation succeeded for %s", tid)
                    except Exception:
                        pipe = getattr(self.system, "pipeline", None)
                        if pipe:
                            pipe._force_topology = False
            else:
                task_passed = False
                if not error:
                    error = "no solution generated"

            if task_passed:
                passed_count += 1

            # Enrich trace with AVR repair and eval info
            trace["avr_attempted"] = bool(eval_stderr and solution and not error)
            trace["avr_repaired"] = bool(trace.get("avr_attempted") and task_passed and eval_stderr)
            # 2026-04-21 audit (docs/audits/2026-04-21-bcb-hard-failure-analysis.md):
            # prior 200-char cap was often eaten by matplotlib/scipy warning
            # preambles, masking the real assertion/error and misclassifying
            # 10/80 failures as "truncated_warning_prefix". 2000 chars keeps
            # the full Python traceback visible to both the classifier and the
            # AVR repair prompt while staying well under typical LLM context
            # headroom.
            trace["eval_error_snippet"] = eval_stderr[:2000] if eval_stderr and not task_passed else ""
            trace["generation_error"] = error

            # Track prediction + trace for JSONL submission
            self._predictions.append({
                "task_id": task_id,
                "solution": solution or "",
                "_trace": trace,
            })

            results.append(TaskResult(
                task_id=task_id,
                passed=task_passed,
                system_used=trace.get("system", 0),
                latency_ms=latency_ms,
                cost_usd=trace.get("pipeline_cost", 0.0),
                error=error,
            ))

            status = "PASS" if task_passed else "FAIL"
            log.info("[%d/%d] %s %s (%.0fms)", i + 1, len(task_ids), status, task_id, latency_ms)

        total = len(results)
        # Aggregate cost from per-task results. Reading agent_loop.total_cost_usd
        # at the end was wrong in topology mode: the top-level agent_loop
        # doesn't run per-task in multi-agent paths, so the value was whatever
        # had accumulated from the last bypass run. Per-result costs come from
        # trace.pipeline_cost (per-task pipeline ctx.cost, now provider-reported
        # since the 2026-04-18 P0.3 wiring) — summing them is the actual total.
        cost = sum(r.cost_usd or 0.0 for r in results)
        # Build routing breakdown from collected traces
        routing = {"S1": 0, "S2": 0, "S3": 0}
        for r in results:
            key = f"S{r.system_used}" if r.system_used in (1, 2, 3) else "unknown"
            routing[key] = routing.get(key, 0) + 1

        return BenchReport(
            benchmark=f"bigcodebench-{self.subset}-{self.split}",
            total=total,
            passed=passed_count,
            failed=total - passed_count,
            errors=sum(1 for r in results if r.error),
            pass_rate=passed_count / total if total else 0.0,
            avg_latency_ms=sum(r.latency_ms for r in results) / total if total else 0.0,
            avg_cost_usd=cost / total if total else 0.0,
            routing_breakdown=routing,
            results=results,
        )

    async def _run_with_reasoner(self, prompt: str) -> str:
        """Run prompt with reasoner tier for AVR repair (stronger than generation model).

        Falls back to system.run() if reasoner provider unavailable.
        """
        try:
            from sage.llm.router import ModelRouter
            config = ModelRouter.get_config("reasoner", temperature=0.0)
            provider_pool = getattr(self.system, 'pipeline', None)
            pool = getattr(provider_pool, 'provider_pool', None) if provider_pool else None
            if pool:
                from sage.llm.base import Message, Role
                prov, _ = pool.resolve(config.model)
                resp = await prov.generate(
                    messages=[Message(role=Role.USER, content=prompt)],
                    config=config,
                )
                return resp.content or ""
        except Exception:
            pass
        # Fallback: use normal system.run()
        return await self.system.run(prompt)

    def write_predictions(self, path: str | Path) -> None:
        """Write predictions in JSONL format for official BigCodeBench submission.

        Each line: {"task_id": "BigCodeBench/N", "solution": "..."}
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for pred in self._predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")
        log.info("Wrote %d predictions to %s", len(self._predictions), path)

    @staticmethod
    def _evaluate_solution(
        solution: str,
        test_code: str,
        entry_point: str,
        task_id: str,
        timeout: float = 30.0,
    ) -> bool:
        """Evaluate by running unittest test cases in subprocess."""
        passed, _ = BigCodeBenchBench._evaluate_solution_with_stderr(
            solution=solution,
            test_code=test_code,
            entry_point=entry_point,
            task_id=task_id,
            timeout=timeout,
        )
        return passed

    @staticmethod
    def _evaluate_solution_with_stderr(
        solution: str,
        test_code: str,
        entry_point: str,
        task_id: str,
        timeout: float = 30.0,
    ) -> tuple[bool, str]:
        """Evaluate solution and return (passed, stderr) for AVR retry."""
        # Force headless matplotlib to prevent GUI popups during eval
        script = f"""import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

{solution}

{test_code}

if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=0)
"""
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(script)
                f.flush()
                tmp_path = f.name

            result = subprocess.run(
                [PYTHON, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            stderr = result.stderr or ""
            return result.returncode == 0, stderr

        except subprocess.TimeoutExpired:
            log.debug("Eval timeout for %s", task_id)
            return False, "TIMEOUT: evaluation exceeded time limit"
        except Exception as exc:
            log.debug("Eval error for %s: %s", task_id, exc)
            return False, str(exc)[:500]
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
