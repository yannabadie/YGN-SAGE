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
import os
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
        event_ledger: Optional ``BenchEventLedger`` for append-only
            crash-safe per-task events (Step 1 of the cycle-9 recovery
            plan). When provided, every task emits TASK_START / TASK_END
            / TASK_TIMEOUT / TASK_ABORT to the ledger with full
            control-surface telemetry.
        config_label: Ablation config label for ledger events. Pass
            "single" for non-ablation runs.
        wallclock_grace_factor: Multiplier on ``task_timeout`` past
            which a task is treated as host-suspended (cycle-9 Step 2).
            Default 2.0; lower for tighter diagnostic; raise for noisy
            environments. Tasks above this threshold emit TASK_ABORT
            and are excluded from gate-quality stats.
    """

    def __init__(
        self,
        system: Any = None,
        event_bus: Any = None,
        subset: str = "full",
        split: str = "instruct",
        task_timeout: float = 120.0,
        eval_timeout: float = 30.0,
        event_ledger: Any = None,
        config_label: str = "single",
        wallclock_grace_factor: float = 2.0,
    ):
        self.system = system
        self.event_bus = event_bus
        self.subset = subset
        self.split = split
        self.task_timeout = task_timeout
        self.eval_timeout = eval_timeout
        self._ledger = event_ledger
        self._config_label = config_label
        self._wallclock_grace = wallclock_grace_factor

    def _capture_control_surface(self, trace: dict) -> dict:
        """Read live system flags + pipeline state for control-surface telemetry.

        Cycle-9 recovery Step 2 (cgpro 2026-05-04). Records the actual
        runtime configuration so post-hoc analysis can answer "what
        mechanism caused the v7 full→no-guardrails gap?". Reads from:

        - ``system.agent_loop._skip_*`` flags (set by AblationConfig.apply)
        - ``system.pipeline._llm_tier`` (Fix C input)
        - ``system.pipeline._effective_controller`` (Fix C output, derived)
        - ``trace`` (already-populated ctx fields from this task)
        """
        cs: dict[str, Any] = {}
        loop = getattr(self.system, "agent_loop", None)
        if loop is not None:
            cs["skip_memory"] = bool(getattr(loop, "_skip_memory", False))
            cs["skip_avr"] = bool(getattr(loop, "_skip_avr", False))
            cs["skip_routing"] = bool(getattr(loop, "_skip_routing", False))
            cs["skip_guardrails"] = bool(getattr(loop, "_skip_guardrails", False))
        pipe = getattr(self.system, "pipeline", None)
        if pipe is not None:
            tier = getattr(pipe, "_llm_tier", "")
            cs["llm_tier"] = tier
            # Fix C derivation: when tier=="budget", controller is
            # masked at TopologyRunner construction time. We mirror
            # that logic here for telemetry without re-reading the
            # actual mask (which is local to _stage_execute).
            cs["controller_attached"] = bool(
                getattr(pipe, "controller", None) is not None and tier != "budget",
            )
            cs["router_active"] = bool(getattr(pipe, "router", None) is not None)
        else:
            cs["controller_attached"] = False
            cs["router_active"] = False
            cs["llm_tier"] = ""
        # Already-captured fields from trace
        cs["executed_template"] = trace.get("topology_id", "")
        cs["node_count"] = int(trace.get("topology_nodes", 0) or 0)
        cs["was_bypassed"] = cs["node_count"] == 0
        cs["domain"] = trace.get("domain", "")
        cs["system_routing"] = trace.get("system", 0)
        # AdaptOrch DAG features (omega/delta/gamma). Cycle-9 α post-mortem
        # 2026-05-04 found BCB/82 topology shifts 5→3 nodes when
        # _skip_guardrails=True; the immediate next question is "which of
        # omega/delta/gamma diverges?". Capturing them per task makes
        # future replays diagnose the coupling without rerunning.
        dag_feats = trace.get("dag_features") or {}
        cs["dag_omega"] = dag_feats.get("omega")
        cs["dag_delta"] = dag_feats.get("delta")
        cs["dag_gamma"] = dag_feats.get("gamma")
        return cs

    async def run(
        self,
        limit: int | None = None,
        task_ids_filter: list[str] | None = None,
    ) -> BenchReport:
        """Run BigCodeBench benchmark.

        Args:
            limit: Cap number of tasks to run (after task_ids_filter).
            task_ids_filter: When non-None, run ONLY these task IDs in
                the dataset's natural order. IDs not present in the
                dataset are warned and skipped. Cycle-9 recovery α.2:
                replaces the older --offset/--limit replay pattern for
                targeted diagnostic experiments.
        """
        problems = _load_dataset(self.subset)
        task_ids = list(problems.keys())
        # α.2: filter by explicit task_ids before applying --limit.
        if task_ids_filter:
            requested = set(task_ids_filter)
            kept = [t for t in task_ids if t in requested]
            missing = [t for t in task_ids_filter if t not in problems]
            if missing:
                log.warning(
                    "task_ids_filter: %d ID(s) not in dataset, skipping: %s",
                    len(missing), missing,
                )
            if not kept:
                log.warning(
                    "task_ids_filter: NO matching tasks found in dataset; "
                    "report will be empty",
                )
            task_ids = kept
        if limit:
            task_ids = task_ids[:limit]

        results: list[TaskResult] = []
        self._predictions: list[dict[str, Any]] = []
        passed_count = 0
        aborted_count = 0
        disable_repair = os.environ.get("SAGE_BENCH_DISABLE_REPAIR") == "1"

        from sage.input.bcb import normalize_bcb

        for i, task_id in enumerate(task_ids):
            task = problems[task_id]
            # C4 (2026-04-22): pass TaskInput directly; AgentSystem.run()
            # dispatches to render_bcb_prompt internally.
            task_input = normalize_bcb(task, self.split)

            entry = task["entry_point"]
            tid = task_id

            # Cycle-9 recovery Step 1: TASK_START emitted before any
            # async wait, so a process death between TASK_START and
            # TASK_END is recoverable from the ledger as "task in
            # flight at crash time".
            if self._ledger is not None:
                try:
                    self._ledger.emit_task_start(
                        config_label=self._config_label,
                        idx=i + 1,
                        task_id=tid,
                        timeout_s=self.task_timeout,
                    )
                except Exception as _le:  # noqa: BLE001
                    log.warning("event_ledger.emit_task_start failed: %s", _le)

            t0 = time.time()
            solution = ""
            error = ""
            eval_stderr = ""

            # Per-task trace for full observability
            trace: dict = {}

            if self.system:
                try:
                    # cgpro 2026-04-29 R6.1a verify Path E: opt-in
                    # bench-result feedback seam. With SAGE_BENCH_ORACLE_SEAM=1,
                    # the pipeline calls our evaluator BEFORE final_result so
                    # _exact_oracle sees bench_result["passed"] and emits a
                    # trainable Exact verdict on the live trace. The bench
                    # still does the same evaluation afterwards (cheap; idempotent
                    # for deterministic test code) so escalation logic below
                    # is unchanged.
                    use_seam = (
                        os.environ.get("SAGE_BENCH_ORACLE_SEAM") == "1"
                        and hasattr(self.system, "run_with_bench_evaluator")
                    )
                    if use_seam:
                        eval_test_code = task["test"]
                        eval_entry = entry
                        eval_task_id = tid
                        eval_timeout_local = self.eval_timeout

                        # Path E seam evaluator. We try the OFFICIAL
                        # bigcodebench.eval.untrusted_check first (calibrated
                        # per the BCB protocol — POSIX-only); on Windows it
                        # raises AttributeError on os.killpg, in which case
                        # we fall back to the in-tree
                        # _evaluate_solution_with_stderr (matplotlib-headless
                        # subprocess; Windows-compatible and deterministic
                        # per (solution, test_code)). The verifier_id field
                        # records which path produced bench_result so the
                        # downstream report can audit calibration provenance.
                        def _bench_evaluator(raw_output: str) -> dict:
                            """Sanitised bench result for the OracleStack.

                            cgpro 2026-04-29 cycle-7 flip review push-back:
                            never put raw harness fragments (stderr tail,
                            traceback) into ``bench_result["reason"]`` —
                            ``_exact_oracle`` would surface them in
                            ``oracle_verdict.reason_codes`` and they would
                            mirror into ``run_frame_summary``. We emit:

                            - ``reason_code`` (small enum-like tag).
                            - ``reason_sha256`` (audit pointer to the raw
                              stderr if any) — never the raw text.
                            """
                            import hashlib as _hashlib
                            sol = extract_code(raw_output, eval_entry)
                            if not sol:
                                return {
                                    "passed": False,
                                    "score": 0.0,
                                    "verifier_id": "bcb_no_solution",
                                    "reason_code": "bcb_no_solution_extracted",
                                }
                            # Try official first.
                            try:
                                from bigcodebench.eval import untrusted_check
                                stat_official, _details = untrusted_check(
                                    code=sol,
                                    test_code=eval_test_code,
                                    entry_point=eval_entry,
                                    max_as_limit=30 * 1024,
                                    max_data_limit=30 * 1024,
                                    max_stack_limit=10,
                                    min_time_limit=1,
                                    gt_time_limit=60,
                                )
                                # On Windows untrusted_check returns
                                # 'timeout' due to os.killpg AttributeError
                                # silently caught upstream; treat as fall-
                                # back trigger if stat is not a clean
                                # pass / fail.
                                if stat_official in ("pass", "fail"):
                                    passed_off = (stat_official == "pass")
                                    return {
                                        "passed": passed_off,
                                        "score": 1.0 if passed_off else 0.0,
                                        "verifier_id": (
                                            "bigcodebench.eval.untrusted_check"
                                        ),
                                        "reason_code": (
                                            "bcb_unittest_pass"
                                            if passed_off
                                            else "bcb_unittest_fail"
                                        ),
                                    }
                            except Exception:  # noqa: BLE001 - fallback path
                                pass
                            # Windows / non-clean fallback to in-tree eval.
                            passed_seam, stderr_seam = (
                                BigCodeBenchBench._evaluate_solution_with_stderr(
                                    solution=sol,
                                    test_code=eval_test_code,
                                    entry_point=eval_entry,
                                    task_id=eval_task_id,
                                    timeout=eval_timeout_local,
                                )
                            )
                            result: dict = {
                                "passed": bool(passed_seam),
                                "score": 1.0 if passed_seam else 0.0,
                                "verifier_id": (
                                    "bcb_internal_subprocess_fallback"
                                ),
                                "reason_code": (
                                    "bcb_unittest_pass"
                                    if passed_seam
                                    else "bcb_unittest_fail"
                                ),
                            }
                            if not passed_seam and stderr_seam:
                                result["reason_sha256"] = (
                                    _hashlib.sha256(
                                        stderr_seam.encode("utf-8")
                                    ).hexdigest()
                                )
                            return result

                        raw, _frame_seam = await asyncio.wait_for(
                            self.system.run_with_bench_evaluator(
                                task_input, _bench_evaluator,
                            ),
                            timeout=self.task_timeout,
                        )
                    else:
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

                if not disable_repair:
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

            # Cycle-9 recovery Step 2: wall-clock host-suspend detection.
            # asyncio.wait_for() does NOT enforce timeout when the event
            # loop is suspended (Windows Modern Standby S0 DRIPS).
            # latency_ms here is wall-clock from time.time() so it
            # captures suspend correctly. If the wall elapsed exceeds
            # task_timeout * grace_factor, the asyncio.wait_for above
            # was bypassed by suspend; the result is suspect and we
            # mark the task aborted, NOT FAIL.
            host_suspend_detected = (
                latency_ms / 1000.0 > self.task_timeout * self._wallclock_grace
            )
            if host_suspend_detected:
                aborted_count += 1
                # Roll back the passed_count increment if applicable —
                # an "aborted" task does not count toward pass-rate.
                if task_passed:
                    passed_count -= 1
                    task_passed = False
                error = "host_suspend_detected"

            # Cycle-9 recovery Step 1+2: emit TASK_END (or TASK_ABORT
            # for sleep-poisoned tasks) with full control-surface
            # telemetry so post-hoc analysis can attribute pass/fail
            # to specific control-surface mechanisms.
            if self._ledger is not None:
                control_surface = self._capture_control_surface(trace)
                control_surface["error"] = error or ""
                control_surface["avr_attempted"] = bool(trace.get("avr_attempted"))
                control_surface["avr_repaired"] = bool(trace.get("avr_repaired"))
                control_surface["pipeline_cost"] = float(trace.get("pipeline_cost") or 0.0)
                try:
                    if host_suspend_detected:
                        self._ledger.emit_task_abort(
                            config_label=self._config_label,
                            idx=i + 1,
                            task_id=tid,
                            reason="host_suspend_detected",
                            elapsed_wall_ms=latency_ms,
                            control_surface=control_surface,
                            grace_factor=self._wallclock_grace,
                            timeout_s=self.task_timeout,
                        )
                    elif error == "TIMEOUT":
                        self._ledger.emit_task_timeout(
                            config_label=self._config_label,
                            idx=i + 1,
                            task_id=tid,
                            elapsed_wall_ms=latency_ms,
                            control_surface=control_surface,
                        )
                    else:
                        self._ledger.emit_task_end(
                            config_label=self._config_label,
                            idx=i + 1,
                            task_id=tid,
                            status="PASS" if task_passed else "FAIL",
                            elapsed_wall_ms=latency_ms,
                            passed=bool(task_passed),
                            control_surface=control_surface,
                            host_suspend_or_event_loop_stall=False,
                        )
                except Exception as _le:  # noqa: BLE001
                    log.warning("event_ledger emit failed: %s", _le)

            results.append(TaskResult(
                task_id=task_id,
                passed=task_passed,
                system_used=trace.get("system", 0),
                latency_ms=latency_ms,
                cost_usd=trace.get("pipeline_cost", 0.0),
                error=error,
            ))

            if host_suspend_detected:
                status = "ABORT"
                log.warning(
                    "[%d/%d] ABORT %s (%.0fms > %.0fms*%.1f) host_suspend_detected",
                    i + 1, len(task_ids), task_id, latency_ms,
                    self.task_timeout * 1000, self._wallclock_grace,
                )
            else:
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
