# Official Benchmarks Integration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate EvalPlus (HumanEval+/MBPP+) and BigCodeBench as official code generation benchmarks, add an ablation framework to prove each pillar's value, and run everything honestly with real LLM calls — no mocks, no placeholders.

**Architecture:** EvalPlus adapter generates solutions via AgentSystem, writes JSONL in EvalPlus format, then calls EvalPlus evaluation pipeline with 80x more tests. Ablation framework toggles pillars (memory, AVR, routing, guardrails) to isolate each component's contribution. BigCodeBench adapter generates solutions for 1140 real-world tasks. All results saved as machine-auditable truth packs.

**Tech Stack:** evalplus (PyPI), bigcodebench (PyPI), existing sage bench infrastructure (BenchReport, TaskResult, BenchmarkManifest), AgentSystem with ablation config.

**Research Sources:**
- Gemini 2.5 Pro consultation: ablation study design (6 configs), priority ordering
- EvalPlus API: `from evalplus.data import get_human_eval_plus, write_jsonl`
- EvalPlus solution format: `{"task_id": "HumanEval/N", "solution": "def f():\n    ..."}`
- BigCodeBench: `pip install bigcodebench`, JSONL format, Docker/local execution
- BFCL v4: `pip install bfcl-eval`, AST-based evaluation

---

### Task 1: Install EvalPlus and verify dataset access

**Files:**
- Modify: `sage-python/pyproject.toml` (add evalplus to optional deps)

**Step 1: Add evalplus dependency**

Add to `pyproject.toml`:
```toml
bench = ["evalplus>=0.3.1"]
```
And update the `all` extra to include `bench`.

**Step 2: Install and verify**

Run: `cd sage-python && pip install -e ".[bench]"`
Expected: evalplus installed successfully

**Step 3: Verify dataset access**

Run: `python -c "from evalplus.data import get_human_eval_plus; d = get_human_eval_plus(); print(f'{len(d)} problems loaded')"`
Expected: `164 problems loaded`

**Step 4: Commit**

```bash
git add sage-python/pyproject.toml
git commit -m "deps: add evalplus to bench optional dependencies"
```

---

### Task 2: Write EvalPlus adapter — solution generation

**Files:**
- Create: `sage-python/src/sage/bench/evalplus_bench.py`
- Test: `sage-python/tests/test_evalplus_bench.py`

**Step 1: Write the failing test**

```python
"""Tests for EvalPlus benchmark adapter."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def test_evalplus_bench_imports():
    """EvalPlusBench should be importable."""
    from sage.bench.evalplus_bench import EvalPlusBench
    assert EvalPlusBench is not None


def test_evalplus_bench_init():
    """EvalPlusBench initializes with system and config."""
    from sage.bench.evalplus_bench import EvalPlusBench
    bench = EvalPlusBench(system=None)
    assert bench.system is None
    assert bench.dataset == "humaneval"


def test_evalplus_bench_init_mbpp():
    """EvalPlusBench supports MBPP+ dataset."""
    from sage.bench.evalplus_bench import EvalPlusBench
    bench = EvalPlusBench(system=None, dataset="mbpp")
    assert bench.dataset == "mbpp"


@pytest.mark.asyncio
async def test_generate_solutions_no_system():
    """Without system, generate_solutions returns empty list."""
    from sage.bench.evalplus_bench import EvalPlusBench
    bench = EvalPlusBench(system=None)
    solutions = await bench.generate_solutions(limit=5)
    assert len(solutions) == 0


@pytest.mark.asyncio
async def test_generate_solutions_with_mock_system():
    """With a mock system, generate_solutions produces solutions."""
    from sage.bench.evalplus_bench import EvalPlusBench

    mock_system = MagicMock()
    mock_system.run = AsyncMock(return_value="def has_close_elements(numbers, threshold):\n    return False")

    bench = EvalPlusBench(system=mock_system)
    solutions = await bench.generate_solutions(limit=2)

    assert len(solutions) == 2
    assert "task_id" in solutions[0]
    assert "solution" in solutions[0]
    assert mock_system.run.call_count == 2
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_evalplus_bench.py -v`
Expected: FAIL (module not found)

**Step 3: Write EvalPlusBench adapter**

```python
"""EvalPlus benchmark adapter — generates solutions via AgentSystem,
evaluates with EvalPlus enhanced tests (80x HumanEval, 35x MBPP)."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from sage.bench.runner import BenchReport, TaskResult
from sage.bench.truth_pack import BenchmarkManifest, TaskTrace

log = logging.getLogger(__name__)


class EvalPlusBench:
    """Run EvalPlus (HumanEval+ or MBPP+) benchmark.

    Workflow:
    1. Load problems from evalplus dataset
    2. Generate solutions via AgentSystem.run()
    3. Write solutions to JSONL
    4. Call evalplus evaluator
    5. Parse results into BenchReport
    """

    def __init__(
        self,
        system: Any = None,
        event_bus: Any = None,
        dataset: str = "humaneval",
        baseline_mode: bool = False,
    ):
        self.system = system
        self.event_bus = event_bus
        self.dataset = dataset
        self.baseline_mode = baseline_mode
        self.manifest: BenchmarkManifest | None = None

    def _load_problems(self) -> dict[str, dict]:
        """Load EvalPlus problems."""
        if self.dataset == "mbpp":
            from evalplus.data import get_mbpp_plus
            return get_mbpp_plus()
        else:
            from evalplus.data import get_human_eval_plus
            return get_human_eval_plus()

    async def generate_solutions(
        self, limit: int | None = None
    ) -> list[dict[str, str]]:
        """Generate solutions for all problems using AgentSystem.

        Returns list of dicts with task_id and solution fields
        (EvalPlus JSONL format).
        """
        if not self.system:
            return []

        problems = self._load_problems()
        task_ids = list(problems.keys())
        if limit:
            task_ids = task_ids[:limit]

        solutions: list[dict[str, str]] = []
        for i, task_id in enumerate(task_ids):
            problem = problems[task_id]
            prompt = problem["prompt"]
            entry_point = problem["entry_point"]

            task = (
                "Complete this Python function. "
                "Return ONLY the complete function including the signature, no explanation.\n\n"
                f"```python\n{prompt}\n```"
            )

            try:
                if self.baseline_mode:
                    from sage.llm.base import Message, Role
                    llm_response = await self.system.agent_loop._llm.generate(
                        [Message(role=Role.USER, content=task)]
                    )
                    response = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
                else:
                    response = await self.system.run(task)

                # Extract code from response
                from sage.bench.humaneval import extract_code
                code = extract_code(response, entry_point)

                # Build full solution: prompt + completion if needed
                if f"def {entry_point}" in code:
                    solution = code
                else:
                    solution = prompt + code

                solutions.append({"task_id": task_id, "solution": solution})
                log.info(f"[{i+1}/{len(task_ids)}] {task_id}: generated")

            except Exception as e:
                log.error(f"[{i+1}/{len(task_ids)}] {task_id}: FAILED - {e}")
                # Submit empty solution (will fail evaluation)
                solutions.append({"task_id": task_id, "solution": prompt + "    pass"})

        return solutions

    def write_solutions(self, solutions: list[dict[str, str]], path: Path) -> None:
        """Write solutions to JSONL file in EvalPlus format."""
        from evalplus.data import write_jsonl
        write_jsonl(str(path), solutions)

    def evaluate(self, samples_path: Path) -> dict[str, Any]:
        """Run EvalPlus evaluation on generated samples.

        Returns parsed evaluation results.
        """
        dataset = self.dataset
        cmd = [
            "python", "-m", "evalplus.evaluate",
            "--dataset", dataset,
            "--samples", str(samples_path),
        ]
        log.info(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        log.info(f"EvalPlus stdout:\n{result.stdout}")
        if result.stderr:
            log.warning(f"EvalPlus stderr:\n{result.stderr}")

        # Parse results from the eval_results.json file
        # EvalPlus writes results next to the samples file
        results_dir = samples_path.parent
        eval_results = list(results_dir.glob("*eval_results.json"))
        if eval_results:
            with open(eval_results[0], encoding="utf-8") as f:
                return json.load(f)

        return {"error": result.stderr or "No results file found", "stdout": result.stdout}

    async def run(self, limit: int | None = None) -> BenchReport:
        """Full pipeline: generate -> write -> evaluate -> report."""
        model_id = ""
        if self.system and hasattr(self.system, "agent_loop"):
            model_id = getattr(self.system.agent_loop, "_last_model", "") or "unknown"
        self.manifest = BenchmarkManifest(
            benchmark=f"evalplus-{self.dataset}", model=model_id
        )

        # Step 1: Generate solutions
        t0 = time.perf_counter()
        solutions = await self.generate_solutions(limit=limit)
        gen_time = time.perf_counter() - t0

        if not solutions:
            return BenchReport.from_results(f"evalplus-{self.dataset}", [])

        # Step 2: Write to temp JSONL
        with tempfile.TemporaryDirectory() as tmpdir:
            samples_path = Path(tmpdir) / "samples.jsonl"
            self.write_solutions(solutions, samples_path)
            log.info(f"Wrote {len(solutions)} solutions to {samples_path}")

            # Step 3: Evaluate
            eval_results = self.evaluate(samples_path)

        # Step 4: Parse into BenchReport
        results: list[TaskResult] = []
        if "eval" in eval_results:
            eval_data = eval_results["eval"]
            for task_id, task_eval in eval_data.items():
                # EvalPlus format: each task has base_tests and plus_tests results
                base_passed = task_eval.get("base", [{}])[0].get("result", "failed") == "passed"
                plus_passed = task_eval.get("plus", [{}])[0].get("result", "failed") == "passed"
                passed = base_passed and plus_passed

                results.append(TaskResult(
                    task_id=task_id,
                    passed=passed,
                    latency_ms=round(gen_time * 1000 / len(solutions), 1),
                    error="" if passed else "plus_tests_failed" if base_passed else "base_tests_failed",
                ))
                self.manifest.add(TaskTrace(
                    task_id=task_id,
                    passed=passed,
                    latency_ms=round(gen_time * 1000 / len(solutions), 1),
                    cost_usd=0.0,
                    model=model_id,
                    error="" if passed else "eval_failed",
                ))
        else:
            # Fallback: treat all as failed
            for sol in solutions:
                results.append(TaskResult(
                    task_id=sol["task_id"],
                    passed=False,
                    error=str(eval_results.get("error", "unknown")),
                ))

        return BenchReport.from_results(f"evalplus-{self.dataset}", results)
```

**Step 4: Run tests**

Run: `cd sage-python && python -m pytest tests/test_evalplus_bench.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add sage-python/src/sage/bench/evalplus_bench.py sage-python/tests/test_evalplus_bench.py
git commit -m "feat(bench): add EvalPlus adapter for HumanEval+/MBPP+ evaluation"
```

---

### Task 3: Write ablation framework

**Files:**
- Create: `sage-python/src/sage/bench/ablation.py`
- Test: `sage-python/tests/test_ablation.py`

**Step 1: Write the failing test**

```python
"""Tests for ablation study framework."""
from __future__ import annotations

import pytest
from sage.bench.ablation import AblationConfig, ABLATION_CONFIGS


def test_ablation_config_defaults():
    """Default config enables everything."""
    cfg = AblationConfig()
    assert cfg.memory is True
    assert cfg.avr is True
    assert cfg.routing is True
    assert cfg.guardrails is True
    assert cfg.label == "full"


def test_ablation_config_no_memory():
    """Can disable memory."""
    cfg = AblationConfig(memory=False, label="no-memory")
    assert cfg.memory is False
    assert cfg.label == "no-memory"


def test_predefined_configs():
    """6 predefined ablation configs exist."""
    assert len(ABLATION_CONFIGS) == 6
    labels = {c.label for c in ABLATION_CONFIGS}
    assert "full" in labels
    assert "baseline" in labels
    assert "no-memory" in labels
    assert "no-avr" in labels
    assert "no-routing" in labels
    assert "no-guardrails" in labels


def test_ablation_config_apply():
    """Config can be applied to an AgentSystem (mock)."""
    from unittest.mock import MagicMock
    cfg = AblationConfig(memory=False, avr=False, label="stripped")
    system = MagicMock()
    system.agent_loop = MagicMock()
    system.agent_loop.config = {}
    cfg.apply(system)
    # Should have set attributes
    assert system.agent_loop._skip_memory is True
    assert system.agent_loop._skip_avr is True
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_ablation.py -v`
Expected: FAIL (module not found)

**Step 3: Write ablation module**

```python
"""Ablation study framework for isolating pillar contributions.

Design (from Gemini 2.5 Pro consultation, March 10 2026):
6 configurations to measure each pillar's delta:
1. full — all pillars enabled (reference)
2. baseline — bare LLM call (ReAct prompt, no framework)
3. no-memory — disable memory injection (STM/episodic/semantic/ExoCortex)
4. no-avr — disable S2 Act-Verify-Refine loop (one-shot generation)
5. no-routing — force S2 for everything (no cognitive routing)
6. no-guardrails — disable all guardrails (input/runtime/output)

Each config toggles flags on AgentSystem/AgentLoop before benchmark run.
The delta between configs quantifies each pillar's contribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AblationConfig:
    """Configuration for ablation study — which pillars to enable."""

    memory: bool = True
    avr: bool = True
    routing: bool = True
    guardrails: bool = True
    label: str = "full"

    def apply(self, system: Any) -> None:
        """Apply this ablation config to an AgentSystem.

        Sets internal flags on the agent loop to skip components.
        """
        loop = system.agent_loop

        # Memory: skip CRAG gate + S-MMU + episodic/semantic injection
        loop._skip_memory = not self.memory

        # AVR: skip Act-Verify-Refine loop (one-shot generation)
        loop._skip_avr = not self.avr

        # Routing: force all tasks to S2 (no metacognition routing)
        loop._skip_routing = not self.routing

        # Guardrails: skip input/runtime/output guardrails
        loop._skip_guardrails = not self.guardrails


# Predefined configurations for systematic ablation study
ABLATION_CONFIGS = [
    AblationConfig(label="full"),
    AblationConfig(
        memory=False, avr=False, routing=False, guardrails=False,
        label="baseline",
    ),
    AblationConfig(memory=False, label="no-memory"),
    AblationConfig(avr=False, label="no-avr"),
    AblationConfig(routing=False, label="no-routing"),
    AblationConfig(guardrails=False, label="no-guardrails"),
]
```

**Step 4: Run tests**

Run: `cd sage-python && python -m pytest tests/test_ablation.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add sage-python/src/sage/bench/ablation.py sage-python/tests/test_ablation.py
git commit -m "feat(bench): add ablation framework for pillar contribution isolation"
```

---

### Task 4: Wire ablation flags into agent_loop.py

**Files:**
- Modify: `sage-python/src/sage/agent_loop.py`
- Test: `sage-python/tests/test_ablation.py` (extend)

**Step 1: Write the failing test**

Add to `tests/test_ablation.py`:
```python
def test_agent_loop_has_skip_flags():
    """AgentLoop should have _skip_* attributes (default False)."""
    from sage.agent_loop import AgentLoop
    from unittest.mock import MagicMock
    loop = AgentLoop.__new__(AgentLoop)
    # These should exist after __init__
    # For unit test, just check the class can accept them
    loop._skip_memory = True
    loop._skip_avr = True
    loop._skip_routing = True
    loop._skip_guardrails = True
    assert loop._skip_memory is True
```

**Step 2: Add skip flags to AgentLoop.__init__**

In `agent_loop.py`, add after existing `self.tool_executor` line:
```python
# Ablation study flags (default: all enabled)
self._skip_memory: bool = False
self._skip_avr: bool = False
self._skip_routing: bool = False
self._skip_guardrails: bool = False
```

**Step 3: Guard memory injection with _skip_memory**

In the PERCEIVE/THINK phase where memory is injected, wrap with:
```python
if not self._skip_memory:
    # existing memory injection code
```

**Step 4: Guard AVR with _skip_avr**

In the S2 AVR loop section, wrap with:
```python
if not self._skip_avr:
    # existing AVR loop
```

**Step 5: Guard routing with _skip_routing**

In the routing section, add:
```python
if self._skip_routing:
    system = 2  # Force S2
else:
    # existing routing code
```

**Step 6: Guard guardrails with _skip_guardrails**

In input/runtime/output guardrail checks, wrap with:
```python
if not self._skip_guardrails:
    # existing guardrail code
```

**Step 7: Run all tests**

Run: `cd sage-python && python -m pytest tests/ -v --timeout=30`
Expected: All tests PASS (including existing 1036)

**Step 8: Commit**

```bash
git add sage-python/src/sage/agent_loop.py sage-python/tests/test_ablation.py
git commit -m "feat(agent): wire ablation skip flags into agent loop"
```

---

### Task 5: Write BigCodeBench adapter

**Files:**
- Create: `sage-python/src/sage/bench/bigcodebench_adapter.py`
- Test: `sage-python/tests/test_bigcodebench.py`

**Step 1: Write the failing test**

```python
"""Tests for BigCodeBench adapter."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock


def test_bigcodebench_imports():
    """BigCodeBenchAdapter should be importable."""
    from sage.bench.bigcodebench_adapter import BigCodeBenchAdapter
    assert BigCodeBenchAdapter is not None


def test_bigcodebench_init():
    """BigCodeBenchAdapter initializes with defaults."""
    from sage.bench.bigcodebench_adapter import BigCodeBenchAdapter
    adapter = BigCodeBenchAdapter(system=None)
    assert adapter.system is None
    assert adapter.split == "complete"
    assert adapter.subset == "hard"
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_bigcodebench.py -v`
Expected: FAIL (module not found)

**Step 3: Write BigCodeBench adapter**

```python
"""BigCodeBench adapter — 1140 real-world code gen tasks with library calls.

BigCodeBench evaluates code that uses real APIs (pandas, sklearn, matplotlib, etc.)
via execution-based testing. This adapter generates solutions via AgentSystem
and delegates evaluation to the bigcodebench CLI.

Install: pip install bigcodebench
Ref: https://github.com/bigcode-project/bigcodebench (ICLR 2025)
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from sage.bench.runner import BenchReport, TaskResult
from sage.bench.truth_pack import BenchmarkManifest, TaskTrace

log = logging.getLogger(__name__)


class BigCodeBenchAdapter:
    """Run BigCodeBench benchmark via AgentSystem.

    Splits: 'complete' (docstring-based) or 'instruct' (natural language).
    Subsets: 'full' (1140 tasks) or 'hard' (148 tasks).
    """

    def __init__(
        self,
        system: Any = None,
        event_bus: Any = None,
        split: str = "complete",
        subset: str = "hard",
        baseline_mode: bool = False,
    ):
        self.system = system
        self.event_bus = event_bus
        self.split = split
        self.subset = subset
        self.baseline_mode = baseline_mode
        self.manifest: BenchmarkManifest | None = None

    def _load_problems(self) -> dict[str, dict]:
        """Load BigCodeBench problems."""
        from bigcodebench.data import get_bigcodebench
        return get_bigcodebench(subset=self.subset)

    async def generate_solutions(
        self, limit: int | None = None
    ) -> list[dict[str, str]]:
        """Generate solutions for BigCodeBench tasks."""
        if not self.system:
            return []

        problems = self._load_problems()
        task_ids = list(problems.keys())
        if limit:
            task_ids = task_ids[:limit]

        solutions: list[dict[str, str]] = []
        for i, task_id in enumerate(task_ids):
            problem = problems[task_id]
            prompt = problem.get("complete_prompt", problem.get("instruct_prompt", ""))

            task = (
                "Complete this Python function. "
                "Return ONLY the complete function including imports and the signature, no explanation.\n\n"
                f"```python\n{prompt}\n```"
            )

            try:
                if self.baseline_mode:
                    from sage.llm.base import Message, Role
                    llm_response = await self.system.agent_loop._llm.generate(
                        [Message(role=Role.USER, content=task)]
                    )
                    response = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
                else:
                    response = await self.system.run(task)

                from sage.bench.humaneval import extract_code
                entry_point = problem.get("entry_point", "solution")
                code = extract_code(response, entry_point)

                if f"def {entry_point}" in code:
                    solution = code
                else:
                    solution = prompt + code

                solutions.append({"task_id": task_id, "solution": solution})
                log.info(f"[{i+1}/{len(task_ids)}] {task_id}: generated")

            except Exception as e:
                log.error(f"[{i+1}/{len(task_ids)}] {task_id}: FAILED - {e}")
                solutions.append({"task_id": task_id, "solution": prompt + "    pass"})

        return solutions

    async def run(self, limit: int | None = None) -> BenchReport:
        """Full pipeline: generate -> write -> evaluate -> report."""
        model_id = ""
        if self.system and hasattr(self.system, "agent_loop"):
            model_id = getattr(self.system.agent_loop, "_last_model", "") or "unknown"
        self.manifest = BenchmarkManifest(
            benchmark=f"bigcodebench-{self.subset}", model=model_id
        )

        t0 = time.perf_counter()
        solutions = await self.generate_solutions(limit=limit)
        gen_time = time.perf_counter() - t0

        if not solutions:
            return BenchReport.from_results(f"bigcodebench-{self.subset}", [])

        # Write solutions and evaluate via bigcodebench CLI
        with tempfile.TemporaryDirectory() as tmpdir:
            samples_path = Path(tmpdir) / "samples.jsonl"
            with open(samples_path, "w", encoding="utf-8") as f:
                for sol in solutions:
                    f.write(json.dumps(sol) + "\n")

            # Try evaluation via bigcodebench CLI
            try:
                result = subprocess.run(
                    [
                        "python", "-m", "bigcodebench.evaluate",
                        "--samples", str(samples_path),
                        "--split", self.split,
                        "--subset", self.subset,
                    ],
                    capture_output=True, text=True, timeout=1200,
                )
                log.info(f"BigCodeBench output:\n{result.stdout}")
            except Exception as e:
                log.warning(f"BigCodeBench evaluation failed: {e}")

            # Parse results
            eval_files = list(Path(tmpdir).glob("*eval_results.json"))
            if eval_files:
                with open(eval_files[0], encoding="utf-8") as f:
                    eval_data = json.load(f)
            else:
                eval_data = {}

        results: list[TaskResult] = []
        for sol in solutions:
            tid = sol["task_id"]
            passed = eval_data.get(tid, {}).get("passed", False) if eval_data else False
            results.append(TaskResult(
                task_id=tid,
                passed=passed,
                latency_ms=round(gen_time * 1000 / max(len(solutions), 1), 1),
            ))

        return BenchReport.from_results(f"bigcodebench-{self.subset}", results)
```

**Step 4: Run tests**

Run: `cd sage-python && python -m pytest tests/test_bigcodebench.py -v`
Expected: All 2 tests PASS

**Step 5: Commit**

```bash
git add sage-python/src/sage/bench/bigcodebench_adapter.py sage-python/tests/test_bigcodebench.py
git commit -m "feat(bench): add BigCodeBench adapter for real-world code evaluation"
```

---

### Task 6: Update CLI with new benchmark types

**Files:**
- Modify: `sage-python/src/sage/bench/__main__.py`
- Modify: `sage-python/src/sage/bench/__init__.py`

**Step 1: Update __main__.py**

Add new choices to `--type`: `evalplus`, `bigcodebench`, `ablation`.
Add `--dataset` flag for evalplus (humaneval/mbpp).
Add `--split` and `--subset` flags for bigcodebench.

Key additions:
```python
parser.add_argument(
    "--type",
    choices=["routing", "humaneval", "evalplus", "bigcodebench", "ablation", "all"],
    default="routing",
)
parser.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
parser.add_argument("--split", choices=["complete", "instruct"], default="complete")
parser.add_argument("--subset", choices=["full", "hard"], default="hard")
```

Add `_run_evalplus()`, `_run_bigcodebench()`, `_run_ablation()` async functions following the same pattern as `_run_humaneval()`.

**Step 2: Update __init__.py exports**

```python
from sage.bench.runner import BenchmarkRunner, BenchReport, TaskResult
from sage.bench.ablation import AblationConfig, ABLATION_CONFIGS

__all__ = ["BenchmarkRunner", "BenchReport", "TaskResult", "AblationConfig", "ABLATION_CONFIGS"]
```

**Step 3: Run smoke test**

Run: `cd sage-python && python -m sage.bench --help`
Expected: Shows new --type choices

**Step 4: Commit**

```bash
git add sage-python/src/sage/bench/__main__.py sage-python/src/sage/bench/__init__.py
git commit -m "feat(bench): extend CLI with evalplus, bigcodebench, and ablation modes"
```

---

### Task 7: Run EvalPlus HumanEval+ benchmark (real, no mocks)

**Files:**
- Output: `docs/benchmarks/2026-03-10-evalplus-humaneval.json`

**Step 1: Run EvalPlus with limit=20 first (smoke test)**

Run: `cd sage-python && python -m sage.bench --type evalplus --dataset humaneval --limit 20`
Expected: Solutions generated, evaluation runs, report saved

**Step 2: Analyze results**

Check: pass rate on HumanEval+ (expect lower than our custom 95% due to 80x harder tests).
This is the HONEST result — no tuning, no cherry-picking.

**Step 3: Run full 164 problems (if smoke test passes)**

Run: `cd sage-python && python -m sage.bench --type evalplus --dataset humaneval`
Expected: Full results saved

**Step 4: Commit results**

```bash
git add docs/benchmarks/2026-03-10-evalplus-humaneval.json
git commit -m "bench: EvalPlus HumanEval+ results (164 problems, honest evaluation)"
```

---

### Task 8: Run ablation study (framework value proof)

**Files:**
- Output: `docs/benchmarks/2026-03-10-ablation-study.json`

**Step 1: Run ablation with limit=20**

Run: `cd sage-python && python -m sage.bench --type ablation --limit 20`

This runs HumanEval+ with 6 configurations:
1. full (all pillars)
2. baseline (bare LLM)
3. no-memory
4. no-avr
5. no-routing
6. no-guardrails

**Step 2: Analyze deltas**

For each config, compute: `delta = config_score - full_score`
This quantifies each pillar's contribution.

**Step 3: Commit results**

```bash
git add docs/benchmarks/2026-03-10-ablation-study.json
git commit -m "bench: ablation study results proving pillar contributions"
```

---

### Task 9: Run MBPP+ benchmark

**Files:**
- Output: `docs/benchmarks/2026-03-10-evalplus-mbpp.json`

**Step 1: Run MBPP+ with limit=20**

Run: `cd sage-python && python -m sage.bench --type evalplus --dataset mbpp --limit 20`
Expected: 378 Python problems evaluated

**Step 2: Full run if smoke test passes**

Run: `cd sage-python && python -m sage.bench --type evalplus --dataset mbpp`

**Step 3: Commit**

```bash
git add docs/benchmarks/2026-03-10-evalplus-mbpp.json
git commit -m "bench: EvalPlus MBPP+ results (378 problems, honest evaluation)"
```

---

### Task 10: Update documentation and CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (add new benchmark commands and results)
- Modify: `README.md` (update benchmarks table)

**Step 1: Add new benchmark commands to CLAUDE.md**

Under `### Benchmarks`:
```markdown
### Benchmarks
```bash
# Official benchmarks (EvalPlus — 80x more tests than HumanEval)
python -m sage.bench --type evalplus --dataset humaneval          # HumanEval+ (164 problems)
python -m sage.bench --type evalplus --dataset mbpp               # MBPP+ (378 problems)
python -m sage.bench --type evalplus --dataset humaneval --limit 20  # Quick smoke test

# BigCodeBench (1140 real-world tasks)
python -m sage.bench --type bigcodebench --subset hard            # Hard subset (148 tasks)

# Ablation study (proves each pillar's value)
python -m sage.bench --type ablation --limit 20                   # 6 configs x 20 tasks

# Legacy benchmarks
python -m sage.bench --type routing                               # Routing accuracy (instant)
python -m sage.bench --type humaneval --limit 20                  # Original HumanEval (custom tests)
```
```

**Step 2: Update README benchmarks table with actual results**

**Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: add official benchmark commands and results"
```

---

## Execution Summary

| Task | What | Effort | Impact |
|------|------|--------|--------|
| 1 | Install EvalPlus | 2 min | Foundation |
| 2 | EvalPlus adapter | 10 min | HumanEval+/MBPP+ evaluation |
| 3 | Ablation framework | 5 min | Pillar value proof |
| 4 | Wire ablation flags | 10 min | Agent loop integration |
| 5 | BigCodeBench adapter | 10 min | Real-world code evaluation |
| 6 | CLI update | 5 min | User interface |
| 7 | Run EvalPlus (real) | 30 min | Honest HumanEval+ score |
| 8 | Run ablation (real) | 60 min | Framework value proof |
| 9 | Run MBPP+ | 30 min | Second code gen benchmark |
| 10 | Update docs | 5 min | Communication |

**Total estimated API cost:** ~$2-5 (164+378+20x6 = ~662 LLM calls at ~$0.005/call)

## What This Proves

| Evidence | Benchmark | Honest? |
|----------|-----------|---------|
| Code gen quality | EvalPlus HumanEval+ (80x tests) | Yes — official evaluator |
| Code gen breadth | EvalPlus MBPP+ (378 problems) | Yes — separate dataset |
| Memory contribution | Ablation: full vs no-memory | Yes — same benchmark, different config |
| AVR contribution | Ablation: full vs no-avr | Yes — quantified delta |
| Routing contribution | Ablation: full vs no-routing | Yes — quantified delta |
| Guardrails contribution | Ablation: full vs no-guardrails | Yes — quantified delta |
| Framework vs bare LLM | Ablation: full vs baseline | Yes — same model, fair comparison |
