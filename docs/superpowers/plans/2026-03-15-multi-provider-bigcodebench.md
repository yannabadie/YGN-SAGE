# Multi-Provider Wiring + BigCodeBench Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire all 6 LLM providers into ProviderPool for dynamic per-node model selection, then add BigCodeBench adapter for non-saturated benchmarking.

**Architecture:** Fix 3 bugs in boot.py/connector.py that prevent multi-provider from working (providers dict never passed, Google excluded, DeepSeek env var mismatch). Then add BigCodeBench adapter following the existing EvalPlus pattern. Rust side needs no changes — it's a pure decision engine returning model_ids; Python handles provider resolution.

**Tech Stack:** Python 3.12+, google-genai, openai (for OpenAI-compat providers), bigcodebench pip package, pytest, subprocess sandbox.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `sage-python/src/sage/providers/connector.py:41` | Modify | Fix DEEP_SEEK_API_KEY → accept both spellings |
| `sage-python/src/sage/boot.py:840-951` | Modify | Wire all providers into ProviderPool |
| `sage-python/tests/test_provider_pool_wiring.py` | Create | E2E test for multi-provider resolution |
| `docker-compose.yml` | Modify | Add all 6 provider env vars |
| `sage-python/src/sage/bench/bigcodebench_bench.py` | Create | BigCodeBench adapter |
| `sage-python/src/sage/bench/__main__.py:294,356+` | Modify | Register bigcodebench CLI type |
| `sage-python/tests/test_bigcodebench.py` | Create | BigCodeBench adapter unit tests |

---

## Chunk 1: Multi-Provider Wiring

### Task 1: Fix DeepSeek env var mismatch

**Files:**
- Modify: `sage-python/src/sage/providers/connector.py:41`
- Test: `sage-python/tests/test_provider_pool_wiring.py`

- [ ] **Step 1: Write the failing test**

Create `sage-python/tests/test_provider_pool_wiring.py`:

```python
"""Tests for multi-provider wiring — ProviderPool gets all providers from boot."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


class TestDeepSeekEnvVar:
    """DeepSeek should be discoverable with both env var spellings."""

    def test_deepseek_primary_env_var(self):
        """PROVIDER_CONFIGS should use DEEPSEEK_API_KEY as primary env var."""
        from sage.providers.connector import PROVIDER_CONFIGS

        ds_cfg = next(c for c in PROVIDER_CONFIGS if c["provider"] == "deepseek")
        assert ds_cfg["api_key_env"] == "DEEPSEEK_API_KEY", (
            f"Expected DEEPSEEK_API_KEY, got {ds_cfg['api_key_env']}"
        )

    def test_deepseek_legacy_fallback_in_discovery(self):
        """discover_all() should find DeepSeek via legacy DEEP_SEEK_API_KEY too."""
        from sage.providers.connector import ProviderConnector

        # Only set the legacy spelling
        env = {"DEEP_SEEK_API_KEY": "sk-test-legacy"}
        with patch.dict(os.environ, env, clear=False):
            connector = ProviderConnector()
            # discover_all is async, but we just test the env var resolution
            api_key = os.environ.get("DEEPSEEK_API_KEY", "") or os.environ.get("DEEP_SEEK_API_KEY", "")
            assert api_key == "sk-test-legacy"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_provider_pool_wiring.py::TestDeepSeekEnvVar -v`
Expected: FAIL — `DEEP_SEEK_API_KEY` not set, `key` is empty.

- [ ] **Step 3: Fix connector.py**

In `sage-python/src/sage/providers/connector.py`, line 41, change the env var name:

```python
    {
        "provider": "deepseek",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "sdk": "openai",
    },
```

Also add a fallback in `discover_all()` after line 95:

```python
            api_key = os.environ.get(cfg["api_key_env"], "")
            # Fallback: also check legacy DEEP_SEEK_API_KEY spelling
            if not api_key and cfg["provider"] == "deepseek":
                api_key = os.environ.get("DEEP_SEEK_API_KEY", "")
```

- [ ] **Step 4: Run test**

Run: `cd sage-python && python -m pytest tests/test_provider_pool_wiring.py::TestDeepSeekEnvVar -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sage-python/src/sage/providers/connector.py sage-python/tests/test_provider_pool_wiring.py
git commit -m "fix: accept both DEEPSEEK_API_KEY and DEEP_SEEK_API_KEY spellings"
```

---

### Task 2: Wire all providers into ProviderPool

**Files:**
- Modify: `sage-python/src/sage/boot.py:840-951`
- Modify: `sage-python/tests/test_provider_pool_wiring.py`

- [ ] **Step 1: Write the failing test**

Append to `test_provider_pool_wiring.py`:

```python
class TestProviderPoolWiring:
    """ProviderPool should contain all discovered providers, not just default."""

    def test_boot_wires_google_into_pool(self):
        """Google provider should be in ProviderPool, not just OpenAI-compat."""
        from sage.boot import boot_agent_system
        from sage.events.bus import EventBus

        system = boot_agent_system(use_mock_llm=False, llm_tier="fast", event_bus=EventBus())

        pool = system.pipeline._provider_pool if system.pipeline else None
        if pool is None:
            pytest.skip("Pipeline not available")

        # ProviderPool._providers should contain "google"
        assert "google" in pool._providers, (
            f"Google not in ProviderPool._providers. Keys: {list(pool._providers.keys())}"
        )

    def test_boot_wires_multiple_providers(self):
        """ProviderPool should have >1 provider if multiple API keys are set."""
        from sage.boot import boot_agent_system
        from sage.events.bus import EventBus

        system = boot_agent_system(use_mock_llm=False, llm_tier="fast", event_bus=EventBus())

        pool = system.pipeline._provider_pool if system.pipeline else None
        if pool is None:
            pytest.skip("Pipeline not available")

        provider_count = len(pool._providers)
        print(f"  ProviderPool has {provider_count} providers: {list(pool._providers.keys())}")
        assert provider_count >= 1, "ProviderPool should have at least 1 provider"

    def test_resolve_returns_correct_provider_for_google_model(self):
        """resolve('gemini-2.5-flash') should return GoogleProvider, not default."""
        from sage.boot import boot_agent_system
        from sage.events.bus import EventBus

        system = boot_agent_system(use_mock_llm=False, llm_tier="fast", event_bus=EventBus())

        pool = system.pipeline._provider_pool if system.pipeline else None
        if pool is None:
            pytest.skip("Pipeline not available")

        provider, config = pool.resolve("gemini-2.5-flash")
        # Should resolve to the Google provider specifically
        provider_type = type(provider).__name__
        print(f"  resolve('gemini-2.5-flash') → {provider_type}")
        assert "Google" in provider_type or "google" in config.provider, (
            f"Expected Google provider, got {provider_type} (config.provider={config.provider})"
        )

    def test_resolve_returns_correct_provider_for_deepseek(self):
        """resolve('deepseek-chat') should return OpenAICompatProvider for deepseek."""
        import os
        from sage.boot import boot_agent_system
        from sage.events.bus import EventBus

        if not os.environ.get("DEEPSEEK_API_KEY") and not os.environ.get("DEEP_SEEK_API_KEY"):
            pytest.skip("DEEPSEEK_API_KEY not set")

        system = boot_agent_system(use_mock_llm=False, llm_tier="fast", event_bus=EventBus())
        pool = system.pipeline._provider_pool if system.pipeline else None
        if pool is None:
            pytest.skip("Pipeline not available")

        if "deepseek" not in pool._providers:
            pytest.skip("DeepSeek not discovered at boot")

        provider, config = pool.resolve("deepseek-chat")
        print(f"  resolve('deepseek-chat') → {type(provider).__name__}, config.provider={config.provider}")
        assert config.provider == "deepseek"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_provider_pool_wiring.py::TestProviderPoolWiring::test_boot_wires_google_into_pool -v -s`
Expected: FAIL — "Google not in ProviderPool._providers".

- [ ] **Step 3: Fix boot.py — include Google in _runtime_adapters and pass to ProviderPool**

In `sage-python/src/sage/boot.py`, replace lines 840-854 with:

```python
        from sage.providers.capabilities import CapabilityMatrix as _CapMatrix
        from sage.providers.connector import PROVIDER_CONFIGS
        from sage.providers.openai_compat import OpenAICompatProvider
        _cap_matrix = _CapMatrix()
        _discovered_providers = {p.provider for p in registry.list_available()}
        _runtime_adapters: dict[str, Any] = {}
        for _cfg in PROVIDER_CONFIGS:
            _pname = _cfg["provider"]
            if _pname not in _discovered_providers:
                continue
            _api_key = os.environ.get(_cfg["api_key_env"], "")
            # Fallback for legacy env var spelling
            if not _api_key and _pname == "deepseek":
                _api_key = os.environ.get("DEEP_SEEK_API_KEY", "")
            if not _api_key:
                continue
            if _cfg.get("sdk") == "google-genai":
                from sage.llm.google import GoogleProvider
                _runtime_adapters[_pname] = GoogleProvider(api_key=_api_key)
            else:
                _runtime_adapters[_pname] = OpenAICompatProvider(
                    api_key=_api_key,
                    base_url=_cfg.get("base_url"),
                    provider_name=_pname,
                )
        _cap_matrix.populate_from_providers(
            list(_discovered_providers), adapters=_runtime_adapters,
        )
```

Then replace lines 947-951 (ProviderPool construction):

```python
            _provider_pool = ProviderPool(
                default_provider=provider,
                registry=registry,
                default_config=llm_config,
                providers=_runtime_adapters,
            )
```

Also add a log line after ProviderPool creation:

```python
            _log.info("ProviderPool: %d live providers — %s", len(_runtime_adapters), list(_runtime_adapters.keys()))
```

- [ ] **Step 4: Run tests**

Run: `cd sage-python && python -m pytest tests/test_provider_pool_wiring.py -v -s`
Expected: All tests PASS (some may skip if API keys not set).

- [ ] **Step 5: Run full test suite for regressions**

Run: `cd sage-python && python -m pytest tests/ -q --ignore=tests/test_a2a_server.py`
Expected: 1443+ passed, 0 failed.

- [ ] **Step 6: Commit**

```bash
git add sage-python/src/sage/boot.py sage-python/tests/test_provider_pool_wiring.py
git commit -m "feat: wire all discovered providers into ProviderPool for multi-model execution"
```

---

### Task 3: Update docker-compose with all provider env vars

**Files:**
- Modify: `docker-compose.yml`

- [ ] **Step 1: Add all 6 provider env vars to both services**

In `docker-compose.yml`, update the `sage` service environment:

```yaml
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:-}
      - GROK_API_KEY=${GROK_API_KEY:-}
      - KIMI_API_KEY=${KIMI_API_KEY:-}
      - MINIMAX_API_KEY=${MINIMAX_API_KEY:-}
      - SAGE_DASHBOARD_TOKEN=${SAGE_DASHBOARD_TOKEN:-}
      - PYTHONIOENCODING=utf-8
```

Same for `dashboard` service.

- [ ] **Step 2: Commit**

```bash
git add docker-compose.yml
git commit -m "feat: pass all 6 provider API keys through docker-compose"
```

---

### Task 4: Rebuild Rust with SMT feature

**Files:**
- No code changes — build command only.

- [ ] **Step 1: Rebuild sage-core with all features**

Run: `cd sage-core && maturin develop --features smt,onnx,cognitive`
Expected: Build succeeds, `SmtVerifier` available in sage_core.

- [ ] **Step 2: Verify**

Run: `python -c "from sage_core import SmtVerifier; v = SmtVerifier(); print('OxiZ OK:', v.verify_arithmetic(2, 3, 6, 'multiply'))"`
Expected: `OxiZ OK: True`

- [ ] **Step 3: No commit needed** (build artifact only)

---

## Chunk 2: BigCodeBench Adapter

### Task 5: Install bigcodebench and create adapter

**Files:**
- Create: `sage-python/src/sage/bench/bigcodebench_bench.py`
- Test: `sage-python/tests/test_bigcodebench.py`

- [ ] **Step 1: Install bigcodebench**

Run: `pip install bigcodebench --upgrade`

- [ ] **Step 2: Write the unit test**

Create `sage-python/tests/test_bigcodebench.py`:

```python
"""Tests for BigCodeBench adapter."""

from __future__ import annotations

import pytest


class TestBigCodeBenchLoader:
    """Test dataset loading and task format."""

    def test_import(self):
        """Adapter module is importable."""
        from sage.bench.bigcodebench_bench import BigCodeBenchBench
        assert callable(BigCodeBenchBench)

    def test_load_dataset_hard(self):
        """Hard subset loads and has expected structure."""
        try:
            from bigcodebench.data import get_bigcodebench
        except ImportError:
            pytest.skip("bigcodebench not installed")

        problems = get_bigcodebench(subset="hard")
        assert len(problems) > 100, f"Expected 100+ tasks, got {len(problems)}"

        # Check first task structure
        first_id = next(iter(problems))
        task = problems[first_id]
        assert "task_id" in task
        assert "instruct_prompt" in task
        assert "test" in task
        assert "entry_point" in task
        print(f"  Loaded {len(problems)} hard tasks, first: {first_id}")

    def test_load_dataset_full(self):
        """Full dataset loads 1140 tasks."""
        try:
            from bigcodebench.data import get_bigcodebench
        except ImportError:
            pytest.skip("bigcodebench not installed")

        problems = get_bigcodebench(subset="full")
        assert len(problems) >= 1100, f"Expected 1100+ tasks, got {len(problems)}"
        print(f"  Loaded {len(problems)} full tasks")


class TestBigCodeBenchEval:
    """Test local evaluation via subprocess."""

    def test_eval_correct_solution(self):
        """A correct canonical solution should pass its tests."""
        try:
            from bigcodebench.data import get_bigcodebench
        except ImportError:
            pytest.skip("bigcodebench not installed")
        from sage.bench.bigcodebench_bench import BigCodeBenchBench

        problems = get_bigcodebench(subset="hard")
        first_id = next(iter(problems))
        task = problems[first_id]

        # Use canonical solution — should pass
        passed = BigCodeBenchBench._evaluate_solution(
            solution=task["canonical_solution"],
            test_code=task["test"],
            entry_point=task["entry_point"],
            task_id=first_id,
            timeout=30,
        )
        assert passed, f"Canonical solution for {first_id} should pass its own tests"

    def test_eval_empty_solution_fails(self):
        """An empty solution should fail tests."""
        try:
            from bigcodebench.data import get_bigcodebench
        except ImportError:
            pytest.skip("bigcodebench not installed")
        from sage.bench.bigcodebench_bench import BigCodeBenchBench

        problems = get_bigcodebench(subset="hard")
        first_id = next(iter(problems))
        task = problems[first_id]

        passed = BigCodeBenchBench._evaluate_solution(
            solution="",
            test_code=task["test"],
            entry_point=task["entry_point"],
            task_id=first_id,
            timeout=10,
        )
        assert not passed, "Empty solution should fail"
```

- [ ] **Step 3: Run to verify tests fail (adapter not created yet)**

Run: `cd sage-python && python -m pytest tests/test_bigcodebench.py::TestBigCodeBenchLoader::test_import -v`
Expected: FAIL — `ImportError: cannot import name 'BigCodeBenchBench'`

- [ ] **Step 4: Create the adapter**

Create `sage-python/src/sage/bench/bigcodebench_bench.py`:

```python
"""BigCodeBench adapter: 1140 real-world coding tasks (ICLR '25).

Wraps the bigcodebench package to generate solutions via AgentSystem,
evaluate locally with unittest subprocess, and optionally run official CLI.

Install: pip install bigcodebench
Dataset: https://huggingface.co/datasets/bigcode/bigcodebench
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from sage.bench.humaneval import extract_code
from sage.bench.runner import BenchReport, TaskResult

log = logging.getLogger(__name__)


def _load_dataset(subset: str = "full") -> dict[str, dict[str, Any]]:
    """Load BigCodeBench dataset.

    Args:
        subset: "full" (1140 tasks) or "hard" (~150 tasks).

    Returns:
        Dict keyed by task_id (e.g. "BigCodeBench/0").
    """
    from bigcodebench.data import get_bigcodebench
    return get_bigcodebench(subset=subset)


class BigCodeBenchBench:
    """BigCodeBench adapter for SAGE.

    Args:
        system: AgentSystem to benchmark. If None, no solutions generated.
        event_bus: EventBus for emitting BENCH_RESULT events.
        subset: "full" (1140 tasks) or "hard" (~150 tasks).
        split: "instruct" (NL prompt) or "complete" (docstring prompt).
        task_timeout: Max seconds per task for LLM generation.
        eval_timeout: Max seconds per task for test evaluation.
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
        """Run BigCodeBench benchmark.

        Args:
            limit: Max number of tasks to run (None = all).

        Returns:
            BenchReport with pass@1 results.
        """
        problems = _load_dataset(self.subset)
        task_ids = list(problems.keys())
        if limit:
            task_ids = task_ids[:limit]

        results: list[TaskResult] = []
        passed_count = 0

        for i, task_id in enumerate(task_ids):
            task = problems[task_id]
            prompt_key = "instruct_prompt" if self.split == "instruct" else "complete_prompt"
            prompt = task.get(prompt_key, task.get("instruct_prompt", ""))

            t0 = time.time()
            solution = ""
            error = ""

            # Generate solution
            if self.system:
                try:
                    raw = await asyncio.wait_for(
                        self.system.run(prompt),
                        timeout=self.task_timeout,
                    )
                    solution = extract_code(raw, task["entry_point"])
                except asyncio.TimeoutError:
                    error = "TIMEOUT"
                except Exception as exc:
                    error = str(exc)[:200]

            latency_ms = (time.time() - t0) * 1000

            # Evaluate
            if solution and not error:
                task_passed = self._evaluate_solution(
                    solution=solution,
                    test_code=task["test"],
                    entry_point=task["entry_point"],
                    task_id=task_id,
                    timeout=self.eval_timeout,
                )
            else:
                task_passed = False
                if not error:
                    error = "no solution generated"

            if task_passed:
                passed_count += 1

            results.append(TaskResult(
                task_id=task_id,
                passed=task_passed,
                latency_ms=latency_ms,
                error=error,
            ))

            status = "PASS" if task_passed else "FAIL"
            log.info("[%d/%d] %s %s (%.0fms)", i + 1, len(task_ids), status, task_id, latency_ms)

        total = len(results)
        return BenchReport(
            benchmark=f"bigcodebench-{self.subset}-{self.split}",
            total=total,
            passed=passed_count,
            failed=total - passed_count,
            errors=sum(1 for r in results if r.error),
            pass_rate=passed_count / total if total else 0.0,
            avg_latency_ms=sum(r.latency_ms for r in results) / total if total else 0.0,
            avg_cost_usd=0.0,
            routing_breakdown={},
            results=results,
        )

    @staticmethod
    def _evaluate_solution(
        solution: str,
        test_code: str,
        entry_point: str,
        task_id: str,
        timeout: float = 30.0,
    ) -> bool:
        """Evaluate a solution by running its unittest test cases in a subprocess.

        Composes solution code + test code into a single script, runs it with
        `python -m pytest` (or unittest), returns True if all tests pass.
        """
        # Compose the full test script.
        # BigCodeBench tests may need imports from the task setup.
        # The solution should include all necessary imports.
        script = f"""{solution}

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
                ["python", tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode == 0

        except subprocess.TimeoutExpired:
            log.debug("Eval timeout for %s", task_id)
            return False
        except Exception as exc:
            log.debug("Eval error for %s: %s", task_id, exc)
            return False
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
```

- [ ] **Step 5: Run tests**

Run: `cd sage-python && python -m pytest tests/test_bigcodebench.py -v -s`
Expected: All tests PASS (loader + eval).

- [ ] **Step 6: Commit**

```bash
git add sage-python/src/sage/bench/bigcodebench_bench.py sage-python/tests/test_bigcodebench.py
git commit -m "feat: add BigCodeBench adapter (1140 tasks, ICLR '25, non-saturated)"
```

---

### Task 6: Register BigCodeBench in CLI

**Files:**
- Modify: `sage-python/src/sage/bench/__main__.py:294`

- [ ] **Step 1: Add "bigcodebench" to --type choices**

In `__main__.py`, line 294, add `"bigcodebench"` to the choices list:

```python
        choices=["routing", "humaneval", "evalplus", "ablation", "routing_gt", "memory_ablation", "evolution_ablation", "swebench", "heterogeneous", "gaia", "bigcodebench", "all"],
```

- [ ] **Step 2: Add --subset argument extension**

In `__main__.py`, update the `--dataset` argument (line 312) to include bigcodebench subsets, or add a new `--subset` argument after line 350:

```python
    parser.add_argument(
        "--subset",
        choices=["full", "hard"],
        default="full",
        help="BigCodeBench subset: full (1140) or hard (~150)",
    )
    parser.add_argument(
        "--split",
        choices=["instruct", "complete"],
        default="instruct",
        help="BigCodeBench split: instruct (NL) or complete (docstring)",
    )
```

- [ ] **Step 3: Add handler function**

Add after the existing handler functions:

```python
async def _run_bigcodebench(output: str | None, limit: int | None, subset: str, split: str) -> None:
    from sage.bench.bigcodebench_bench import BigCodeBenchBench

    if os.environ.get("GOOGLE_API_KEY"):
        system, bus = _boot_system()
        bench = BigCodeBenchBench(system=system, event_bus=bus, subset=subset, split=split)
    else:
        bench = BigCodeBenchBench(subset=subset, split=split)

    report = await bench.run(limit=limit)
    _print_report(report)
    _save_report(report, bench, output, f"bigcodebench-{subset}-{split}")
```

- [ ] **Step 4: Wire handler in main()**

After the existing `if args.type == "gaia":` block, add:

```python
    if args.type == "bigcodebench":
        asyncio.run(_run_bigcodebench(args.output, args.limit, args.subset, args.split))
```

- [ ] **Step 5: Verify CLI works**

Run: `cd sage-python && python -m sage.bench --help 2>&1 | grep bigcodebench`
Expected: `bigcodebench` appears in the --type choices list.

- [ ] **Step 6: Commit**

```bash
git add sage-python/src/sage/bench/__main__.py
git commit -m "feat: register BigCodeBench in bench CLI (--type bigcodebench)"
```

---

### Task 7: Update documentation

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add BigCodeBench to benchmark commands section**

In `CLAUDE.md`, in the `### Benchmarks` section, add:

```bash
# BigCodeBench (ICLR '25, 1140 tasks, non-saturated ~62% SOTA)
python -m sage.bench --type bigcodebench --subset hard --limit 20    # Hard subset smoke
python -m sage.bench --type bigcodebench --subset full --limit 50    # Full subset sample
python -m sage.bench --type bigcodebench --subset hard --split instruct  # NL instructions
```

- [ ] **Step 2: Update Required Environment Variables section**

Add all provider env vars:

```bash
export GOOGLE_API_KEY="..."                  # Required for Gemini models
export OPENAI_API_KEY="..."                  # Optional: OpenAI GPT-5.x
export DEEPSEEK_API_KEY="..."               # Optional: DeepSeek models
export GROK_API_KEY="..."                    # Optional: xAI Grok models
export KIMI_API_KEY="..."                    # Optional: Moonshot Kimi
export MINIMAX_API_KEY="..."                # Optional: MiniMax models
```

- [ ] **Step 3: Update Known Issues to remove multi-provider claim**

Add note that multi-provider is now functional (was previously vapor).

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add BigCodeBench commands and all provider env vars to CLAUDE.md"
```

---

## Chunk 3: Validation

### Task 8: Smoke test — multi-provider boot + BigCodeBench

**Files:**
- No new files.

- [ ] **Step 1: Verify multi-provider boot**

Run: `cd sage-python && python -c "
from sage.boot import boot_agent_system
from sage.events.bus import EventBus
s = boot_agent_system(use_mock_llm=False, llm_tier='fast', event_bus=EventBus())
pool = s.pipeline._provider_pool
print(f'Providers: {list(pool._providers.keys())}')
print(f'Count: {len(pool._providers)}')
for name, p in pool._providers.items():
    print(f'  {name}: {type(p).__name__}')
"`
Expected: Shows 2+ providers (at minimum google + any available OpenAI-compat).

- [ ] **Step 2: Verify per-node resolution**

Run: `cd sage-python && python -c "
from sage.boot import boot_agent_system
from sage.events.bus import EventBus
s = boot_agent_system(use_mock_llm=False, llm_tier='fast', event_bus=EventBus())
pool = s.pipeline._provider_pool
for model in ['gemini-2.5-flash', 'deepseek-chat', 'grok-3']:
    p, c = pool.resolve(model)
    print(f'{model} → {type(p).__name__} (provider={c.provider})')
"`
Expected: Different provider types for different models.

- [ ] **Step 3: BigCodeBench smoke (1 task)**

Run: `cd sage-python && python -m sage.bench --type bigcodebench --subset hard --limit 1`
Expected: Runs 1 task, shows pass/fail result.

- [ ] **Step 4: Run full test suite**

Run: `cd sage-python && python -m pytest tests/ -q --ignore=tests/test_a2a_server.py`
Expected: 1446+ passed, 0 failed.

- [ ] **Step 5: Push**

```bash
git push origin dev
```

---

## Summary

| Sprint | Tasks | Deliverables |
|--------|-------|-------------|
| **Multi-Provider** | Tasks 1-4 | DeepSeek fix, ProviderPool wired, Docker updated, Rust rebuilt |
| **BigCodeBench** | Tasks 5-7 | Adapter + CLI + docs |
| **Validation** | Task 8 | Boot verification, per-node resolution, smoke test |

**Total:** 8 tasks. Sprint 3 (multi-model benchmarks) deferred to separate plan after multi-provider is proven working.
