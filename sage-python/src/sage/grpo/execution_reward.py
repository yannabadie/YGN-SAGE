"""Execution reward for GRPO v4 — real sandbox execution + TopologyReward Rust.

Fixes 5 critical issues from v3:
1. Real subprocess execution (was len(response) > 20)
2. Edges built in TopologyGraph (was 0 edges -> s_edge always 1.0)
3. System inferred from difficulty field (was hardcoded to 2)
4-5: format_reward range and LR fixed in train_topology_grpo.py

Graduated rewards (AgentConductor-style):
  PASSED=1.5, WRONG_ANSWER=1.0, RUNTIME_ERROR=0.7,
  TIMEOUT=0.5, RUNS_OK=0.3, CRASH=0.0
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re as _re
import sys as _sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("grpo_v2")

CONCURRENCY = int(os.environ.get("SAGE_DEEPSEEK_CONCURRENCY", "8"))

# Rust reward infrastructure (optional)
try:
    from sage_core import (
        TopologyReward, TopologyDensity, TopologyGraph,
        TopologyNode, TopologyEdge, PyHybridVerifier,
    )
    _reward_scorer = TopologyReward()
    _density_scorer = TopologyDensity()
    _verifier = PyHybridVerifier()
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False

# Re-use SandboxResult from existing infra
from sage.tools.sandbox_executor import SandboxResult

# Lazy-loaded provider
_PROVIDER = None


# ── Stats ────────────────────────────────────────────────────────

class ExecutionStats:
    def __init__(self):
        self.total = 0
        self.success = 0
        self.failed = 0
        self.timeout = 0
        self.total_tokens = 0
        self.total_latency = 0.0
        self.status_counts: dict[str, int] = {}

    def record(self, *, success: bool, timeout: bool = False,
               tokens: int = 0, latency: float = 0.0, status: str = ""):
        self.total += 1
        if success:
            self.success += 1
        elif timeout:
            self.timeout += 1
        else:
            self.failed += 1
        self.total_tokens += tokens
        self.total_latency += latency
        if status:
            self.status_counts[status] = self.status_counts.get(status, 0) + 1
        if self.total % 10 == 0:
            dist = " ".join(f"{k}={v}" for k, v in sorted(self.status_counts.items()))
            log.info(
                "Exec: %d total | %d ok (%.0f%%) | %d fail | %d timeout | "
                "avg %.1fs | %s",
                self.total, self.success,
                100 * self.success / max(self.total, 1),
                self.failed, self.timeout,
                self.total_latency / max(self.total, 1),
                dist,
            )


_stats = ExecutionStats()


# ── Code extraction ──────────────────────────────────────────────

def extract_python_code(response: str) -> str | None:
    """Extract Python code block from LLM response (handles <think> tags)."""
    text = _re.sub(r'<think>.*?</think>', '', response, flags=_re.DOTALL).strip()
    if not text:
        return None
    # 1. Fenced ```python ... ```
    match = _re.search(r'```python\s*\n(.*?)```', text, _re.DOTALL)
    if match:
        return match.group(1).strip()
    # 2. Fenced ``` ... ``` (no language tag)
    match = _re.search(r'```\s*\n(.*?)```', text, _re.DOTALL)
    if match:
        code = match.group(1).strip()
        if any(code.startswith(kw) for kw in ('def ', 'import ', 'class ', 'from ', '#')):
            return code
    # 3. Entire response looks like code
    if any(text.startswith(kw) for kw in ('def ', 'import ', 'class ', 'from ')):
        return text
    return None


# ── Sandbox execution ────────────────────────────────────────────

async def run_code_sandbox(code: str, timeout: int = 30) -> SandboxResult:
    """Execute raw Python code in an isolated subprocess. No stdin wrapper."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8",
    )
    script_path = tmp.name
    try:
        tmp.write(code)
        tmp.close()
        proc = await asyncio.create_subprocess_exec(
            _sys.executable, script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
            return SandboxResult(
                stdout=stdout_b.decode("utf-8", errors="replace"),
                stderr=stderr_b.decode("utf-8", errors="replace"),
                exit_code=proc.returncode or 0,
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return SandboxResult(stdout="", stderr="TIMEOUT", exit_code=-1, timed_out=True)
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


async def compute_execution_score(
    code: str, task_id: str, timeout: int = 30,
) -> tuple[float, str]:
    """Execute code and return (score, status). Graduated like AgentConductor.

    Without test cases (most common path for BigCodeBench prompts):
      RUNS_OK=0.3, TIMEOUT=0.5, CRASH=0.0

    With test cases (HumanEval if task_id matches):
      PASSED=1.5, WRONG_ANSWER=1.0, RUNTIME_ERROR=0.7, TIMEOUT=0.5
    """
    test_cases = _load_test_cases()
    test_code = test_cases.get(task_id, "")

    if test_code:
        full_code = code + "\n\n" + test_code
        result = await run_code_sandbox(full_code, timeout=timeout)
        if result.exit_code == 0:
            return 1.5, "PASSED"
        if result.timed_out:
            return 0.5, "TIMEOUT"
        if "ImportError" in result.stderr or "ModuleNotFoundError" in result.stderr:
            return 0.3, "IMPORT_ERROR"
        if any(kw in result.stderr for kw in ("Error", "Exception", "Traceback")):
            return 0.7, "RUNTIME_ERROR"
        return 1.0, "WRONG_ANSWER"

    # No test cases — run code standalone
    result = await run_code_sandbox(code, timeout=timeout)
    if result.exit_code == 0:
        return 0.3, "RUNS_OK"
    if result.timed_out:
        return 0.5, "TIMEOUT"
    return 0.0, "CRASH"


# Test case loading (HumanEval — loaded once)
_TEST_CASES: dict[str, str] | None = None


def _load_test_cases() -> dict[str, str]:
    global _TEST_CASES
    if _TEST_CASES is not None:
        return _TEST_CASES
    _TEST_CASES = {}
    for path in [
        Path(__file__).parent.parent / "bench" / "humaneval_data.json",
        Path("src/sage/bench/humaneval_data.json"),
    ]:
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                for item in data:
                    tid = item.get("task_id", "")
                    test = item.get("test", "")
                    if tid and test:
                        _TEST_CASES[tid] = test
                log.info("Loaded %d HumanEval test cases from %s", len(_TEST_CASES), path)
            except Exception as exc:
                log.warning("Failed to load test cases: %s", str(exc)[:80])
            break
    return _TEST_CASES


# ── Provider ─────────────────────────────────────────────────────

def _get_provider():
    global _PROVIDER
    if _PROVIDER is not None:
        return _PROVIDER
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        log.warning("DEEPSEEK_API_KEY not set")
        return None
    try:
        from sage.providers.openai_compat import OpenAICompatProvider
        _PROVIDER = OpenAICompatProvider(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            provider_name="deepseek",
        )
        log.info("DeepSeek Reasoner initialized (concurrency=%d)", CONCURRENCY)
        return _PROVIDER
    except Exception as exc:
        log.warning("DeepSeek init failed: %s", str(exc)[:100])
        return None


# ── Topology graph builder (FIX #2 + #3) ────────────────────────

def _build_topology_graph(topology_dict: dict) -> Any:
    """Build TopologyGraph with nodes AND edges. System from difficulty."""
    if not _RUST_AVAILABLE:
        return None
    nodes = topology_dict.get("nodes", [])
    if not nodes:
        return None

    # FIX #3: system from difficulty field (model generates this in YAML)
    difficulty = topology_dict.get("difficulty", "moderate")
    system = {"simple": 1, "moderate": 2, "complex": 3}.get(
        str(difficulty).lower(), 2
    )

    graph = TopologyGraph("sequential")

    for node_data in nodes:
        if not isinstance(node_data, dict):
            continue
        node = TopologyNode(
            role=node_data.get("role", "agent"),
            model_id=node_data.get("model_tier", ""),
            system=system,
            prompt=node_data.get("prompt", ""),
        )
        graph.add_node(node)

    # FIX #2: build edges from YAML
    for edge_data in topology_dict.get("edges", []):
        if not isinstance(edge_data, dict):
            continue
        from_idx = edge_data.get("from_idx", 0)
        to_idx = edge_data.get("to_idx", 0)
        flow_type = edge_data.get("flow_type", "message")
        if 0 <= from_idx < graph.node_count() and 0 <= to_idx < graph.node_count():
            graph.add_edge(from_idx, to_idx, TopologyEdge(flow_type))

    # Fallback: if no edges but >1 node, create sequential chain
    if graph.edge_count() == 0 and graph.node_count() > 1:
        for i in range(graph.node_count() - 1):
            graph.add_edge(i, i + 1, TopologyEdge("message"))

    return graph


# ── Rust reward computation ──────────────────────────────────────

def _compute_rust_reward(graph: Any, exec_score: float, system: int) -> float:
    """Combine Rust structural/density with graduated execution score."""
    exec_norm = min(exec_score / 1.5, 1.0)
    if not _RUST_AVAILABLE or graph is None:
        return exec_norm
    try:
        density = _density_scorer.compute(graph, system)
        verification = _verifier.verify(graph)
        structural_score = 1.0 if verification.valid else 0.5

        reward = _reward_scorer.compute(
            execution_passed=(exec_score >= 1.0),
            structural_score=structural_score,
            density_score=density.s_complex,
            temporal_score=None,
        )
        score = reward.total
        if density.over_budget:
            score *= 0.5
        return float(score)
    except Exception:
        return exec_norm


# ── Main evaluation (FIX #1) ────────────────────────────────────

async def evaluate_topology(
    task: str, topology_dict: dict, semaphore: asyncio.Semaphore,
) -> float:
    """Execute topology via DeepSeek Reasoner, then run code in real sandbox."""
    async with semaphore:
        t0 = time.time()

        difficulty = topology_dict.get("difficulty", "moderate")
        system = {"simple": 1, "moderate": 2, "complex": 3}.get(
            str(difficulty).lower(), 2
        )

        graph = _build_topology_graph(topology_dict)

        provider = _get_provider()
        if provider is None:
            return 0.0

        from sage.llm.base import Message, Role, LLMConfig
        nodes = topology_dict.get("nodes", [])
        system_prompt = (
            nodes[0].get("prompt", "You are a helpful assistant.")
            if nodes else "You are a helpful assistant."
        )
        messages = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=task[:2000]),
        ]
        config = LLMConfig(provider="deepseek", model="deepseek-reasoner")

        try:
            response = await asyncio.wait_for(
                provider.generate(messages=messages, config=config),
                timeout=120.0,
            )
            result_text = response.content or ""
        except (asyncio.TimeoutError, Exception):
            _stats.record(success=False, timeout=True, latency=time.time() - t0, status="API_TIMEOUT")
            return 0.0

        # FIX #1: extract code and run in real sandbox
        code = extract_python_code(result_text)
        if code is None:
            _stats.record(success=False, latency=time.time() - t0, status="NO_CODE")
            return _compute_rust_reward(graph, 0.0, system)

        task_id = topology_dict.get("_task_id", "")
        exec_score, status = await compute_execution_score(code, task_id, timeout=30)

        tokens = len(result_text) // 4
        _stats.record(
            success=(exec_score >= 1.0), tokens=tokens,
            latency=time.time() - t0, status=status,
        )

        return _compute_rust_reward(graph, exec_score, system)


# ── Batch entry point (called by GRPOTrainer) ───────────────────

def execution_reward_batch(completions: list, **kwargs) -> list[float]:
    """Sync wrapper for batch execution reward. Called by GRPOTrainer."""
    prompts = kwargs.get("prompts", [])

    tasks_and_topos = []
    indices = []
    for i, completion in enumerate(completions):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        prompt = prompts[i] if i < len(prompts) else ""
        if isinstance(prompt, list):
            prompt = prompt[-1]["content"] if prompt else ""

        topo = None
        try:
            topo = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            try:
                topo = yaml.safe_load(text)
            except Exception:
                pass

        if isinstance(topo, dict) and "nodes" in topo and len(topo.get("nodes", [])) > 0:
            tasks_and_topos.append((prompt, topo))
            indices.append(i)

    n_valid = len(tasks_and_topos)
    n_invalid = len(completions) - n_valid
    if (n_valid + n_invalid) > 0 and (n_valid + n_invalid) % 8 == 0:
        log.info(
            "Parse: %d valid (%.0f%%) | %d invalid",
            n_valid, 100 * n_valid / max(n_valid + n_invalid, 1), n_invalid,
        )

    rewards = [0.0] * len(completions)
    if not tasks_and_topos:
        return rewards

    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def _run_all():
        coros = [
            evaluate_topology(task, topo, semaphore)
            for task, topo in tasks_and_topos
        ]
        return await asyncio.gather(*coros, return_exceptions=True)

    try:
        results = asyncio.run(_run_all())
    except RuntimeError:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            results = pool.submit(lambda: asyncio.run(_run_all())).result(timeout=300)

    for idx, result in zip(indices, results):
        if isinstance(result, (int, float)):
            rewards[idx] = float(result)

    return rewards
