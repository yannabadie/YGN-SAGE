"""Execution reward for GRPO v5 — full multi-agent topology execution.

The model (Phi-4-mini) generates TOPOLOGY YAML, not code.
The reward measures: "does this ORGANIZATION of agents solve the problem?"

Flow:
  1. Parse YAML → TopologyGraph (Rust, with edges)
  2. Execute ALL nodes via TopologyRunner (Gemini Flash, fast+cheap)
  3. Extract code from final node output
  4. Test code in sandbox + BigCodeBench tests
  5. Graduated reward → GRPO reinforces better topologies

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

# Re-use SandboxResult from existing infra
from sage.tools.sandbox_executor import SandboxResult

log = logging.getLogger("grpo_v2")

CONCURRENCY = int(os.environ.get("SAGE_GRPO_CONCURRENCY", "8"))

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

# Lazy-loaded agent provider (Gemini Flash — fast+cheap for node execution)
_AGENT_PROVIDER = None
_AGENT_MODEL = ""


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

    # Path A: code-append tests (BigCodeBench, HumanEval)
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

    # Path B: stdin/stdout tests (CodeContests)
    stdin_pairs = (_STDIN_TESTS or {}).get(task_id, [])
    if stdin_pairs:
        return await _run_stdin_tests(code, stdin_pairs, timeout)

    # Path C: No test cases — run code standalone
    result = await run_code_sandbox(code, timeout=timeout)
    if result.exit_code == 0:
        return 0.3, "RUNS_OK"
    if result.timed_out:
        return 0.5, "TIMEOUT"
    return 0.0, "CRASH"


async def _run_stdin_tests(
    code: str, pairs: list[tuple[str, str]], timeout: int = 30,
) -> tuple[float, str]:
    """Run code against stdin/stdout test pairs (competitive programming)."""
    import tempfile
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8",
    )
    tmp.write(code)
    tmp.close()
    script_path = tmp.name

    passed = 0
    wrong = 0
    errors = 0
    try:
        for inp, expected in pairs:
            proc = await asyncio.create_subprocess_exec(
                _sys.executable, script_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(input=inp.encode("utf-8")),
                    timeout=timeout // max(len(pairs), 1) + 5,
                )
                got = stdout_b.decode("utf-8", errors="replace").strip()
                want = expected.strip()
                if got == want:
                    passed += 1
                else:
                    wrong += 1
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                return 0.5, "TIMEOUT"
            except Exception:
                errors += 1
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass

    total = passed + wrong + errors
    if total == 0:
        return 0.0, "CRASH"
    if passed == total:
        return 1.5, "PASSED"
    if passed > 0:
        return 1.0, "WRONG_ANSWER"  # partial pass
    if errors > wrong:
        return 0.7, "RUNTIME_ERROR"
    return 1.0, "WRONG_ANSWER"


# Test case loading (BigCodeBench + HumanEval — loaded once)
_TEST_CASES: dict[str, str] | None = None
# CodeContests stdin/stdout pairs — separate from code-append tests
_STDIN_TESTS: dict[str, list[tuple[str, str]]] | None = None


def _load_test_cases() -> dict[str, str]:
    """Load test cases from BigCodeBench (cached on HF) + HumanEval (local JSON)."""
    global _TEST_CASES
    if _TEST_CASES is not None:
        return _TEST_CASES
    _TEST_CASES = {}

    # 1. BigCodeBench — matches our SFT task_ids ("BigCodeBench/0", etc.)
    try:
        from bigcodebench.data import get_bigcodebench
        bcb = get_bigcodebench()
        for tid, item in bcb.items():
            test = item.get("test", "")
            if tid and test:
                # BigCodeBench tests use unittest — entry_point hook is read
                # at evaluation time inside the test source, no wrap needed here.
                _TEST_CASES[tid] = test
        log.info("Loaded %d BigCodeBench test cases", len(_TEST_CASES))
    except Exception as exc:
        log.warning("BigCodeBench test cases unavailable: %s", str(exc)[:80])

    # 2. HumanEval — fallback for local testing
    for path in [
        Path(__file__).parent.parent / "bench" / "humaneval_data.json",
        Path("src/sage/bench/humaneval_data.json"),
    ]:
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                count = 0
                for item in data:
                    tid = item.get("task_id", "")
                    test = item.get("test", "")
                    if tid and test and tid not in _TEST_CASES:
                        _TEST_CASES[tid] = test
                        count += 1
                if count:
                    log.info("Loaded %d HumanEval test cases from %s", count, path)
            except Exception as exc:
                log.warning("HumanEval test cases failed: %s", str(exc)[:80])
            break

    # 3. CodeContests — stdin/stdout competitive programming (separate dict)
    global _STDIN_TESTS
    if _STDIN_TESTS is None:
        _STDIN_TESTS = {}
        for cc_path in [
            Path(__file__).parent.parent.parent.parent / "data" / "code_contests_test.parquet",
            Path("data/code_contests_test.parquet"),
        ]:
            if cc_path.exists():
                try:
                    import pandas as _pd
                    cc_df = _pd.read_parquet(str(cc_path))
                    for idx, row in cc_df.iterrows():
                        tid = f"CodeContests/{idx}"
                        pairs: list[tuple[str, str]] = []
                        for col in ("public_tests", "private_tests"):
                            tests = row.get(col)
                            if isinstance(tests, dict):
                                for inp, out in zip(
                                    tests.get("input", []), tests.get("output", [])
                                ):
                                    if inp is not None and out is not None:
                                        pairs.append((str(inp), str(out)))
                            if len(pairs) >= 10:
                                break
                        if pairs:
                            _STDIN_TESTS[tid] = pairs[:10]
                    log.info("Loaded %d CodeContests stdin/stdout tests from %s",
                             len(_STDIN_TESTS), cc_path)
                except Exception as exc:
                    log.warning("CodeContests tests failed: %s", str(exc)[:80])
                break

    log.info("Total test cases: %d code-append + %d stdin/stdout",
             len(_TEST_CASES), len(_STDIN_TESTS or {}))
    return _TEST_CASES


# ── Agent provider (for executing topology nodes) ────────────────

def _get_agent_provider():
    """Provider for executing topology nodes. DeepSeek Chat primary, Gemini fallback.

    Why DeepSeek Chat over Reasoner:
      - Same model (V3.2), same per-token price ($0.28/$0.42 per 1M)
      - Reasoner generates 2000-5000 CoT tokens per call that are DISCARDED
        (topology executor never uses them) → 3.4x more expensive for zero benefit
      - Chat: 4-8s/call vs Reasoner: 20-30s/call
      - No rate limits on DeepSeek (vs Gemini 150-300 RPM hard cap)
      - Full training cost: Chat ~$258 vs Reasoner ~$888 vs Gemini ~$1,288

    Why not Gemini Flash as primary:
      - 150-300 RPM hard cap makes 50+ concurrent requests infeasible
      - "Ghost 429" bugs reported in 2026
      - 7x more expensive output tokens ($3.00 vs $0.42 per 1M)
    """
    global _AGENT_PROVIDER, _AGENT_MODEL
    if _AGENT_PROVIDER is not None:
        return _AGENT_PROVIDER, _AGENT_MODEL

    from sage.providers.connector import get_available_providers
    from sage.providers.openai_compat import OpenAICompatProvider

    # Try providers in connector config order (single source of truth)
    for cfg in get_available_providers():
        try:
            if cfg.get("sdk") == "google-genai":
                # Google via OpenAI-compat endpoint for execution
                _AGENT_PROVIDER = OpenAICompatProvider(
                    api_key=os.environ.get(cfg["api_key_env"], ""),
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    provider_name="google",
                )
                _AGENT_MODEL = cfg.get("default_model", "gemini-3.1-flash-lite-preview")
            else:
                _AGENT_PROVIDER = OpenAICompatProvider(
                    api_key=os.environ.get(cfg["api_key_env"], ""),
                    base_url=cfg["base_url"],
                    provider_name=cfg["provider"],
                )
                _AGENT_MODEL = cfg.get("default_model", "")
            log.info("Agent provider: %s (%s)", cfg["provider"], _AGENT_MODEL)
            return _AGENT_PROVIDER, _AGENT_MODEL
        except Exception:
            continue

    log.error("No agent provider available — set at least one API key")
    return None, ""


# Fallback provider (lazy, only created if primary fails at runtime)
_FALLBACK_PROVIDER = None
_FALLBACK_MODEL = ""


def _get_fallback_provider():
    """Runtime fallback when primary provider fails mid-execution."""
    global _FALLBACK_PROVIDER, _FALLBACK_MODEL
    if _FALLBACK_PROVIDER is not None:
        return _FALLBACK_PROVIDER, _FALLBACK_MODEL

    from sage.providers.connector import get_available_providers
    from sage.providers.openai_compat import OpenAICompatProvider

    # Pick second available provider (first is likely the primary that failed)
    available = get_available_providers()
    if len(available) < 2:
        return None, ""
    cfg = available[1]
    try:
        if cfg.get("sdk") == "google-genai":
            _FALLBACK_PROVIDER = OpenAICompatProvider(
                api_key=os.environ.get(cfg["api_key_env"], ""),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                provider_name="google",
            )
        else:
            _FALLBACK_PROVIDER = OpenAICompatProvider(
                api_key=os.environ.get(cfg["api_key_env"], ""),
                base_url=cfg["base_url"],
                provider_name=cfg["provider"],
            )
        _FALLBACK_MODEL = cfg.get("default_model", "")
        log.info("Fallback provider: %s (%s)", cfg["provider"], _FALLBACK_MODEL)
        return _FALLBACK_PROVIDER, _FALLBACK_MODEL
    except Exception:
        return None, ""


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
    """Combine Rust structural/density with graduated execution score.

    Per-difficulty density bounds (AgentConductor 2602.17100):
      simple  → N_max=4 (S1)
      moderate→ N_max=7 (S2)
      complex → N_max=10 (S3)
    Over-budget penalty: tanh((N_max - |V|) / N_max) — from AgentConductor Eq.13.
    """
    import math
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

        # AgentConductor Eq.13: over-budget gets tanh penalty (can go negative)
        if density.over_budget:
            n_nodes = graph.node_count()
            penalty = math.tanh((density.n_max - n_nodes) / max(density.n_max, 1))
            score = score * max(0.0, 1.0 + penalty)  # penalty is negative when over

        return float(score)
    except Exception:
        return exec_norm


# ── Main evaluation — full topology execution ───────────────────

async def evaluate_topology(
    task: str, topology_dict: dict, semaphore: asyncio.Semaphore,
) -> float:
    """Execute the FULL topology as a multi-agent system, then test the result.

    This is what makes GRPO learn topology quality:
    - Topology A (coder alone) → might solve 30%
    - Topology B (planner→coder→reviewer) → might solve 50%
    - GRPO reinforces B over A
    """
    async with semaphore:
        t0 = time.time()

        difficulty = topology_dict.get("difficulty", "moderate")
        system = {"simple": 1, "moderate": 2, "complex": 3}.get(
            str(difficulty).lower(), 2
        )

        # Build Rust graph (for structural/density scoring)
        graph = _build_topology_graph(topology_dict)

        provider, model = _get_agent_provider()
        if provider is None:
            return 0.0

        # Execute topology via TopologyRunner (all nodes, in order, with context)
        from sage.llm.base import LLMConfig
        try:
            from sage.topology.runner import TopologyRunner
            from sage_core import TopologyExecutor

            if graph is None:
                _stats.record(success=False, latency=time.time() - t0, status="NO_GRAPH")
                return 0.0

            executor = TopologyExecutor(graph)
            config = LLMConfig(provider="agent", model=model)
            runner = TopologyRunner(
                graph=graph,
                executor=executor,
                llm_provider=provider,
                llm_config=config,
            )
            final_output = await asyncio.wait_for(
                runner.run(task[:2000]),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            _stats.record(success=False, timeout=True, latency=time.time() - t0, status="TOPO_TIMEOUT")
            return _compute_rust_reward(graph, 0.0, system)
        except Exception as exc:
            # Fallback: retry with DeepSeek chat if Gemini failed
            fb_provider, fb_model = _get_fallback_provider()
            if fb_provider is not None:
                try:
                    from sage.topology.runner import TopologyRunner as _TR
                    from sage_core import TopologyExecutor as _TE
                    executor2 = _TE(graph)
                    fb_config = LLMConfig(provider="fallback", model=fb_model)
                    runner2 = _TR(graph=graph, executor=executor2, llm_provider=fb_provider, llm_config=fb_config)
                    final_output = await asyncio.wait_for(runner2.run(task[:2000]), timeout=120.0)
                    log.info("Fallback provider succeeded for topology")
                except Exception:
                    log.warning("TopologyRunner failed (primary + fallback): %s", str(exc)[:100])
                    _stats.record(success=False, latency=time.time() - t0, status="RUNNER_ERROR")
                    return _compute_rust_reward(graph, 0.0, system)
            else:
                log.warning("TopologyRunner failed: %s", str(exc)[:100])
                _stats.record(success=False, latency=time.time() - t0, status="RUNNER_ERROR")
                return _compute_rust_reward(graph, 0.0, system)

        # Extract code from the final node's output
        code = extract_python_code(final_output)

        # If no code block found, ask the agent to return just the code
        if code is None and final_output and len(final_output.strip()) > 20:
            from sage.llm.base import Message, Role, LLMConfig
            try:
                followup = await asyncio.wait_for(
                    provider.generate(
                        messages=[
                            Message(role=Role.SYSTEM, content="Extract the Python code from the conversation and return ONLY the code in a ```python block. No explanation."),
                            Message(role=Role.USER, content=final_output[:3000]),
                        ],
                        config=LLMConfig(provider="agent", model=model),
                    ),
                    timeout=30.0,
                )
                code = extract_python_code(followup.content or "")
            except Exception:
                pass

        if code is None:
            _stats.record(success=False, latency=time.time() - t0, status="NO_CODE")
            return _compute_rust_reward(graph, 0.0, system)

        # Test the code in sandbox + BigCodeBench tests
        task_id = topology_dict.get("_task_id", "")
        exec_score, status = await compute_execution_score(code, task_id, timeout=30)

        n_nodes = len(topology_dict.get("nodes", []))
        _stats.record(
            success=(exec_score >= 1.0), tokens=len(final_output) // 4,
            latency=time.time() - t0, status=status,
        )

        if _stats.total % 5 == 0:
            log.info("Topo: %s | nodes=%d | %.1fs | task=%s",
                     status, n_nodes, time.time() - t0, task_id[:30])

        return _compute_rust_reward(graph, exec_score, system)


# ── Batch entry point (called by GRPOTrainer) ───────────────────

def execution_reward_batch(completions: list, **kwargs) -> list[float]:
    """Sync wrapper for batch execution reward. Called by GRPOTrainer.

    TRL passes all Dataset columns as kwargs. We use:
      - prompts: list[str] — the task descriptions
      - task_id: list[str] — e.g. ["BigCodeBench/13", ...] for test-case matching
    """
    prompts = kwargs.get("prompts", [])
    task_ids = kwargs.get("task_id", [])

    tasks_and_topos = []
    indices = []
    for i, completion in enumerate(completions):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        prompt = prompts[i] if i < len(prompts) else ""
        if isinstance(prompt, list):
            prompt = prompt[-1]["content"] if prompt else ""
        tid = task_ids[i] if i < len(task_ids) else ""

        topo = None
        try:
            topo = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            try:
                topo = yaml.safe_load(text)
            except Exception:
                pass

        if isinstance(topo, dict) and "nodes" in topo and len(topo.get("nodes", [])) > 0:
            topo["_task_id"] = tid  # inject for evaluate_topology
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
