# GRPO v4 — Real Execution Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 5 critical issues in GRPO training so the model learns from real code execution, not response length.

**Architecture:** Replace the proxy execution reward (`len(response) > 20`) with real subprocess execution via SAGE's existing `execute_python_in_sandbox()`. Build edges into TopologyGraph from YAML. Infer system (S1/S2/S3) from the model's `difficulty` output. Align format_reward range and learning rate with AgentConductor.

**Tech Stack:** Python 3.13, sage-core (Rust/PyO3), TRL 0.29 GRPOTrainer, DeepSeek Reasoner API

---

## Files Map

| Action | File | Responsibility |
|--------|------|----------------|
| **Rewrite** | `sage-python/src/sage/grpo/execution_reward.py` | Fixes #1 (real exec), #2 (edges), #3 (system) |
| **Modify** | `sage-python/scripts/train_topology_grpo.py:165-189` | Fix #4 (format_reward range) |
| **Modify** | `sage-python/scripts/train_topology_grpo.py:486` | Fix #5 (learning rate) |
| **Create** | `sage-python/tests/test_grpo_execution_reward.py` | Tests for all new functions |

---

## Rust API Reference (verified from topology_graph.rs #[pymethods])

```python
from sage_core import TopologyGraph, TopologyNode, TopologyEdge, TopologyReward, TopologyDensity, PyHybridVerifier

# Constructors
graph = TopologyGraph("sequential")                          # template_type: str
node = TopologyNode(role="coder", model_id="", system=2)     # role, model_id required; system default=1
edge = TopologyEdge("message")                               # edge_type: str; gate="open", weight=1.0

# Methods
idx = graph.add_node(node)                                   # returns usize (node index)
graph.add_edge(from_idx, to_idx, TopologyEdge("message"))    # from: usize, to: usize, edge: TopologyEdge
n = graph.node_count()                                       # returns usize
e = graph.edge_count()                                       # returns usize

# Density
density = TopologyDensity()
score = density.compute(graph, system)                       # graph: TopologyGraph, system: u8 (1/2/3)
# score.s_complex, score.s_node, score.s_edge, score.s_depth, score.n_max, score.over_budget

# Verification
verifier = PyHybridVerifier()
result = verifier.verify(graph)                              # returns VerificationResult
# result.valid: bool, result.errors: list[str], result.warnings: list[str]

# Reward
scorer = TopologyReward()
reward = scorer.compute(execution_passed=True, structural_score=1.0, density_score=0.8, temporal_score=None)
# reward.total, reward.execution, reward.structural, reward.density, reward.temporal
```

## Existing Sandbox API (from sage/tools/sandbox_executor.py)

```python
from sage.tools.sandbox_executor import execute_python_in_sandbox, SandboxResult

# IMPORTANT: execute_python_in_sandbox wraps code in a template that reads args from stdin.
# For GRPO, we need raw code execution WITHOUT the stdin wrapper.
# Solution: use asyncio.create_subprocess_exec directly (same pattern, skip wrapper).
```

The existing `execute_python_in_sandbox` injects `import json, sys; args = json.load(sys.stdin)` before user code. This breaks standalone code execution. We'll use the same async subprocess pattern but without the wrapper.

## SFT Data Facts

- `task_id`: "BigCodeBench/13", "CodeContests/42", etc.
- `difficulty`: "simple" (810), "moderate" (438), "complex" (57) — in both entry AND topology dict
- `edges`: list of `{from_idx, to_idx, flow_type}` — already present in SFT data
- `nodes`: list of `{role, prompt, model_tier}`
- HumanEval test cases: 164 problems in `src/sage/bench/humaneval_data.json` with `test` field
- The model outputs YAML including `difficulty`, `nodes`, `edges` fields

---

### Task 1: Test infrastructure for execution reward

**Files:**
- Create: `sage-python/tests/test_grpo_execution_reward.py`

- [ ] **Step 1: Write tests for extract_python_code**

```python
"""Tests for GRPO v4 execution reward — real sandbox execution."""
import pytest


class TestExtractPythonCode:
    def test_fenced_python_block(self):
        from sage.grpo.execution_reward import extract_python_code
        text = "Here is the solution:\n```python\ndef add(a, b):\n    return a + b\n```\nDone."
        assert extract_python_code(text) == "def add(a, b):\n    return a + b"

    def test_fenced_block_no_lang(self):
        from sage.grpo.execution_reward import extract_python_code
        text = "```\nimport math\nprint(math.pi)\n```"
        assert "import math" in extract_python_code(text)

    def test_raw_code(self):
        from sage.grpo.execution_reward import extract_python_code
        assert extract_python_code("def foo():\n    pass") == "def foo():\n    pass"

    def test_think_tags_stripped(self):
        from sage.grpo.execution_reward import extract_python_code
        text = "<think>reasoning here</think>\n```python\nprint(42)\n```"
        code = extract_python_code(text)
        assert "<think>" not in code
        assert "print(42)" in code

    def test_no_code_returns_none(self):
        from sage.grpo.execution_reward import extract_python_code
        assert extract_python_code("This is just text with no code.") is None

    def test_empty_string(self):
        from sage.grpo.execution_reward import extract_python_code
        assert extract_python_code("") is None
```

- [ ] **Step 2: Run tests — verify they fail (module not found)**

Run: `cd sage-python && python -m pytest tests/test_grpo_execution_reward.py::TestExtractPythonCode -v`
Expected: FAIL with `ImportError: cannot import name 'extract_python_code'`

---

### Task 2: Implement extract_python_code

**Files:**
- Modify: `sage-python/src/sage/grpo/execution_reward.py`

- [ ] **Step 3: Add extract_python_code to execution_reward.py**

Add after the imports section:

```python
import re as _re

def extract_python_code(response: str) -> str | None:
    """Extract Python code block from LLM response (handles <think> tags)."""
    # Strip DeepSeek Reasoner <think>...</think> tags
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
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `cd sage-python && python -m pytest tests/test_grpo_execution_reward.py::TestExtractPythonCode -v`
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add sage-python/tests/test_grpo_execution_reward.py sage-python/src/sage/grpo/execution_reward.py
git commit -m "feat: extract_python_code for GRPO v4 (strips <think> tags)"
```

---

### Task 3: Implement async sandbox runner and compute_execution_score

**Files:**
- Modify: `sage-python/src/sage/grpo/execution_reward.py`
- Modify: `sage-python/tests/test_grpo_execution_reward.py`

- [ ] **Step 6: Write tests for run_code_sandbox and compute_execution_score**

Add to test file:

```python
import asyncio


class TestRunCodeSandbox:
    def test_simple_print(self):
        from sage.grpo.execution_reward import run_code_sandbox
        result = asyncio.run(run_code_sandbox("print(42)", timeout=10))
        assert result.exit_code == 0
        assert "42" in result.stdout

    def test_syntax_error(self):
        from sage.grpo.execution_reward import run_code_sandbox
        result = asyncio.run(run_code_sandbox("def (", timeout=10))
        assert result.exit_code != 0

    def test_timeout(self):
        from sage.grpo.execution_reward import run_code_sandbox
        result = asyncio.run(run_code_sandbox("import time; time.sleep(60)", timeout=2))
        assert result.timed_out

    def test_runtime_error(self):
        from sage.grpo.execution_reward import run_code_sandbox
        result = asyncio.run(run_code_sandbox("raise ValueError('boom')", timeout=10))
        assert result.exit_code != 0
        assert "ValueError" in result.stderr


class TestComputeExecutionScore:
    def test_code_runs_ok(self):
        from sage.grpo.execution_reward import compute_execution_score
        score, status = asyncio.run(compute_execution_score("print(42)", ""))
        assert score == 0.3
        assert status == "RUNS_OK"

    def test_code_crash(self):
        from sage.grpo.execution_reward import compute_execution_score
        score, status = asyncio.run(compute_execution_score("raise Exception()", ""))
        assert score == 0.0
        assert status == "CRASH"

    def test_code_timeout(self):
        from sage.grpo.execution_reward import compute_execution_score
        score, status = asyncio.run(compute_execution_score(
            "import time; time.sleep(60)", "", timeout=2
        ))
        assert score == 0.5
        assert status == "TIMEOUT"
```

- [ ] **Step 7: Run tests — verify they fail**

Run: `cd sage-python && python -m pytest tests/test_grpo_execution_reward.py -k "Sandbox or ExecutionScore" -v`
Expected: FAIL with `ImportError`

- [ ] **Step 8: Implement run_code_sandbox and compute_execution_score**

Add to execution_reward.py:

```python
import tempfile
import sys as _sys
from pathlib import Path

# Re-use SandboxResult from existing infra
from sage.tools.sandbox_executor import SandboxResult


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


# HumanEval test cases (loaded once)
_TEST_CASES: dict[str, str] | None = None

def _load_test_cases() -> dict[str, str]:
    global _TEST_CASES
    if _TEST_CASES is not None:
        return _TEST_CASES
    _TEST_CASES = {}
    for path in [
        Path(__file__).parent.parent / "bench" / "humaneval_data.json",
        Path("sage-python/src/sage/bench/humaneval_data.json"),
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


async def compute_execution_score(
    code: str, task_id: str, timeout: int = 30,
) -> tuple[float, str]:
    """Execute code and return (score, status). Graduated like AgentConductor."""
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
```

- [ ] **Step 9: Run tests — verify they pass**

Run: `cd sage-python && python -m pytest tests/test_grpo_execution_reward.py -v`
Expected: ALL PASSED

- [ ] **Step 10: Commit**

```bash
git add sage-python/src/sage/grpo/execution_reward.py sage-python/tests/test_grpo_execution_reward.py
git commit -m "feat: real sandbox execution + graduated rewards for GRPO v4"
```

---

### Task 4: Fix _build_topology_graph (edges + system from difficulty)

**Files:**
- Modify: `sage-python/src/sage/grpo/execution_reward.py`
- Modify: `sage-python/tests/test_grpo_execution_reward.py`

- [ ] **Step 11: Write tests for _build_topology_graph**

```python
class TestBuildTopologyGraph:
    def test_nodes_and_edges_built(self):
        from sage.grpo.execution_reward import _build_topology_graph
        topo = {
            "difficulty": "moderate",
            "nodes": [
                {"role": "planner", "prompt": "Plan the task"},
                {"role": "coder", "prompt": "Write the code"},
            ],
            "edges": [{"from_idx": 0, "to_idx": 1, "flow_type": "message"}],
        }
        graph = _build_topology_graph(topo)
        if graph is None:
            pytest.skip("sage_core not available")
        assert graph.node_count() == 2
        assert graph.edge_count() == 1  # WAS ALWAYS 0 IN v3

    def test_system_from_difficulty(self):
        from sage.grpo.execution_reward import _build_topology_graph
        for diff, expected_sys in [("simple", 1), ("moderate", 2), ("complex", 3)]:
            topo = {"difficulty": diff, "nodes": [{"role": "coder"}]}
            graph = _build_topology_graph(topo)
            if graph is None:
                pytest.skip("sage_core not available")
            # system is internal — verify via density N_max
            from sage_core import TopologyDensity
            density = TopologyDensity()
            score = density.compute(graph, expected_sys)
            assert score.n_max == {1: 4, 2: 7, 3: 10}[expected_sys]

    def test_fallback_sequential_edges(self):
        from sage.grpo.execution_reward import _build_topology_graph
        topo = {"nodes": [{"role": "a"}, {"role": "b"}, {"role": "c"}]}  # no edges
        graph = _build_topology_graph(topo)
        if graph is None:
            pytest.skip("sage_core not available")
        assert graph.edge_count() == 2  # a->b, b->c fallback chain

    def test_invalid_edge_indices_skipped(self):
        from sage.grpo.execution_reward import _build_topology_graph
        topo = {
            "nodes": [{"role": "a"}],
            "edges": [{"from_idx": 0, "to_idx": 99, "flow_type": "message"}],  # out of bounds
        }
        graph = _build_topology_graph(topo)
        if graph is None:
            pytest.skip("sage_core not available")
        assert graph.edge_count() == 0  # invalid edge skipped
```

- [ ] **Step 12: Run tests — verify they fail**

Run: `cd sage-python && python -m pytest tests/test_grpo_execution_reward.py::TestBuildTopologyGraph -v`

- [ ] **Step 13: Rewrite _build_topology_graph**

Replace the existing `_build_topology_graph` in execution_reward.py:

```python
def _build_topology_graph(topology_dict: dict) -> Any:
    """Build TopologyGraph with nodes AND edges. System inferred from difficulty."""
    if not _RUST_AVAILABLE:
        return None
    nodes = topology_dict.get("nodes", [])
    if not nodes:
        return None

    # System from difficulty field (model generates this in YAML)
    difficulty = topology_dict.get("difficulty", "moderate")
    system = {"simple": 1, "moderate": 2, "complex": 3}.get(str(difficulty).lower(), 2)

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

    # Build edges from YAML
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
```

Also add `TopologyEdge` to the import block at top:

```python
from sage_core import TopologyReward, TopologyDensity, TopologyGraph, TopologyNode, TopologyEdge
```

- [ ] **Step 14: Run tests — verify they pass**

Run: `cd sage-python && python -m pytest tests/test_grpo_execution_reward.py -v`

- [ ] **Step 15: Commit**

```bash
git add sage-python/src/sage/grpo/execution_reward.py sage-python/tests/test_grpo_execution_reward.py
git commit -m "fix: build edges in TopologyGraph + system from difficulty field"
```

---

### Task 5: Rewrite evaluate_topology and _compute_rust_reward for real execution

**Files:**
- Modify: `sage-python/src/sage/grpo/execution_reward.py`

- [ ] **Step 16: Rewrite _compute_rust_reward to accept float score**

Replace existing `_compute_rust_reward`:

```python
def _compute_rust_reward(graph: Any, exec_score: float, system: int) -> float:
    """Combine Rust structural/density signals with graduated execution score.

    Uses TopologyReward for structural+density, then equally weights with exec.
    exec_score: 0.0 (crash) to 1.5 (passed). Normalized to [0,1].
    """
    exec_norm = min(exec_score / 1.5, 1.0)  # normalize to [0, 1]
    if not _RUST_AVAILABLE or graph is None:
        return exec_norm

    try:
        density = _density_scorer.compute(graph, system)
        verification = _verifier.verify(graph)
        structural_score = 1.0 if verification.valid else 0.5

        # TopologyReward.compute expects bool — use threshold
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
```

- [ ] **Step 17: Rewrite evaluate_topology for real execution**

Replace existing `evaluate_topology`:

```python
async def evaluate_topology(
    task: str, topology_dict: dict, semaphore: asyncio.Semaphore,
) -> float:
    """Execute topology via DeepSeek Reasoner, then run code in sandbox."""
    async with semaphore:
        t0 = time.time()

        # System from difficulty
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
            _stats.record(success=False, timeout=True, latency=time.time() - t0)
            return 0.0

        # Extract code from response
        code = extract_python_code(result_text)
        if code is None:
            _stats.record(success=False, latency=time.time() - t0)
            return _compute_rust_reward(graph, 0.0, system)

        # Real sandbox execution
        task_id = topology_dict.get("_task_id", "")
        exec_score, status = await compute_execution_score(code, task_id, timeout=30)

        tokens = len(result_text) // 4
        _stats.record(
            success=(exec_score >= 1.0), tokens=tokens, latency=time.time() - t0,
        )

        if _stats.total % 10 == 0:
            log.info("Exec status: %s (%.1f) task=%s", status, exec_score, task_id[:30])

        return _compute_rust_reward(graph, exec_score, system)
```

- [ ] **Step 18: Verify syntax**

Run: `cd /c/Code/YGN-SAGE && python -c "import ast; ast.parse(open('sage-python/src/sage/grpo/execution_reward.py').read()); print('OK')"`

- [ ] **Step 19: Run all tests**

Run: `cd sage-python && python -m pytest tests/test_grpo_execution_reward.py -v`

- [ ] **Step 20: Commit**

```bash
git add sage-python/src/sage/grpo/execution_reward.py
git commit -m "feat: evaluate_topology with real sandbox execution + graduated rewards"
```

---

### Task 6: Fix format_reward range (Point #4)

**Files:**
- Modify: `sage-python/scripts/train_topology_grpo.py:165-189`

- [ ] **Step 21: Change format_reward success from +0.5 to +1.0**

In `train_topology_grpo.py`, in the `run_grpo_v2` function, modify `format_reward`:

```python
    def format_reward(completions: list[str], **kwargs) -> list[float]:
        """Graduated YAML format reward. Aligned with AgentConductor scale."""
        import yaml
        rewards = []
        for completion in completions:
            text = completion[0]["content"] if isinstance(completion, list) else completion
            try:
                data = yaml.safe_load(text)
                if not isinstance(data, dict):
                    rewards.append(-1.5)
                    continue
                if "nodes" not in data:
                    rewards.append(-0.5)
                    continue
                nodes = data["nodes"]
                if not isinstance(nodes, list) or len(nodes) == 0:
                    rewards.append(-0.25)
                    continue
                # Valid topology — reward 1.0 (was 0.5, misaligned with exec range)
                rewards.append(1.0)
            except yaml.YAMLError:
                rewards.append(-2.0)
            except Exception:
                rewards.append(-2.0)
        return rewards
```

Change is: line 184 `rewards.append(0.5)` -> `rewards.append(1.0)`.

- [ ] **Step 22: Commit**

```bash
git add sage-python/scripts/train_topology_grpo.py
git commit -m "fix: format_reward +0.5 -> +1.0 (align range with execution reward)"
```

---

### Task 7: Fix learning rate (Point #5)

**Files:**
- Modify: `sage-python/scripts/train_topology_grpo.py:486`

- [ ] **Step 23: Change learning_rate from 5e-6 to 1e-6**

In `train_topology_grpo.py` line 486:

```python
        learning_rate=1e-6,  # AgentConductor uses 1e-6 (was 5e-6 = 5x too aggressive)
```

- [ ] **Step 24: Commit**

```bash
git add sage-python/scripts/train_topology_grpo.py
git commit -m "fix: LR 5e-6 -> 1e-6 (match AgentConductor, prevent divergence)"
```

---

### Task 8: End-to-end validation test

**Files:**
- None new

- [ ] **Step 25: Run full test suite to verify nothing broken**

```bash
cd /c/Code/YGN-SAGE/sage-python && python -m pytest tests/test_grpo_execution_reward.py -v
```

Expected: ALL PASSED

- [ ] **Step 26: Manual end-to-end test with DeepSeek**

```bash
cd /c/Code/YGN-SAGE && set -a && source .env && set +a && cd sage-python && python -c "
import asyncio, sys
sys.path.insert(0, 'src')
from sage.grpo.execution_reward import evaluate_topology

topo = {
    '_task_id': '',
    'difficulty': 'simple',
    'nodes': [
        {'role': 'coder', 'prompt': 'Write a Python function to reverse a string.'},
    ],
    'edges': [],
}
task = 'Write a function reverse_string(s: str) -> str that reverses a string.'
sem = asyncio.Semaphore(1)
score = asyncio.run(evaluate_topology(task, topo, sem))
print(f'Score: {score}')
print('SUCCESS' if score > 0 else 'FAILED — check DeepSeek API')
"
```

Expected: Score > 0 (DeepSeek generates code, sandbox executes it, Rust scores it)

- [ ] **Step 27: Final commit with all changes**

```bash
git push origin dev
```

---

### Task 9: Launch GRPO v4 overnight

**Files:**
- None

- [ ] **Step 28: Launch training**

```bash
set -a && source .env && set +a && cd sage-python
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0 nohup python -u \
    scripts/train_topology_grpo.py --mode grpo-v3 \
    --sft-checkpoint models/topology_sft/ \
    > data/grpo_v4_overnight.log 2>&1 &
echo "PID: $!"
```

- [ ] **Step 29: Verify first 2 minutes**

```bash
sleep 120 && tail -20 data/grpo_v4_overnight.log | grep -E "Pool|Parse|Exec|Error"
```

Expected: "DeepSeek Reasoner initialized", "Exec status: RUNS_OK" or similar

- [ ] **Step 30: Do NOT touch anything further. Review tomorrow morning.**

Pause/resume for PC move:
```bash
kill -STOP <PID>
kill -CONT <PID>
```

---

## Summary of all 5 fixes

| # | Problem | Fix | File | Task |
|---|---------|-----|------|------|
| 1 | `len(response) > 20` proxy | Real subprocess sandbox + graduated rewards | execution_reward.py | 3, 5 |
| 2 | Zero edges in TopologyGraph | Build edges from YAML `edges` field | execution_reward.py | 4 |
| 3 | `system=2` hardcoded | Read `difficulty` from model output YAML | execution_reward.py | 4 |
| 4 | `format_reward` max +0.5 | Change to +1.0 to align ranges | train_topology_grpo.py | 6 |
| 5 | LR 5e-6 (5x AgentConductor) | Change to 1e-6 | train_topology_grpo.py | 7 |
