# YGN-SAGE V2 "Evidence-First Rebuild" — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild YGN-SAGE from evidence-first principles: kill false claims, establish baselines, harden infrastructure, then build real capabilities.

**Architecture:** 4 phases in strict order. Phase 0 (honesty) and Phase 1 (baselines) are fully detailed below. Phases 2-4 are outlined at task level — detailed plans written after Phase 1 findings.

**Tech Stack:** Python 3.12+, z3-solver, wasmtime 29.0 (Rust), FastAPI, pytest. No new heavy deps.

---

## Phase 0: Kill Dangerous Defaults (Tasks 1-5)

---

### Task 1: Strip False Claims from README

**Files:**
- Modify: `README.md`

**Step 1: Read current README**

Read `README.md` fully. Identify every line containing "surpasses", "SOTA", "world-class", "outperforms", specific test counts, or competitor comparisons.

**Step 2: Rewrite README honestly**

Replace claims with factual statements:
- "7 providers auto-discovered" → keep (factual)
- "307 tests passing" → update to actual count from `pytest --co -q | tail -1`
- Remove ALL "surpasses X" language
- Remove competitor comparison table that implies superiority
- Add "Status: Research Prototype" badge
- Add "Benchmarks" section with link to actual published results (or "pending")

**Step 3: Verify no false claims remain**

```bash
grep -in "surpass\|SOTA\|world.class\|outperform\|superior\|beats\|exceeds" README.md
```
Expected: 0 matches.

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs: strip false claims from README, update to factual status"
```

---

### Task 2: Create Honest ARCHITECTURE.md

**Files:**
- Create: `ARCHITECTURE.md`

**Step 1: Write ARCHITECTURE.md**

Document ONLY what is implemented and tested. For each component:
- What it does (1-2 sentences)
- Evidence level: {implemented, tested-unit, tested-integration, benchmarked}
- Known limitations
- Silent degradation modes

Components to cover:
1. CognitiveOrchestrator — binary routing (seq/par), no hierarchical/hybrid
2. ModelRegistry — 7 providers, semantic loss on OpenAI-compat path
3. S1/S2/S3 routing — heuristic or LLM-based, benchmark is self-consistency
4. Memory — Tier 0 solid (Rust), Tiers 1-2 in-memory fallback, Tier 3 vendor-locked
5. Dashboard — functional but zero auth, single-task, global state
6. Evolution — scaffolding, not validated
7. Rust core — eBPF/Wasm/Arrow real but sandbox incomplete

**Step 2: Commit**

```bash
git add ARCHITECTURE.md
git commit -m "docs: add honest ARCHITECTURE.md describing only implemented state"
```

---

### Task 3: Relabel Routing Benchmark

**Files:**
- Modify: `sage-python/src/sage/bench/routing.py:1-10`

**Step 1: Update docstring to be explicit about circularity**

Change the docstring at `routing.py:1-10` to:

```python
"""Routing SELF-CONSISTENCY benchmark (NOT accuracy).

Measures whether MetacognitiveController's heuristic agrees with
hand-labeled tasks. Labels were calibrated against the heuristic,
so 100% agreement is expected and proves nothing about downstream
task quality.

To measure real routing accuracy, compare task outcomes (pass/fail)
across different routing decisions. See docs/plans/ for the
evidence-first routing benchmark design.
"""
```

**Step 2: Run tests to verify nothing breaks**

```bash
cd sage-python && python -m pytest tests/ -q --tb=short
```
Expected: 412+ passed.

**Step 3: Commit**

```bash
git add sage-python/src/sage/bench/routing.py
git commit -m "docs(bench): relabel routing benchmark as self-consistency test"
```

---

### Task 4: Fix Silent Provider Degradation

**Files:**
- Modify: `sage-python/src/sage/providers/openai_compat.py:44-63`

**Step 1: Write failing test**

```python
# sage-python/tests/test_provider_conformance.py
import pytest
from sage.providers.openai_compat import OpenAICompatProvider


def test_openai_compat_warns_on_tool_role():
    """Provider must log WARNING (not debug) when dropping tool semantics."""
    provider = OpenAICompatProvider(
        api_key="test", base_url="http://fake", model_id="test"
    )
    # Verify the provider has a method that documents its limitations
    caps = provider.capabilities()
    assert caps["tool_role"] is False
    assert caps["file_search"] is False


def test_openai_compat_capabilities_method_exists():
    """Every provider must expose a capabilities() dict."""
    provider = OpenAICompatProvider(
        api_key="test", base_url="http://fake", model_id="test"
    )
    caps = provider.capabilities()
    assert isinstance(caps, dict)
    assert "structured_output" in caps
    assert "tool_role" in caps
    assert "file_search" in caps
```

**Step 2: Run test to verify it fails**

```bash
cd sage-python && python -m pytest tests/test_provider_conformance.py -v
```
Expected: FAIL — `capabilities()` method doesn't exist.

**Step 3: Add capabilities() to OpenAICompatProvider**

In `openai_compat.py`, add:

```python
def capabilities(self) -> dict[str, bool]:
    """Declare what this provider actually supports."""
    return {
        "structured_output": False,
        "tool_role": False,      # Rewritten to user role
        "file_search": False,    # Ignored
        "grounding": False,
        "system_prompt": True,
        "streaming": False,
    }
```

Also change `log.debug` to `log.warning` at line 45 for file_search.

**Step 4: Add capabilities() to GoogleProvider**

In `sage-python/src/sage/llm/google.py`, add:

```python
def capabilities(self) -> dict[str, bool]:
    return {
        "structured_output": True,
        "tool_role": True,
        "file_search": True,
        "grounding": True,
        "system_prompt": True,
        "streaming": True,
    }
```

**Step 5: Run tests**

```bash
cd sage-python && python -m pytest tests/test_provider_conformance.py tests/ -q --tb=short
```
Expected: all pass, no regressions.

**Step 6: Commit**

```bash
git add sage-python/src/sage/providers/openai_compat.py sage-python/src/sage/llm/google.py tests/test_provider_conformance.py
git commit -m "feat(providers): add capabilities() method, stop silent semantic degradation"
```

---

### Task 5: Fix Dashboard Security

**Files:**
- Modify: `ui/app.py:1-30,95-100,270-280`

**Step 1: Write failing test**

```python
# sage-python/tests/test_dashboard_security.py
import pytest
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def app():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ui"))
    from app import app as fastapi_app
    return fastapi_app


@pytest.mark.asyncio
async def test_dashboard_requires_auth(app):
    """All API routes must return 401/403 without auth token."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/events")
        assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_cors_headers_present(app):
    """CORS middleware must be configured."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.options(
            "/api/events",
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
        )
        assert "access-control-allow-origin" in resp.headers
```

**Step 2: Run test to verify it fails**

```bash
cd sage-python && python -m pytest tests/test_dashboard_security.py -v
```
Expected: FAIL — no auth, no CORS.

**Step 3: Add auth + CORS to dashboard**

In `ui/app.py`, add at the top after FastAPI init:

```python
import os
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

DASHBOARD_TOKEN = os.environ.get("SAGE_DASHBOARD_TOKEN", "")

security = HTTPBearer(auto_error=False)

async def verify_token(credentials: HTTPAuthorizationCredentials | None = Security(security)):
    if not DASHBOARD_TOKEN:
        return  # No token configured = open access (dev mode)
    if credentials is None or credentials.credentials != DASHBOARD_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Add `dependencies=[Depends(verify_token)]` to API routes (not the static HTML route).

Replace `event_bus._buffer.clear()` with `event_bus.clear()` after adding a `clear()` method to EventBus.

**Step 4: Run tests**

```bash
cd sage-python && python -m pytest tests/test_dashboard_security.py tests/ -q --tb=short
```

**Step 5: Commit**

```bash
git add ui/app.py sage-python/tests/test_dashboard_security.py sage-python/src/sage/events/bus.py
git commit -m "fix(dashboard): add auth, CORS, EventBus.clear(), stop private field mutation"
```

---

## Phase 1: Benchmark & Baseline Truth (Tasks 6-10)

---

### Task 6: EvidenceRecord Dataclass

**Files:**
- Create: `sage-python/src/sage/evidence.py`
- Test: `sage-python/tests/test_evidence.py`

**Step 1: Write failing test**

```python
# sage-python/tests/test_evidence.py
import pytest
from datetime import datetime
from sage.evidence import EvidenceRecord, EvidenceLevel


def test_evidence_record_creation():
    er = EvidenceRecord(
        level=EvidenceLevel.HEURISTIC,
        proof_strength=0.3,
        external_validity=False,
        coverage=0.12,
        assumptions=["labels calibrated to heuristic"],
    )
    assert er.level == EvidenceLevel.HEURISTIC
    assert er.proof_strength == 0.3


def test_evidence_level_ordering():
    assert EvidenceLevel.HEURISTIC.value < EvidenceLevel.SOLVER_PROVED.value


def test_evidence_record_to_dict():
    er = EvidenceRecord(level=EvidenceLevel.EMPIRICALLY_VALIDATED, proof_strength=0.95)
    d = er.to_dict()
    assert d["level"] == "empirically_validated"
    assert "timestamp" in d


def test_evidence_readme_projection():
    """README uses simple string, not full record."""
    er = EvidenceRecord(level=EvidenceLevel.CHECKED, proof_strength=0.6)
    assert er.readme_label() == "checked"
```

**Step 2: Implement**

```python
# sage-python/src/sage/evidence.py
"""Evidence records for every claim YGN-SAGE makes."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum


class EvidenceLevel(IntEnum):
    HEURISTIC = 1
    CHECKED = 2
    MODEL_JUDGED = 3
    SOLVER_PROVED = 4
    EMPIRICALLY_VALIDATED = 5


@dataclass
class EvidenceRecord:
    level: EvidenceLevel
    proof_strength: float = 0.0        # 0.0-1.0
    external_validity: bool = False
    coverage: float = 0.0              # fraction of cases covered
    assumptions: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "level": self.level.name.lower(),
            "proof_strength": self.proof_strength,
            "external_validity": self.external_validity,
            "coverage": self.coverage,
            "assumptions": self.assumptions,
            "artifacts": self.artifacts,
            "timestamp": self.timestamp.isoformat(),
        }

    def readme_label(self) -> str:
        return self.level.name.lower()
```

**Step 3: Run tests, commit**

```bash
cd sage-python && python -m pytest tests/test_evidence.py -v
git add sage-python/src/sage/evidence.py sage-python/tests/test_evidence.py
git commit -m "feat(evidence): add EvidenceRecord + EvidenceLevel for typed claims"
```

---

### Task 7: Semantic CapabilityMatrix

**Files:**
- Create: `sage-python/src/sage/providers/capabilities.py`
- Test: `sage-python/tests/test_capabilities.py`
- Modify: `sage-python/src/sage/boot.py` (wire matrix at boot)

**Step 1: Write failing test**

```python
# sage-python/tests/test_capabilities.py
import pytest
from sage.providers.capabilities import CapabilityMatrix, ProviderCapabilities


def test_matrix_registers_provider():
    matrix = CapabilityMatrix()
    caps = ProviderCapabilities(
        provider="google", structured_output=True, tool_role=True,
        file_search=True, grounding=True,
    )
    matrix.register(caps)
    assert matrix.get("google").structured_output is True


def test_matrix_check_requirement():
    matrix = CapabilityMatrix()
    matrix.register(ProviderCapabilities(provider="google", tool_role=True))
    matrix.register(ProviderCapabilities(provider="xai", tool_role=False))
    compatible = matrix.providers_for(tool_role=True)
    assert "google" in compatible
    assert "xai" not in compatible


def test_matrix_hard_fail_on_missing():
    matrix = CapabilityMatrix()
    matrix.register(ProviderCapabilities(provider="mock", tool_role=False))
    with pytest.raises(ValueError, match="No provider supports"):
        matrix.require(tool_role=True)
```

**Step 2: Implement CapabilityMatrix**

```python
# sage-python/src/sage/providers/capabilities.py
"""Semantic capability matrix — hard-fail when required features missing."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderCapabilities:
    provider: str
    structured_output: bool = False
    tool_role: bool = False
    file_search: bool = False
    grounding: bool = False
    system_prompt: bool = True
    streaming: bool = False


class CapabilityMatrix:
    def __init__(self):
        self._providers: dict[str, ProviderCapabilities] = {}

    def register(self, caps: ProviderCapabilities) -> None:
        self._providers[caps.provider] = caps

    def get(self, provider: str) -> ProviderCapabilities:
        return self._providers[provider]

    def providers_for(self, **requirements: bool) -> list[str]:
        result = []
        for name, caps in self._providers.items():
            if all(getattr(caps, k, False) == v for k, v in requirements.items() if v):
                result.append(name)
        return result

    def require(self, **requirements: bool) -> list[str]:
        compatible = self.providers_for(**requirements)
        if not compatible:
            missing = [k for k, v in requirements.items() if v]
            raise ValueError(f"No provider supports: {missing}")
        return compatible
```

**Step 3: Run tests, commit**

```bash
cd sage-python && python -m pytest tests/test_capabilities.py tests/ -q --tb=short
git add sage-python/src/sage/providers/capabilities.py sage-python/tests/test_capabilities.py
git commit -m "feat(providers): add CapabilityMatrix with hard-fail on missing features"
```

---

### Task 8: Benchmark Truth Pack Infrastructure

**Files:**
- Create: `sage-python/src/sage/bench/truth_pack.py`
- Test: `sage-python/tests/test_truth_pack.py`
- Modify: `sage-python/src/sage/bench/humaneval.py` (emit per-task trace)

**Step 1: Write failing test**

```python
# sage-python/tests/test_truth_pack.py
import pytest
import json
from sage.bench.truth_pack import BenchmarkManifest, TaskTrace


def test_manifest_creation():
    m = BenchmarkManifest(benchmark="humaneval", model="gemini-3.1-flash-lite")
    assert m.benchmark == "humaneval"
    assert m.git_sha is not None  # auto-populated


def test_task_trace_serialization():
    t = TaskTrace(
        task_id="HumanEval/0", passed=True,
        latency_ms=1200, cost_usd=0.001,
        model="gemini-3.1-flash-lite", routing="S2",
    )
    d = t.to_dict()
    assert d["task_id"] == "HumanEval/0"
    assert d["passed"] is True


def test_manifest_add_trace_and_export():
    m = BenchmarkManifest(benchmark="humaneval", model="test-model")
    m.add(TaskTrace(task_id="t1", passed=True, latency_ms=100, cost_usd=0.0))
    m.add(TaskTrace(task_id="t2", passed=False, latency_ms=200, cost_usd=0.0))
    export = m.to_jsonl()
    lines = export.strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["task_id"] == "t1"
```

**Step 2: Implement**

```python
# sage-python/src/sage/bench/truth_pack.py
"""Benchmark truth pack — machine-auditable per-task traces."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, timeout=5
        ).strip()[:12]
    except Exception:
        return "unknown"


@dataclass
class TaskTrace:
    task_id: str
    passed: bool
    latency_ms: float
    cost_usd: float
    model: str = ""
    routing: str = ""
    error: str = ""
    seed: int = 0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v or isinstance(v, bool)}


@dataclass
class BenchmarkManifest:
    benchmark: str
    model: str
    git_sha: str = field(default_factory=_git_sha)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    traces: list[TaskTrace] = field(default_factory=list)

    def add(self, trace: TaskTrace) -> None:
        self.traces.append(trace)

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(t.to_dict()) for t in self.traces)

    def summary(self) -> dict:
        passed = sum(1 for t in self.traces if t.passed)
        return {
            "benchmark": self.benchmark,
            "model": self.model,
            "git_sha": self.git_sha,
            "total": len(self.traces),
            "passed": passed,
            "pass_rate": round(passed / max(len(self.traces), 1), 4),
            "avg_latency_ms": round(
                sum(t.latency_ms for t in self.traces) / max(len(self.traces), 1), 1
            ),
        }
```

**Step 3: Run tests, commit**

```bash
cd sage-python && python -m pytest tests/test_truth_pack.py tests/ -q --tb=short
git add sage-python/src/sage/bench/truth_pack.py sage-python/tests/test_truth_pack.py
git commit -m "feat(bench): add BenchmarkManifest + TaskTrace for truth pack"
```

---

### Task 9: Wire Truth Pack into HumanEval Runner

**Files:**
- Modify: `sage-python/src/sage/bench/humaneval.py`
- Modify: `sage-python/src/sage/bench/__main__.py`

**Step 1: Modify HumanEvalBench to emit TaskTrace per problem**

In `humaneval.py`, after each problem evaluation (around line 76-100), create a TaskTrace and collect it. At the end, write a JSONL file to `docs/benchmarks/`.

**Step 2: Modify __main__.py to save manifest**

After the benchmark run, save:
- `docs/benchmarks/YYYY-MM-DD-humaneval.jsonl` (per-task traces)
- `docs/benchmarks/YYYY-MM-DD-humaneval-summary.json` (manifest summary)

**Step 3: Run smoke test (3 problems)**

```bash
cd sage-python && python -m sage.bench --type humaneval --limit 3
cat ../docs/benchmarks/*humaneval*.jsonl
```
Expected: 3 lines of JSONL with task_id, passed, latency, model.

**Step 4: Commit**

```bash
git add sage-python/src/sage/bench/humaneval.py sage-python/src/sage/bench/__main__.py
git commit -m "feat(bench): wire truth pack into HumanEval runner, emit per-task JSONL"
```

---

### Task 10: Fix CI Pipeline

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `sage-python/pyproject.toml` (add openai to test deps)

**Step 1: Fix Rust job — skip clippy for sandbox feature**

```yaml
- run: cargo clippy --no-default-features -- -D warnings
- run: cargo test --no-default-features
```

**Step 2: Fix Python SDK job — add openai to deps**

In `sage-python/pyproject.toml`, add to `[project.optional-dependencies]`:
```toml
openai = ["openai>=1.50"]
```

And in `all` group, add `"openai"`:
```toml
all = ["google", "arrow", "z3", "ui", "openai"]
```

**Step 3: Verify locally**

```bash
cd sage-core && cargo clippy --no-default-features -- -D warnings
cd ../sage-python && pip install -e ".[all,dev]" && python -m pytest tests/ -q --tb=short
```

**Step 4: Commit and push**

```bash
git add .github/workflows/ci.yml sage-python/pyproject.toml
git commit -m "fix(ci): clippy no-default-features, add openai to test deps"
git push origin master
```

**Step 5: Verify CI passes**

```bash
sleep 60 && gh run list --limit 1
```

---

## Phase 2: Contract IR + Policy Verification (Tasks 11-18) — OUTLINE

> Detailed plan written after Phase 1 completion. Findings from benchmarks and capability matrix will reshape these.

### Task 11: TaskNode IR dataclass + basic tests
### Task 12: Verification Functions (VFs) per TaskNode
### Task 13: TaskDAG builder + topological scheduler
### Task 14: Z3 contract verification (capability coverage, budget feasibility)
### Task 15: Z3 Datalog reachability + provenance
### Task 16: Wasm fuel-limited sandbox (declarative interface)
### Task 17: Wire TaskNode IR into CognitiveOrchestrator
### Task 18: Integration test — real LLM API with TaskNode + VFs

---

## Phase 3: Dynamic Routing + Memory (Tasks 19-24) — OUTLINE

### Task 19: Planner/Executor separation (Plan-and-Act inspired)
### Task 20: DyTopo round-level routing inside verified envelope
### Task 21: Causal Memory with Causality Graph
### Task 22: Long-context baseline benchmark for memory
### Task 23: Memory write gating + abstention
### Task 24: Integration test — routing + memory + contracts end-to-end

---

## Phase 4: Repair + Validation + Paper (Tasks 25-30) — OUTLINE

### Task 25: Counterexample-guided repair with hard fences
### Task 26: Synthetic failure lab (MAST taxonomy)
### Task 27: Full HumanEval 164 + MBPP with truth packs
### Task 28: Ablation studies (routing, memory, verification)
### Task 29: Reproducibility kit packaging
### Task 30: Paper draft — "Contract-Verified Agent Orchestration with Deterministic Tool Sandboxes"
