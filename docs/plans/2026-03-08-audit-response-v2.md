# Audit Response V2 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all 18 confirmed issues from the 4-audit verification (AuditC, AuditG, AuditQ, AuditCG), prioritized by impact × effort.

**Architecture:** Three sprints: Sprint 1 (quick wins, 1 day), Sprint 2 (structural fixes, 3-5 days), Sprint 3 (evidence & validation, 1-2 weeks). Each task is independent and commitable.

**Tech Stack:** Python 3.12+, Rust (sage-core), Z3, google-genai, FastAPI, usearch-rs (new)

**Oracle Consensus:** Gemini 3.1 Pro + GPT-5.4 Codex + Context7 docs consulted for SOTA solutions.

---

## Sprint 1 — Quick Wins (1 day, 8 tasks)

### Task 1: Fix SchemaGuardrail misconfiguration

**Files:**
- Modify: `sage-python/src/sage/guardrails/builtin.py:46-95`
- Modify: `sage-python/src/sage/boot.py:287-290`
- Test: `sage-python/tests/test_guardrails.py`

**Problem:** SchemaGuardrail expects JSON `{"response": ...}` but AgentLoop returns raw text. The guardrail either always blocks valid output or is silently skipped.

**Step 1: Write the failing test**

```python
# In test_guardrails.py — add these tests
import pytest
from sage.guardrails.builtin import SchemaGuardrail, CostGuardrail, OutputGuardrail

@pytest.mark.asyncio
async def test_output_guardrail_passes_nonempty_text():
    g = OutputGuardrail(min_length=1)
    r = await g.check(output="Hello world")
    assert r.passed is True

@pytest.mark.asyncio
async def test_output_guardrail_blocks_empty():
    g = OutputGuardrail(min_length=1)
    r = await g.check(output="")
    assert r.passed is False

@pytest.mark.asyncio
async def test_output_guardrail_blocks_refusal():
    g = OutputGuardrail(min_length=10, refusal_patterns=["I cannot", "I'm sorry"])
    r = await g.check(output="I cannot help with that.")
    assert r.passed is False

@pytest.mark.asyncio
async def test_schema_guardrail_still_works_for_json():
    g = SchemaGuardrail(required_fields=["answer"])
    r = await g.check(output='{"answer": "42"}')
    assert r.passed is True
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_guardrails.py::test_output_guardrail_passes_nonempty_text -v`
Expected: FAIL (OutputGuardrail not defined)

**Step 3: Implement OutputGuardrail**

Replace SchemaGuardrail with OutputGuardrail in the default pipeline. SchemaGuardrail stays for structured-output mode.

```python
# Add to sage-python/src/sage/guardrails/builtin.py after SchemaGuardrail:

class OutputGuardrail(Guardrail):
    """Validate free-text agent output: non-empty, minimum length, no refusal patterns.

    Use this instead of SchemaGuardrail when the agent returns raw text.
    SchemaGuardrail should only be used when structured JSON output is expected.
    """

    name: str = "output"

    def __init__(
        self,
        min_length: int = 1,
        max_length: int = 100_000,
        refusal_patterns: list[str] | None = None,
    ) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.refusal_patterns = refusal_patterns or [
            "I cannot", "I'm sorry", "I am unable",
        ]

    async def check(
        self,
        input: str = "",
        output: str = "",
        context: dict | None = None,
    ) -> GuardrailResult:
        if not output:
            return GuardrailResult(passed=True)  # Skip during input phase

        text = output.strip()
        if len(text) < self.min_length:
            return GuardrailResult(
                passed=False,
                reason=f"Output too short ({len(text)} < {self.min_length})",
                severity="warn",
            )
        if len(text) > self.max_length:
            return GuardrailResult(
                passed=False,
                reason=f"Output too long ({len(text)} > {self.max_length})",
                severity="warn",
            )
        lower = text.lower()
        for pattern in self.refusal_patterns:
            if pattern.lower() in lower:
                return GuardrailResult(
                    passed=False,
                    reason=f"Refusal pattern detected: '{pattern}'",
                    severity="warn",
                )
        return GuardrailResult(passed=True)
```

Then update `boot.py:287-290`:
```python
    loop.guardrail_pipeline = GuardrailPipeline([
        CostGuardrail(max_usd=10.0),
        OutputGuardrail(min_length=1),  # Validates free-text output
    ])
```

**Step 4: Run tests**

Run: `cd sage-python && python -m pytest tests/test_guardrails.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add sage-python/src/sage/guardrails/builtin.py sage-python/src/sage/boot.py sage-python/tests/test_guardrails.py
git commit -m "fix(guardrails): replace misconfigured SchemaGuardrail with OutputGuardrail for text output"
```

---

### Task 2: Replace 3 trivial Z3 checks with Python native (2000x speedup)

**Files:**
- Modify: `sage-python/src/sage/contracts/z3_verify.py:40-198`
- Test: `sage-python/tests/test_z3_contracts.py`

**Problem:** capability_coverage, budget_feasibility, type_compatibility use Z3 (~950µs each) for checks that Python does in ~0.5µs. Provider assignment SAT is the only genuine Z3 use.

**Step 1: Write failing tests for Python-native versions**

```python
# In test_z3_contracts.py — add benchmark comparison tests
def test_capability_coverage_no_z3_required():
    """Verify capability_coverage works without z3 import."""
    # This test verifies the Python-native path works
    from sage.contracts.z3_verify import verify_capability_coverage
    from sage.contracts.task_node import TaskNode, TaskNodeBudget, IOSchema
    from sage.contracts.dag import TaskDAG

    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="a",
        capabilities_required=["code_gen"],
        budget=TaskNodeBudget(),
        input_schema=IOSchema(), output_schema=IOSchema(),
    ))
    result = verify_capability_coverage(dag, {"code_gen", "search"})
    assert result.satisfied is True

    result2 = verify_capability_coverage(dag, {"search"})
    assert result2.satisfied is False
    assert "code_gen" in (result2.counterexample or "")
```

**Step 2: Replace Z3 checks with Python native**

```python
# Replace lines 40-198 of z3_verify.py with:

def verify_capability_coverage(
    dag: TaskDAG,
    available_capabilities: set[str],
) -> ContractVerdict:
    """Verify every node's required capabilities are available."""
    missing = []
    for nid in dag.node_ids:
        node = dag.get_node(nid)
        if node.capabilities_required:
            m = set(node.capabilities_required) - available_capabilities
            if m:
                missing.append(f"node '{nid}' requires {m}")
    return ContractVerdict(
        satisfied=not missing,
        property_name="capability_coverage",
        counterexample="; ".join(missing) if missing else None,
    )


def verify_budget_feasibility(
    dag: TaskDAG,
    total_budget_usd: float,
) -> ContractVerdict:
    """Verify sum of per-node max_cost_usd <= total budget."""
    total = sum(
        dag.get_node(nid).budget.max_cost_usd
        for nid in dag.node_ids
        if dag.get_node(nid).budget.max_cost_usd > 0
    )
    ok = total <= total_budget_usd
    return ContractVerdict(
        satisfied=ok,
        property_name="budget_feasibility",
        counterexample=f"Total cost ${total:.4f} > budget ${total_budget_usd:.4f}" if not ok else None,
    )


def verify_type_compatibility(dag: TaskDAG) -> ContractVerdict:
    """Verify for each edge A->B, A.output_fields >= B.input_fields."""
    missing = []
    for from_id in dag.node_ids:
        src = dag.get_node(from_id)
        for to_id in dag.successors(from_id):
            dst = dag.get_node(to_id)
            for field_name in dst.input_schema.fields:
                if field_name not in src.output_schema.fields:
                    missing.append(f"edge {from_id}->{to_id}: '{field_name}' missing")
    return ContractVerdict(
        satisfied=not missing,
        property_name="type_compatibility",
        counterexample="; ".join(missing) if missing else None,
    )
```

**Step 3: Remove `_require_z3()` calls from replaced functions**

The `_require_z3()` function and Z3 import stay for `verify_provider_assignment()`.

**Step 4: Run tests**

Run: `cd sage-python && python -m pytest tests/test_z3_contracts.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add sage-python/src/sage/contracts/z3_verify.py sage-python/tests/test_z3_contracts.py
git commit -m "perf(contracts): replace 3 trivial Z3 checks with Python native (~2000x speedup)"
```

---

### Task 3: Fix routing keyword matching (substring → word boundary)

**Files:**
- Modify: `sage-python/src/sage/strategy/metacognition.py:197-224`
- Test: `sage-python/tests/test_metacognition.py`

**Problem:** `"test" in lower` matches "attestation", "What is a unit test?", etc.

**Step 1: Write failing test**

```python
import pytest
from sage.strategy.metacognition import ComplexityRouter

def test_heuristic_word_boundary():
    router = ComplexityRouter.__new__(ComplexityRouter)
    # "test" in a question context should NOT trigger tool_required
    profile = router._assess_heuristic("What is a unit test?")
    assert profile.tool_required is False  # Conceptual question, no tool needed

def test_heuristic_tool_required_for_imperative():
    router = ComplexityRouter.__new__(ComplexityRouter)
    # "test this code" should trigger tool_required
    profile = router._assess_heuristic("test this code for errors")
    assert profile.tool_required is True
```

**Step 2: Run to verify failure**

Run: `cd sage-python && python -m pytest tests/test_metacognition.py::test_heuristic_word_boundary -v`
Expected: FAIL (tool_required is True for "What is a unit test?")

**Step 3: Replace substring with word-boundary regex**

```python
# In metacognition.py, add import at top:
import re

# Replace lines 202-217 with:
    def _assess_heuristic(self, task: str) -> CognitiveProfile:
        """Fast keyword-based fallback (no LLM call)."""
        lower = task.lower()

        complexity = 0.3
        if re.search(r'\b(?:debug|fix|error|crash)\b', lower):
            complexity += 0.3
        if re.search(r'\b(?:optimize|evolve|design|architect)\b', lower):
            complexity += 0.2
        if len(task) > 500:
            complexity += 0.1

        uncertainty = 0.2
        if "?" in task:
            uncertainty += 0.2
        if re.search(r'\b(?:maybe|possibly|explore|investigate)\b', lower):
            uncertainty += 0.2

        # Imperative tool keywords — only match at start of sentence or after punctuation
        # to avoid false positives on conceptual questions like "What is a test?"
        tool_required = bool(re.search(
            r'(?:^|[.!?]\s*)\b(?:file|search|run|execute|compile|test|deploy)\b',
            lower,
        ))

        return CognitiveProfile(
            complexity=min(1.0, complexity),
            uncertainty=min(1.0, uncertainty),
            tool_required=tool_required,
            reasoning="heuristic",
        )
```

**Step 4: Run tests**

Run: `cd sage-python && python -m pytest tests/test_metacognition.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add sage-python/src/sage/strategy/metacognition.py sage-python/tests/test_metacognition.py
git commit -m "fix(routing): replace substring matching with word-boundary regex in heuristic router"
```

---

### Task 4: Fix cost estimation — use API usage_metadata

**Files:**
- Modify: `sage-python/src/sage/agent_loop.py:23-37`
- Modify: `sage-python/src/sage/llm/google.py` (extract usage_metadata)
- Test: `sage-python/tests/test_agent_loop.py`

**Problem:** `len(text)//4` is ±200% off for code/non-English. tiktoken is wrong tokenizer for Gemini (BPE ≠ SentencePiece).

**Step 1: Modify _estimate_tokens to accept actual token count**

```python
# Replace agent_loop.py lines 23-37 with:

# Approximate cost per 1K tokens (USD) for dashboard estimation.
# Used ONLY when API doesn't return usage_metadata.
_COST_PER_1K = {
    "gpt-5.3-codex": 0.03,
    "gpt-5.2": 0.06,
    "gemini-3.1-pro-preview": 0.007,
    "gemini-3-flash-preview": 0.0015,
    "gemini-3.1-flash-lite-preview": 0.0005,
    "gemini-2.5-flash-lite": 0.0003,
    "gemini-2.5-flash": 0.001,
}


def _estimate_tokens(text: str, actual_count: int | None = None) -> int:
    """Return actual token count from API if available, else rough estimate."""
    if actual_count is not None and actual_count > 0:
        return actual_count
    return max(1, len(text) // 4)
```

**Step 2: Update LLMProvider.generate() to return token counts**

In `sage-python/src/sage/llm/base.py`, add `usage` field to response:
```python
@dataclass
class LLMResponse:
    text: str
    model: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
```

In `sage-python/src/sage/llm/google.py`, extract usage_metadata:
```python
# After response = await client.models.generate_content_async(...)
usage = getattr(response, "usage_metadata", None)
prompt_tokens = getattr(usage, "prompt_token_count", None) if usage else None
completion_tokens = getattr(usage, "candidates_token_count", None) if usage else None
```

**Step 3: Run tests**

Run: `cd sage-python && python -m pytest tests/test_agent_loop.py tests/test_llm.py -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add sage-python/src/sage/agent_loop.py sage-python/src/sage/llm/base.py sage-python/src/sage/llm/google.py
git commit -m "fix(cost): use API usage_metadata for accurate token counting, fallback to heuristic"
```

---

### Task 5: Fix S-MMU context retrieval — return content not IDs

**Files:**
- Modify: `sage-python/src/sage/memory/smmu_context.py:72-77`
- Test: `sage-python/tests/test_smmu_context.py`

**Problem:** `retrieve_smmu_context()` returns `"Chunk 3 (relevance: 0.82)"` which is useless — the LLM needs the actual chunk content.

**Step 1: Update retrieval to include chunk summary**

```python
# Replace smmu_context.py lines 72-77 with:

        # Format as injectable context with chunk summaries
        lines = ["[S-MMU Graph Memory] Relevant context from compacted memory:"]
        for chunk_id, score in top_hits:
            # Try to get the chunk summary/content from working memory
            summary = ""
            try:
                summary = working_memory.get_chunk_summary(chunk_id)
            except (AttributeError, Exception):
                pass
            if summary:
                lines.append(f"- [{score:.2f}] {summary}")
            else:
                lines.append(f"- Chunk {chunk_id} (relevance: {score:.2f})")
```

**Step 2: Add `get_chunk_summary()` to working memory mock**

In `sage-python/src/sage/memory/working.py` mock class, add:
```python
def get_chunk_summary(self, chunk_id: int) -> str:
    return ""  # Mock returns empty
```

In Rust `sage-core/src/memory/smmu.rs`, add a method to retrieve chunk summary text from metadata.

**Step 3: Run tests**

Run: `cd sage-python && python -m pytest tests/test_smmu_context.py -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add sage-python/src/sage/memory/smmu_context.py sage-python/src/sage/memory/working.py
git commit -m "fix(smmu): retrieve chunk summaries instead of bare IDs in context injection"
```

---

### Task 6: Fix WebSocket auth — First-Message pattern

**Files:**
- Modify: `ui/app.py:436-444`
- Test: manual (WebSocket endpoint)

**Problem:** Token in query param leaks in logs, browser history, proxy logs.

**Step 1: Replace query param auth with First-Message auth**

```python
# Replace app.py lines 436-444 with:

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # First-Message authentication: client must send auth message within 3s
    if DASHBOARD_TOKEN:
        import asyncio
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=3.0)
            data = json.loads(raw)
            if data.get("action") != "auth" or data.get("token") != DASHBOARD_TOKEN:
                await websocket.close(code=4001, reason="Invalid auth")
                return
        except (asyncio.TimeoutError, json.JSONDecodeError, KeyError):
            await websocket.close(code=4001, reason="Auth timeout or invalid format")
            return
```

**Step 2: Update dashboard HTML client**

In `ui/static/index.html`, update the WebSocket connection code:
```javascript
// Replace ws connection with auth handshake:
ws = new WebSocket(wsUrl);
ws.onopen = () => {
    if (token) {
        ws.send(JSON.stringify({action: "auth", token: token}));
    }
};
```

**Step 3: Test manually**

Run: `python ui/app.py` and verify WebSocket connects with auth.

**Step 4: Commit**

```bash
git add ui/app.py ui/static/index.html
git commit -m "fix(dashboard): move WebSocket auth from query param to First-Message pattern"
```

---

### Task 7: Add CausalMemory SQLite persistence

**Files:**
- Modify: `sage-python/src/sage/memory/causal.py`
- Test: `sage-python/tests/test_causal_memory.py`

**Problem:** CausalMemory is in-memory only — lost on restart.

**Step 1: Write failing test**

```python
import pytest
import tempfile, os
from sage.memory.causal import CausalMemory

def test_causal_memory_persistence():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        cm = CausalMemory(db_path=db_path)
        cm.add_entity("A")
        cm.add_entity("B")
        cm.add_causal_edge("A", "caused", "B")
        cm.save()

        cm2 = CausalMemory(db_path=db_path)
        cm2.load()
        assert "A" in cm2.entities()
        assert "B" in cm2.entities()
    finally:
        os.unlink(db_path)
```

**Step 2: Implement save/load (same pattern as semantic.py)**

Add `save()` and `load()` methods using `sqlite3.connect()` with tables for entities, relations, and causal_edges.

**Step 3: Wire in boot.py**

```python
_causal_db = Path.home() / ".sage" / "causal.db"
causal_memory = CausalMemory(db_path=str(_causal_db))
causal_memory.load()
```

**Step 4: Run tests**

Run: `cd sage-python && python -m pytest tests/test_causal_memory.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add sage-python/src/sage/memory/causal.py sage-python/tests/test_causal_memory.py sage-python/src/sage/boot.py
git commit -m "feat(memory): add SQLite persistence for CausalMemory"
```

---

### Task 8: Fix z3_topology.py silent catch + add sentence-transformers to deps

**Files:**
- Modify: `sage-python/src/sage/topology/z3_topology.py:131-140`
- Modify: `sage-python/pyproject.toml`

**Step 1: Add logging to silent catch**

```python
# z3_topology.py line 139-140, replace:
        except Exception:
            return ""
# with:
        except Exception:
            logger.warning("Z3 topology verification failed (optional, continuing)", exc_info=True)
            return ""
```

**Step 2: Add sentence-transformers to pyproject.toml extras**

```toml
[project.optional-dependencies]
embeddings = ["sentence-transformers>=2.2"]
all = [..., "sentence-transformers>=2.2"]
```

**Step 3: Run tests**

Run: `cd sage-python && python -m pytest tests/ -v --tb=short -q`
Expected: All 691+ PASS

**Step 4: Commit**

```bash
git add sage-python/src/sage/topology/z3_topology.py sage-python/pyproject.toml
git commit -m "fix: add logging to z3_topology silent catch, add sentence-transformers to extras"
```

---

## Sprint 2 — Structural Fixes (3-5 days, 5 tasks)

### Task 9: S-MMU O(n²) → ANN index with usearch

**Files:**
- Modify: `sage-core/Cargo.toml` (add usearch dependency)
- Modify: `sage-core/src/memory/smmu.rs:121-150`
- Test: `sage-core/tests/smmu_bench.rs` (new)

**Problem:** register_chunk() iterates ALL chunks for cosine similarity. O(n²) total.

**Solution (oracle consensus):** Replace full scan with `usearch-rs` HNSW ANN index. Top-k=16→32 approximate → rerank exact cosine → threshold 0.5. Target: p95 < 1ms at 10K chunks.

**Step 1: Add usearch to Cargo.toml**

```toml
[dependencies]
usearch = { version = "2", optional = true }

[features]
ann = ["dep:usearch"]
```

**Step 2: Implement ANN-backed semantic edge construction**

Create `sage-core/src/memory/ann_index.rs` with HNSW wrapper, then modify `register_chunk()` to:
1. Insert embedding into ANN index
2. Query top-32 approximate neighbors
3. Rerank with exact cosine
4. Add edges for sim > 0.5

**Step 3: Add benchmark test**

```rust
#[test]
fn bench_register_1000_chunks() {
    let mut smmu = SMMUGraph::new();
    let start = std::time::Instant::now();
    for i in 0..1000 {
        let emb = vec![0.1; 384]; // Dummy 384-dim embedding
        smmu.register_chunk(format!("chunk_{i}"), Some(emb), vec!["test"], None);
    }
    let elapsed = start.elapsed();
    assert!(elapsed.as_millis() < 5000, "1000 chunks took {}ms", elapsed.as_millis());
}
```

**Step 4: Commit**

```bash
git add sage-core/
git commit -m "perf(smmu): replace O(n²) scan with usearch ANN index for semantic edges"
```

---

### Task 10: ExoCortex provider abstraction

**Files:**
- Create: `sage-python/src/sage/memory/rag_backend.py` (Protocol)
- Modify: `sage-python/src/sage/memory/remote_rag.py` (adapter)
- Test: `sage-python/tests/test_rag_backend.py`

**Problem:** ExoCortex hardcoded to Google File Search API.

**Solution:** Hexagonal Architecture with `KnowledgeStore(Protocol)`.

**Step 1: Define protocol**

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class KnowledgeStore(Protocol):
    async def search(self, query: str, top_k: int = 5) -> list[dict]: ...
    async def ingest(self, content: str, metadata: dict | None = None) -> str: ...
```

**Step 2: Make ExoCortex implement the protocol**

**Step 3: Add env var selection in boot.py**

```python
backend = os.environ.get("SAGE_RAG_BACKEND", "google")
```

**Step 4: Commit**

```bash
git add sage-python/src/sage/memory/rag_backend.py sage-python/src/sage/memory/remote_rag.py
git commit -m "refactor(exocortex): extract KnowledgeStore protocol for vendor independence"
```

---

### Task 11: Fix verify_provider_assignment encoding bug

**Files:**
- Modify: `sage-python/src/sage/contracts/z3_verify.py:282`
- Test: `sage-python/tests/test_z3_contracts.py`

**Problem (found by GPT-5.4 Codex):** Line 282 encodes `z3.Or(...)` = "at least one provider" but docstring says "exactly one". The constraint should be exactly-one (mutual exclusion between providers for same node).

**Step 1: Write failing test**

```python
def test_provider_assignment_exactly_one():
    """Provider assignment should assign exactly one provider per node."""
    # Create DAG with one node requiring capabilities both providers have
    # Verify the model assigns exactly one (not multiple)
    ...
```

**Step 2: Replace Or with exactly-one constraint**

```python
# Replace line 282:
solver.add(z3.Or(*node_vars.values()))
# With:
# Exactly one provider assigned: sum of bools == 1
solver.add(z3.PbEq([(v, 1) for v in node_vars.values()], 1))
```

**Step 3: Run tests**

Run: `cd sage-python && python -m pytest tests/test_z3_contracts.py -v`

**Step 4: Commit**

```bash
git add sage-python/src/sage/contracts/z3_verify.py sage-python/tests/test_z3_contracts.py
git commit -m "fix(z3): provider assignment uses exactly-one constraint (was at-least-one)"
```

---

### Task 12: Dashboard task queue (replace single-slot)

**Files:**
- Modify: `ui/app.py:259-263`

**Problem:** Returns 409 on concurrent task submission.

**Solution:** Add a simple asyncio.Queue for task queuing.

**Step 1: Add task queue to DashboardState**

```python
class DashboardState:
    def __init__(self):
        ...
        self.task_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        self.active_tasks: dict[str, asyncio.Task] = {}
```

**Step 2: Update run endpoint to queue instead of reject**

**Step 3: Commit**

```bash
git add ui/app.py
git commit -m "feat(dashboard): add task queue (replace single-slot 409 rejection)"
```

---

### Task 13: Honest documentation update

**Files:**
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `CLAUDE.md`

**Problem:** Several claims need updating after Sprint 1-2 fixes.

Updates needed:
- Z3: "3 checks use Python native, 1 uses genuine Z3 SAT"
- Routing: "word-boundary regex heuristic, LLM assessment available"
- Cost: "uses API usage_metadata when available, heuristic fallback"
- GuardRails: "OutputGuardrail for text, SchemaGuardrail for JSON mode"
- ExoCortex: "KnowledgeStore protocol, Google adapter default"
- Dashboard: "First-Message WebSocket auth"
- Memory: "CausalMemory now persistent (SQLite)"

**Step 1: Update all docs**

**Step 2: Commit**

```bash
git add README.md ARCHITECTURE.md CLAUDE.md
git commit -m "docs: sync documentation with audit response Sprint 1-2"
```

---

## Sprint 3 — Evidence & Validation (1-2 weeks, 5 tasks)

### Task 14: Run full HumanEval 164 — 3 configs

**Problem:** Only 20/164 published. No baseline. Statistiquement non significatif.

**Protocol:**
1. Config A: bare Gemini Flash (no routing, no memory, no guardrails)
2. Config B: routing only (S1/S2/S3, no memory injection)
3. Config C: full stack

```bash
# Run all 3 configs, commit traces
python -m sage.bench --type humaneval --baseline > docs/benchmarks/humaneval-bare.jsonl
python -m sage.bench --type humaneval > docs/benchmarks/humaneval-full.jsonl
```

**Commit traces to docs/benchmarks/**

---

### Task 15: Routing value proof — cost-performance frontier

**Protocol (oracle consensus — RouteLLM 2024 methodology):**
1. Dataset: 30+ unseen tasks (NOT the calibrated 30)
2. Baseline A: All tasks → S2 (best model)
3. Baseline B: All tasks → S1 (cheapest)
4. Router: ComplexityRouter decides
5. Proof: `SR(Router) >= SR(A) - 2%` AND `Cost(Router) < Cost(A) / 2`

---

### Task 16: Evolution engine validation protocol

**Protocol (GPT-5.4 Codex recommendation — 4-arm experiment):**
1. Arm 1: Full engine (DGM + SAMPO + MAP-Elites)
2. Arm 2: Random mutation + same archive
3. Arm 3: MAP-Elites sans SAMPO
4. Arm 4: Seed only (no evolution)
5. 20-30 seeds per task family, 3 families minimum
6. Decision: keep if ≥10% AUC improvement vs random

---

### Task 17: Memory vs long-context ablation

**Protocol:**
1. Config A: full 4-tier memory system
2. Config B: no memory (long context window only)
3. Config C: episodic + semantic only (no S-MMU/ExoCortex)
4. Measure: task success, cost, latency over 20+ multi-turn tasks

---

### Task 18: Update MEMORY.md and docs with all results

After all evidence collected, update:
- `MEMORY.md` with new phase status
- `ARCHITECTURE.md` with evidence levels
- `README.md` with honest benchmark numbers
- `docs/audits/2026-03-08-audit-response-v2.md` with verification report

---

## Priority Summary

| Sprint | Tasks | Effort | Impact |
|--------|-------|--------|--------|
| Sprint 1 | Tasks 1-8 | 1 day | Fix 8 confirmed issues (guardrails, Z3, routing, cost, SMMU content, WS auth, causal persistence, silent catch) |
| Sprint 2 | Tasks 9-13 | 3-5 days | ANN index, ExoCortex abstraction, Z3 bug, dashboard queue, docs |
| Sprint 3 | Tasks 14-18 | 1-2 weeks | Full HumanEval, routing proof, evolution validation, memory ablation |

**Critical path:** Sprint 1 unblocks Sprint 3 (fixes must be in place before benchmarks).
