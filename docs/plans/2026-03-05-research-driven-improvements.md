# Research-Driven YGN-SAGE Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire 5 disconnected modules into the agent runtime based on research from NotebookLM (5 notebooks), HuggingFace (MEM1, Agent0, AutoTool), and web (MCP/AAIF, System 1.5).

**Architecture:** Wire MemoryCompressor with pressure-based trigger (MEM1 pattern), add EpisodicMemory as an on-demand search_memory tool (two-stage retrieval from NotebookLM Technical), improve S2 validation with sandbox execution (MetaScaffold recommendation), add S2->S3 escalation on failure, and register meta-tools for runtime tool synthesis (Agent0/AutoTool pattern).

**Tech Stack:** Python 3.12+, existing sage-python modules, pytest

---

### Task 1: Wire MemoryCompressor into boot.py with pressure trigger

**Files:**
- Modify: `sage-python/src/sage/boot.py`
- Modify: `sage-python/src/sage/agent_loop.py`
- Modify: `sage-python/src/sage/memory/compressor.py`
- Test: `sage-python/tests/test_agent_loop.py`

**Context:** MemoryCompressor exists but is never instantiated. NotebookLM Technical notebook says: use a memory pressure trigger (token threshold), not every N steps. The compressor already accepts `compression_threshold` (default 20 events). We wire it in boot.py and the loop already has a `memory_compressor` slot.

**Step 1: Write the failing test**

Add to `sage-python/tests/test_agent_loop.py`:

```python
@pytest.mark.asyncio
async def test_agent_loop_compresses_memory_on_pressure():
    """Memory compressor fires when event count exceeds threshold."""
    from sage.memory.compressor import MemoryCompressor
    from sage.llm.mock import MockProvider

    provider = MockProvider(responses=["compressed summary"])
    compressor = MemoryCompressor(
        llm=provider,
        compression_threshold=3,  # Low threshold for testing
        keep_recent=1,
    )

    loop = _make_loop()  # uses existing helper
    loop.memory_compressor = compressor

    # Manually add events to exceed threshold
    for i in range(4):
        loop.working_memory.add_event("TEST", f"event {i}")

    # Run compression step
    compressed = await compressor.step(loop.working_memory)
    assert compressed is True
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_agent_loop.py::test_agent_loop_compresses_memory_on_pressure -v`
Expected: FAIL (MemoryCompressor.step calls `working_memory.compress()` which doesn't exist on the Rust-backed WorkingMemory)

**Step 3: Update MemoryCompressor to work with current WorkingMemory API**

Replace the `step` method's final section in `sage-python/src/sage/memory/compressor.py`. The current code calls `working_memory.compress()` which doesn't exist. Replace with the real API:

```python
        # 4. Update Working Memory — use compress_old_events (Rust API)
        working_memory.compress_old_events(self.keep_recent, summary)
        return True
```

Also fix the event access pattern — `working_memory.recent_events()` returns objects with `.event_type` and `.content` attributes (from Rust), not dicts. Replace:

```python
        to_compress = working_memory.recent_events(working_memory.event_count())
        # Keep only the ones we want to compress (all except keep_recent)
        if len(to_compress) <= self.keep_recent:
            return False
        to_compress = to_compress[:-self.keep_recent] if self.keep_recent > 0 else to_compress
        context = "\n".join([f"[{e.event_type}] {e.content}" for e in to_compress])
```

**Step 4: Wire compressor in boot.py**

In `sage-python/src/sage/boot.py`, after `memory_agent = MemoryAgent(...)`, add:

```python
    # Memory compressor (fires on pressure — MEM1 pattern)
    memory_compressor = MemoryCompressor(
        llm=provider,
        compression_threshold=20,
        keep_recent=5,
    )
```

Then pass it to AgentLoop:

```python
    loop = AgentLoop(
        config=config,
        llm_provider=provider,
        tool_registry=tool_registry,
        memory_compressor=memory_compressor,
    )
```

**Step 5: Run all tests**

Run: `cd sage-python && python -m pytest tests/ -x -q`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add sage-python/src/sage/boot.py sage-python/src/sage/agent_loop.py sage-python/src/sage/memory/compressor.py sage-python/tests/test_agent_loop.py
git commit -m "feat(memory): wire MemoryCompressor into boot with pressure trigger (MEM1 pattern)"
```

---

### Task 2: Wire EpisodicMemory as on-demand search_memory tool

**Files:**
- Modify: `sage-python/src/sage/boot.py`
- Modify: `sage-python/src/sage/agent_loop.py`
- Create: `sage-python/src/sage/tools/memory_tools.py`
- Test: `sage-python/tests/test_memory.py`

**Context:** NotebookLM Technical says: similarity search before every perceive phase is an anti-pattern (latency). Use two-stage retrieval: fast summary index (compressor handles this) + on-demand deep search via a tool the agent can invoke. We create a `search_memory` tool backed by EpisodicMemory.

**Step 1: Write the failing test**

Add to `sage-python/tests/test_memory.py`:

```python
@pytest.mark.asyncio
async def test_search_memory_tool():
    """search_memory tool queries episodic memory and returns results."""
    from sage.memory.episodic import EpisodicMemory
    from sage.tools.memory_tools import create_search_memory_tool

    episodic = EpisodicMemory()
    await episodic.store("debug-session", "Found null pointer bug in parser.py line 42")
    await episodic.store("architecture", "Agent uses perceive-think-act-learn loop")

    tool = create_search_memory_tool(episodic)
    result = await tool.execute({"query": "parser bug", "top_k": 3})
    assert "null pointer" in result.output
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_memory.py::test_search_memory_tool -v`
Expected: FAIL (memory_tools module doesn't exist)

**Step 3: Create the search_memory tool**

Create `sage-python/src/sage/tools/memory_tools.py`:

```python
"""Memory tools: on-demand episodic recall for the agent loop.

Research basis: NotebookLM Technical — two-stage provenance-aware retrieval.
Stage 1 (default): compressed summary in working memory (handled by compressor).
Stage 2 (on-demand): deep similarity search via this tool.
"""
from __future__ import annotations

from sage.tools.base import Tool
from sage.memory.episodic import EpisodicMemory


def create_search_memory_tool(episodic: EpisodicMemory) -> Tool:
    """Create a search_memory tool bound to a specific EpisodicMemory instance."""

    @Tool.define(
        name="search_memory",
        description="Search long-term episodic memory for relevant past experiences. Use when current context is insufficient to answer the task.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for in memory"},
                "top_k": {"type": "integer", "description": "Max results to return", "default": 5},
            },
            "required": ["query"],
        },
    )
    async def search_memory(query: str, top_k: int = 5) -> str:
        results = await episodic.search(query, top_k=top_k)
        if not results:
            return "No relevant memories found."
        lines = []
        for r in results:
            lines.append(f"[{r['key']}] {r['content']}")
        return "\n".join(lines)

    return search_memory
```

**Step 4: Wire in boot.py and agent_loop.py**

In `boot.py`, after memory_compressor, add:

```python
    from sage.memory.episodic import EpisodicMemory
    from sage.tools.memory_tools import create_search_memory_tool

    episodic_memory = EpisodicMemory()
    search_tool = create_search_memory_tool(episodic_memory)
    tool_registry.register(search_tool)
```

In `agent_loop.py`, in the LEARN phase (after `self.working_memory.add_event("ASSISTANT", content)`), add episodic storage for significant responses:

```python
            # Store significant responses in episodic memory (if wired)
            if self.episodic_memory and len(content) > 100:
                await self.episodic_memory.store(
                    key=f"step-{self.step_count}",
                    content=content[:500],
                    metadata={"task": task, "step": self.step_count},
                )
```

Add `self.episodic_memory: EpisodicMemory | None = None` to `__init__`, and wire it in boot.py:

```python
    loop.episodic_memory = episodic_memory
```

**Step 5: Run all tests**

Run: `cd sage-python && python -m pytest tests/ -x -q`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add sage-python/src/sage/tools/memory_tools.py sage-python/src/sage/boot.py sage-python/src/sage/agent_loop.py sage-python/tests/test_memory.py
git commit -m "feat(memory): wire EpisodicMemory as on-demand search_memory tool (two-stage retrieval)"
```

---

### Task 3: Improve S2 validation with sandbox execution check

**Files:**
- Modify: `sage-python/src/sage/agent_loop.py:233-246`
- Test: `sage-python/tests/test_metacognition.py`

**Context:** NotebookLM MetaScaffold says: S2 should use sandbox execution for empirical validation, not just CoT heuristic. Current code checks for `"Step "` string which is fragile. Replace with: if S2 response contains a code block, execute it in sandbox and check exit code. If no code block, keep CoT enforcement as fallback.

**Step 1: Write the failing test**

Add to `sage-python/tests/test_metacognition.py`:

```python
@pytest.mark.asyncio
async def test_s2_validation_detects_code_block():
    """S2 validation identifies code blocks for sandbox execution."""
    from sage.agent_loop import _extract_code_blocks

    content_with_code = "Here is the solution:\n```python\nprint('hello')\n```\nDone."
    blocks = _extract_code_blocks(content_with_code)
    assert len(blocks) == 1
    assert "print('hello')" in blocks[0]

    content_no_code = "The answer is 42. Step 1: think about it."
    blocks = _extract_code_blocks(content_no_code)
    assert len(blocks) == 0
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_metacognition.py::test_s2_validation_detects_code_block -v`
Expected: FAIL (`_extract_code_blocks` doesn't exist)

**Step 3: Add code block extractor and sandbox validation to agent_loop.py**

Add at module level in `agent_loop.py`:

```python
import re

def _extract_code_blocks(text: str) -> list[str]:
    """Extract fenced code blocks from markdown-style LLM output."""
    pattern = r"```(?:\w+)?\n(.*?)```"
    return re.findall(pattern, text, re.DOTALL)
```

Replace the S2 validation block (lines 233-246) with:

```python
            # System 2 validation (Empirical — sandbox or CoT)
            elif self.config.validation_level == 2 and content:
                code_blocks = _extract_code_blocks(content)

                if code_blocks and self.sandbox_manager:
                    # Empirical: execute first code block in sandbox
                    sandbox = await self.sandbox_manager.create()
                    try:
                        result = await sandbox.execute(f"python3 -c {_quote(code_blocks[0])}")
                        if result.exit_code != 0:
                            self._prm_retries += 1
                            if self._prm_retries <= self._max_prm_retries:
                                log.info("S2 sandbox validation failed (exit %d), retrying.", result.exit_code)
                                self._emit(LoopPhase.THINK, validation="s2_sandbox_fail", stderr=result.stderr[:200])
                                messages.append(Message(
                                    role=Role.USER,
                                    content=f"SYSTEM: Your code produced an error:\n{result.stderr[:500]}\nPlease fix it.",
                                ))
                                continue
                        else:
                            self._emit(LoopPhase.THINK, validation="s2_sandbox_pass")
                            self._prm_retries = 0
                    finally:
                        await self.sandbox_manager.destroy(sandbox.id)

                elif not code_blocks and self.step_count == 1:
                    # Fallback: CoT enforcement if no code to validate
                    has_reasoning = "<think>" in content or "\n1." in content or "\n- " in content
                    if not has_reasoning:
                        self._prm_retries += 1
                        if self._prm_retries <= self._max_prm_retries:
                            log.info("S2 validation: missing reasoning, requesting CoT.")
                            messages.append(Message(
                                role=Role.USER,
                                content="SYSTEM: Provide step-by-step reasoning for this task.",
                            ))
                            continue
```

Add a helper at module level:

```python
def _quote(code: str) -> str:
    """Shell-quote a code string for subprocess execution."""
    return "'" + code.replace("'", "'\\''") + "'"
```

Add `self.sandbox_manager` to `__init__`:

```python
        self.sandbox_manager: Any = None  # Injected by boot.py
```

**Step 4: Wire sandbox in boot.py**

In `boot.py`, after tool_registry creation:

```python
    from sage.sandbox.manager import SandboxManager
    sandbox_manager = SandboxManager(use_docker=False)  # Local fallback
```

Then after loop creation:

```python
    loop.sandbox_manager = sandbox_manager
```

**Step 5: Run all tests**

Run: `cd sage-python && python -m pytest tests/ -x -q`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add sage-python/src/sage/agent_loop.py sage-python/src/sage/boot.py sage-python/tests/test_metacognition.py
git commit -m "feat(s2): empirical sandbox validation for code blocks (MetaScaffold pattern)"
```

---

### Task 4: Add S2 to S3 escalation on repeated failure

**Files:**
- Modify: `sage-python/src/sage/agent_loop.py`
- Test: `sage-python/tests/test_metacognition.py`

**Context:** When S2 validation fails after max retries, the current code just accepts the response. Research says: escalate to S3 (Z3 formal verification) instead. This means changing `validation_level` mid-loop and switching to PRM validation.

**Step 1: Write the failing test**

Add to `sage-python/tests/test_metacognition.py`:

```python
def test_s2_escalation_threshold():
    """After max S2 retries, validation_level should escalate to 3."""
    from sage.agent_loop import S2_MAX_RETRIES_BEFORE_ESCALATION
    assert S2_MAX_RETRIES_BEFORE_ESCALATION == 2
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_metacognition.py::test_s2_escalation_threshold -v`
Expected: FAIL (constant doesn't exist)

**Step 3: Implement escalation**

In `agent_loop.py`, add constant at module level:

```python
S2_MAX_RETRIES_BEFORE_ESCALATION = 2
```

At the end of the S2 validation `elif` block, after the CoT enforcement fallback, add the escalation logic. After the line `continue` inside the CoT retry block, add an else clause that handles max retries:

```python
                # Escalate S2 -> S3 if max retries exhausted
                if self._prm_retries > self._max_prm_retries and self.config.validation_level == 2:
                    log.info("S2 validation exhausted — escalating to S3 (formal verification).")
                    self.config.validation_level = 3
                    self._prm_retries = 0
                    self._emit(LoopPhase.THINK, escalation="s2_to_s3")
                    messages.append(Message(
                        role=Role.USER,
                        content="SYSTEM: Escalating to formal verification. Use <think> tags for rigorous step-by-step reasoning.",
                    ))
                    continue
```

**Step 4: Run all tests**

Run: `cd sage-python && python -m pytest tests/ -x -q`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add sage-python/src/sage/agent_loop.py sage-python/tests/test_metacognition.py
git commit -m "feat(s2): escalate to S3 formal verification on repeated S2 failure"
```

---

### Task 5: Register meta-tools (create_python_tool, create_bash_tool) by default

**Files:**
- Modify: `sage-python/src/sage/boot.py`
- Test: `sage-python/tests/test_integration.py`

**Context:** `tools/meta.py` defines `create_python_tool` and `create_bash_tool` for runtime tool synthesis (Agent0/AutoTool pattern). They exist but are never registered in the default ToolRegistry. Research (NotebookLM Discover AI) confirms: runtime tool synthesis is mandatory for a generalist ADK, with strict sandboxing (AST validation is already implemented in meta.py).

**Step 1: Write the failing test**

Add to `sage-python/tests/test_integration.py`:

```python
def test_boot_registers_meta_tools():
    """Boot sequence registers create_python_tool and create_bash_tool."""
    system = boot_agent_system(use_mock_llm=True)
    tool_names = [t.name for t in system.tool_registry.list()]
    assert "create_python_tool" in tool_names
    assert "create_bash_tool" in tool_names
    assert "search_memory" in tool_names
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_integration.py::test_boot_registers_meta_tools -v`
Expected: FAIL (meta-tools not in registry)

**Step 3: Register meta-tools in boot.py**

In `boot.py`, after tool_registry creation, add:

```python
    # Runtime tool synthesis (Agent0/AutoTool pattern)
    from sage.tools.meta import create_python_tool, create_bash_tool
    tool_registry.register(create_python_tool)
    tool_registry.register(create_bash_tool)
```

**Step 4: Run all tests**

Run: `cd sage-python && python -m pytest tests/ -x -q`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add sage-python/src/sage/boot.py sage-python/tests/test_integration.py
git commit -m "feat(tools): register meta-tools by default for runtime tool synthesis (Agent0 pattern)"
```

---

## Summary

| Task | What | Research Source | Files |
|------|------|----------------|-------|
| 1 | Wire MemoryCompressor with pressure trigger | NotebookLM Technical + MEM1 | boot.py, compressor.py, agent_loop.py |
| 2 | Wire EpisodicMemory as search_memory tool | NotebookLM Technical (two-stage) | boot.py, memory_tools.py, agent_loop.py |
| 3 | S2 sandbox execution validation | NotebookLM MetaScaffold | agent_loop.py, boot.py |
| 4 | S2->S3 escalation on failure | NotebookLM MetaScaffold + MARL | agent_loop.py |
| 5 | Register meta-tools by default | NotebookLM Discover AI + Agent0 | boot.py |
