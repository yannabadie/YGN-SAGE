# V2 Research-Driven Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade YGN-SAGE memory and S2 validation from v1 (wired but basic) to v2 (MEM1 per-step internal state, AgeMem 6-tool memory interface, AVR sandbox loop) based on enriched NotebookLM findings (161 sources).

**Architecture:** Three upgrades: (1) MemoryCompressor gains a `generate_internal_state()` mode that runs every step to produce a rolling `<IS_t>` summary, replacing the threshold-only approach. (2) EpisodicMemory gets `update()`/`delete()` methods, and 5 new tools are registered alongside the existing `search_memory` to give the LLM full CRUD control over both STM and LTM. (3) S2 validation becomes a proper Act-Verify-Refine loop with configurable max iterations and CGRS budget monitoring before escalation.

**Tech Stack:** Python 3.12+, existing sage-python modules, pytest

---

### Task 1: Add per-step Internal State generation to MemoryCompressor (MEM1)

**Files:**
- Modify: `sage-python/src/sage/memory/compressor.py`
- Modify: `sage-python/src/sage/agent_loop.py`
- Test: `sage-python/tests/test_agent_loop.py`

**Context:** MEM1 paper says the compressor should run every step to maintain a rolling internal state `<IS_t>` that merges old state with new observations. The current `step()` only fires when `event_count >= threshold`. We add a new `generate_internal_state()` method that always runs, producing a compact summary of the last few events merged with the previous IS. The existing `step()` (threshold-based bulk compression) stays for backwards compatibility.

**Step 1: Write the failing test**

Add to `sage-python/tests/test_agent_loop.py`:

```python
@pytest.mark.asyncio
async def test_compressor_generates_internal_state():
    """MEM1: compressor generates rolling <IS_t> every step."""
    from sage.memory.compressor import MemoryCompressor
    from sage.llm.mock import MockProvider

    provider = MockProvider(responses=[
        "Current state: user asked about sorting algorithms. Explored quicksort.",
        "Current state: user asked about sorting. Explored quicksort and mergesort. Quicksort preferred for average case.",
    ])
    compressor = MemoryCompressor(llm=provider, compression_threshold=20, keep_recent=5)

    # Step 1: generate IS from scratch
    is_1 = await compressor.generate_internal_state("User asked: explain sorting algorithms")
    assert is_1 != ""
    assert compressor.internal_state == is_1

    # Step 2: generate IS that merges with previous
    is_2 = await compressor.generate_internal_state("Assistant explained quicksort and mergesort")
    assert is_2 != ""
    assert compressor.internal_state == is_2
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_agent_loop.py::test_compressor_generates_internal_state -v`
Expected: FAIL (`generate_internal_state` doesn't exist)

**Step 3: Implement generate_internal_state in compressor.py**

Add to `MemoryCompressor.__init__`:

```python
        self.internal_state: str = ""  # Rolling MEM1 <IS_t>
```

Add method to `MemoryCompressor`:

```python
    async def generate_internal_state(self, new_observation: str) -> str:
        """MEM1: generate rolling internal state by merging previous IS with new observation.

        Runs every step. Produces a compact summary that replaces the previous state.
        This prevents context window expansion by keeping memory near-constant size.
        """
        if self.internal_state:
            prompt = (
                f"You are maintaining a rolling internal state for an AI agent.\n"
                f"Previous state:\n{self.internal_state}\n\n"
                f"New observation:\n{new_observation}\n\n"
                f"Produce an updated internal state that merges the previous state "
                f"with the new observation. Be concise (max 3 sentences). "
                f"Drop details that are no longer relevant."
            )
        else:
            prompt = (
                f"You are maintaining a rolling internal state for an AI agent.\n"
                f"First observation:\n{new_observation}\n\n"
                f"Produce a concise internal state summary (max 2 sentences)."
            )

        response = await self.llm.generate(
            messages=[Message(role=Role.USER, content=prompt)]
        )
        self.internal_state = response.content or new_observation[:200]
        return self.internal_state
```

**Step 4: Wire IS generation in agent_loop.py**

In `agent_loop.py`, in the `run()` method, after the THINK phase emits content (after line ~231 `brake=brake,`), add IS generation:

```python
            # MEM1: generate rolling internal state every step
            if self.memory_compressor and content:
                await self.memory_compressor.generate_internal_state(
                    f"[Step {self.step_count}] {content[:300]}"
                )
```

**Step 5: Run all tests**

Run: `cd sage-python && python -m pytest tests/ -x -q`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add sage-python/src/sage/memory/compressor.py sage-python/src/sage/agent_loop.py sage-python/tests/test_agent_loop.py
git commit -m "feat(memory): MEM1 per-step internal state generation in MemoryCompressor"
```

---

### Task 2: Add update() and delete() to EpisodicMemory

**Files:**
- Modify: `sage-python/src/sage/memory/episodic.py`
- Test: `sage-python/tests/test_memory.py`

**Context:** AgeMem paper requires the agent to have full CRUD over long-term memory. EpisodicMemory currently has `store()` and `search()`. We add `update()` (modify content/metadata by key) and `delete()` (remove by key). Also add `list_keys()` for the filter tool.

**Step 1: Write the failing tests**

Add to `sage-python/tests/test_memory.py`:

```python
@pytest.mark.asyncio
async def test_episodic_memory_update():
    """AgeMem: agent can update existing memory entries."""
    mem = EpisodicMemory()
    await mem.store("auth-fix", "Fixed auth by checking token expiry")
    updated = await mem.update("auth-fix", content="Fixed auth by adding JWT refresh + token expiry check")
    assert updated is True
    results = await mem.search("auth")
    assert "JWT refresh" in results[0]["content"]


@pytest.mark.asyncio
async def test_episodic_memory_update_nonexistent():
    mem = EpisodicMemory()
    updated = await mem.update("nonexistent", content="new")
    assert updated is False


@pytest.mark.asyncio
async def test_episodic_memory_delete():
    """AgeMem: agent can delete obsolete memory entries."""
    mem = EpisodicMemory()
    await mem.store("temp-debug", "Temporary debug finding")
    await mem.store("permanent", "Important architecture note")
    deleted = await mem.delete("temp-debug")
    assert deleted is True
    results = await mem.search("debug")
    assert len(results) == 0
    results = await mem.search("architecture")
    assert len(results) == 1


@pytest.mark.asyncio
async def test_episodic_memory_delete_nonexistent():
    mem = EpisodicMemory()
    deleted = await mem.delete("nonexistent")
    assert deleted is False


@pytest.mark.asyncio
async def test_episodic_memory_list_keys():
    mem = EpisodicMemory()
    await mem.store("key-a", "content a")
    await mem.store("key-b", "content b")
    keys = mem.list_keys()
    assert set(keys) == {"key-a", "key-b"}
```

**Step 2: Run tests to verify they fail**

Run: `cd sage-python && python -m pytest tests/test_memory.py::test_episodic_memory_update -v`
Expected: FAIL (`update` not found)

**Step 3: Implement update, delete, list_keys in episodic.py**

Add to `EpisodicMemory`:

```python
    async def update(
        self,
        key: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update an existing memory entry by key. Returns False if not found."""
        for entry in self._entries:
            if entry["key"] == key:
                if content is not None:
                    entry["content"] = content
                if metadata is not None:
                    entry["metadata"] = metadata
                return True
        return False

    async def delete(self, key: str) -> bool:
        """Delete a memory entry by key. Returns False if not found."""
        for i, entry in enumerate(self._entries):
            if entry["key"] == key:
                self._entries.pop(i)
                return True
        return False

    def list_keys(self) -> list[str]:
        """List all memory entry keys."""
        return [e["key"] for e in self._entries]
```

**Step 4: Run all tests**

Run: `cd sage-python && python -m pytest tests/ -x -q`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add sage-python/src/sage/memory/episodic.py sage-python/tests/test_memory.py
git commit -m "feat(memory): add update/delete/list_keys to EpisodicMemory (AgeMem CRUD)"
```

---

### Task 3: Create AgeMem memory tools (6 total)

**Files:**
- Modify: `sage-python/src/sage/tools/memory_tools.py`
- Test: `sage-python/tests/test_memory.py`

**Context:** AgeMem paper says the agent needs 6 tool actions: 3 for STM (working memory) and 3 for LTM (episodic). The existing `search_memory` becomes one of the LTM tools. We create a factory function that produces all 6 tools bound to the correct memory instances.

STM tools (operate on WorkingMemory):
- `retrieve_context` — get recent N events from working memory
- `summarize_context` — get the compressor's current `<IS_t>` internal state
- `filter_context` — compress/trim working memory to keep only recent events

LTM tools (operate on EpisodicMemory):
- `search_memory` — already exists, search by query
- `store_memory` — store a new episodic memory entry
- `update_memory` — update existing entry by key
- `delete_memory` — delete entry by key

**Step 1: Write the failing tests**

Add to `sage-python/tests/test_memory.py`:

```python
@pytest.mark.asyncio
async def test_store_memory_tool():
    """store_memory tool stores entries in episodic memory."""
    from sage.tools.memory_tools import create_memory_tools
    from sage.memory.working import WorkingMemory

    wm = WorkingMemory(agent_id="test")
    episodic = EpisodicMemory()
    compressor = None  # No compressor needed for this test

    tools = create_memory_tools(wm, episodic, compressor)
    store_tool = next(t for t in tools if t.spec.name == "store_memory")

    result = await store_tool.execute({"key": "finding-1", "content": "Bug in parser line 42"})
    assert not result.is_error
    assert "stored" in result.output.lower() or "success" in result.output.lower()

    # Verify it's searchable
    search_tool = next(t for t in tools if t.spec.name == "search_memory")
    result = await search_tool.execute({"query": "parser bug"})
    assert "parser" in result.output.lower()


@pytest.mark.asyncio
async def test_retrieve_context_tool():
    """retrieve_context tool returns recent working memory events."""
    from sage.tools.memory_tools import create_memory_tools
    from sage.memory.working import WorkingMemory

    wm = WorkingMemory(agent_id="test")
    wm.add_event("USER", "What is quicksort?")
    wm.add_event("ASSISTANT", "Quicksort is a divide-and-conquer sorting algorithm.")

    tools = create_memory_tools(wm, EpisodicMemory(), None)
    retrieve_tool = next(t for t in tools if t.spec.name == "retrieve_context")

    result = await retrieve_tool.execute({"n": 2})
    assert "quicksort" in result.output.lower()


@pytest.mark.asyncio
async def test_delete_memory_tool():
    """delete_memory tool removes entries from episodic memory."""
    from sage.tools.memory_tools import create_memory_tools
    from sage.memory.working import WorkingMemory

    episodic = EpisodicMemory()
    await episodic.store("temp", "Temporary note")

    tools = create_memory_tools(WorkingMemory(agent_id="test"), episodic, None)
    delete_tool = next(t for t in tools if t.spec.name == "delete_memory")

    result = await delete_tool.execute({"key": "temp"})
    assert not result.is_error
    assert episodic.list_keys() == []
```

**Step 2: Run tests to verify they fail**

Run: `cd sage-python && python -m pytest tests/test_memory.py::test_store_memory_tool -v`
Expected: FAIL (`create_memory_tools` doesn't exist)

**Step 3: Rewrite memory_tools.py with all 6 tools**

Replace the entire contents of `sage-python/src/sage/tools/memory_tools.py`:

```python
"""AgeMem memory tools: 6 learnable actions for STM + LTM management.

Research basis:
- AgeMem: Unified tool-based memory policy (RETRIEVE/SUMMARY/FILTER + ADD/UPDATE/DELETE)
- NotebookLM Technical: Two-stage provenance-aware retrieval
- MEM1: Rolling internal state via compressor
"""
from __future__ import annotations

from typing import Any

from sage.tools.base import Tool
from sage.memory.episodic import EpisodicMemory
from sage.memory.working import WorkingMemory


def create_memory_tools(
    working_memory: WorkingMemory,
    episodic: EpisodicMemory,
    compressor: Any | None = None,
) -> list[Tool]:
    """Create all 6 AgeMem memory tools bound to the memory instances.

    STM (Working Memory): retrieve_context, summarize_context, filter_context
    LTM (Episodic): search_memory, store_memory, update_memory, delete_memory
    """
    tools: list[Tool] = []

    # --- STM Tools (Working Memory) ---

    @Tool.define(
        name="retrieve_context",
        description="Retrieve the N most recent events from short-term working memory.",
        parameters={
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Number of recent events to retrieve", "default": 10},
            },
            "required": [],
        },
    )
    async def retrieve_context(n: int = 10) -> str:
        events = working_memory.recent_events(n)
        if not events:
            return "Working memory is empty."
        lines = [f"[{e['type']}] {e['content']}" for e in events]
        return "\n".join(lines)

    tools.append(retrieve_context)

    @Tool.define(
        name="summarize_context",
        description="Get the current internal state summary (rolling MEM1 <IS_t>) of the agent's memory. Returns the compressed understanding of everything so far.",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    async def summarize_context() -> str:
        if compressor and hasattr(compressor, "internal_state") and compressor.internal_state:
            return f"Internal State: {compressor.internal_state}"
        # Fallback: return working memory context string
        ctx = working_memory.to_context_string()
        return ctx[:1000] if ctx else "No context available."

    tools.append(summarize_context)

    @Tool.define(
        name="filter_context",
        description="Trim working memory to keep only the N most recent events. Use to drop irrelevant early context and prevent token overflow.",
        parameters={
            "type": "object",
            "properties": {
                "keep_recent": {"type": "integer", "description": "Number of recent events to keep", "default": 5},
            },
            "required": [],
        },
    )
    async def filter_context(keep_recent: int = 5) -> str:
        count_before = working_memory.event_count()
        if count_before <= keep_recent:
            return f"Nothing to filter. Working memory has {count_before} events."
        working_memory.compress(keep_recent, "Filtered by agent request.")
        return f"Filtered working memory from {count_before} to ~{keep_recent} events."

    tools.append(filter_context)

    # --- LTM Tools (Episodic Memory) ---

    @Tool.define(
        name="search_memory",
        description="Search long-term episodic memory for relevant past experiences. Use when current context is insufficient.",
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
        lines = [f"[{r['key']}] {r['content']}" for r in results]
        return "\n".join(lines)

    tools.append(search_memory)

    @Tool.define(
        name="store_memory",
        description="Store a new entry in long-term episodic memory for future retrieval. Use for important findings, patterns, or facts worth remembering.",
        parameters={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Unique identifier for this memory (e.g. 'auth-bug-fix')"},
                "content": {"type": "string", "description": "The content to store"},
            },
            "required": ["key", "content"],
        },
    )
    async def store_memory(key: str, content: str) -> str:
        await episodic.store(key, content)
        return f"Stored memory '{key}' successfully."

    tools.append(store_memory)

    @Tool.define(
        name="update_memory",
        description="Update an existing long-term memory entry. Use when a previously stored fact needs correction or enrichment.",
        parameters={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Key of the memory to update"},
                "content": {"type": "string", "description": "New content to replace the old"},
            },
            "required": ["key", "content"],
        },
    )
    async def update_memory(key: str, content: str) -> str:
        updated = await episodic.update(key, content=content)
        if updated:
            return f"Updated memory '{key}' successfully."
        return f"Memory '{key}' not found. Use store_memory to create it."

    tools.append(update_memory)

    @Tool.define(
        name="delete_memory",
        description="Delete an obsolete or incorrect long-term memory entry.",
        parameters={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Key of the memory to delete"},
            },
            "required": ["key"],
        },
    )
    async def delete_memory(key: str) -> str:
        deleted = await episodic.delete(key)
        if deleted:
            return f"Deleted memory '{key}' successfully."
        return f"Memory '{key}' not found."

    tools.append(delete_memory)

    return tools
```

**Step 4: Run all tests**

Run: `cd sage-python && python -m pytest tests/ -x -q`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add sage-python/src/sage/tools/memory_tools.py sage-python/tests/test_memory.py
git commit -m "feat(memory): AgeMem 6-tool memory interface (3 STM + 4 LTM)"
```

---

### Task 4: Wire all memory tools in boot.py

**Files:**
- Modify: `sage-python/src/sage/boot.py`
- Modify: `sage-python/tests/test_integration.py`

**Context:** boot.py currently registers only `search_memory` via `create_search_memory_tool()`. Replace with `create_memory_tools()` which returns all 7 tools. Also pass the working memory and compressor references.

**Step 1: Write the failing test**

Update the existing test in `sage-python/tests/test_integration.py`:

```python
def test_boot_registers_all_memory_tools():
    """Boot sequence registers all 7 AgeMem memory tools."""
    system = boot_agent_system(use_mock_llm=True)
    tool_names = system.tool_registry.list_tools()
    for expected in [
        "retrieve_context", "summarize_context", "filter_context",
        "search_memory", "store_memory", "update_memory", "delete_memory",
    ]:
        assert expected in tool_names, f"Missing tool: {expected}"
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_integration.py::test_boot_registers_all_memory_tools -v`
Expected: FAIL (only `search_memory` is registered)

**Step 3: Update boot.py**

Replace the imports:

```python
# Old:
from sage.tools.memory_tools import create_search_memory_tool
# New:
from sage.tools.memory_tools import create_memory_tools
```

Replace the episodic memory + search tool section:

```python
    # Old:
    episodic_memory = EpisodicMemory()
    search_tool = create_search_memory_tool(episodic_memory)
    tool_registry.register(search_tool)

    # New:
    episodic_memory = EpisodicMemory()
```

After the loop is created (after `loop.sandbox_manager = sandbox_manager`), register the memory tools that need the loop's working memory:

```python
    # AgeMem: 7 memory tools (3 STM + 4 LTM)
    for tool in create_memory_tools(loop.working_memory, episodic_memory, memory_compressor):
        tool_registry.register(tool)
```

Also update `test_boot_registers_meta_tools` in test_integration.py to check for `store_memory` instead of just `search_memory`:

```python
def test_boot_registers_meta_tools():
    """Boot sequence registers create_python_tool, create_bash_tool, and memory tools."""
    system = boot_agent_system(use_mock_llm=True)
    tool_names = system.tool_registry.list_tools()
    assert "create_python_tool" in tool_names
    assert "create_bash_tool" in tool_names
    assert "search_memory" in tool_names
    assert "store_memory" in tool_names
```

**Step 4: Run all tests**

Run: `cd sage-python && python -m pytest tests/ -x -q`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add sage-python/src/sage/boot.py sage-python/tests/test_integration.py
git commit -m "feat(boot): wire all 7 AgeMem memory tools (3 STM + 4 LTM)"
```

---

### Task 5: Upgrade S2 validation to AVR loop (Act-Verify-Refine)

**Files:**
- Modify: `sage-python/src/sage/agent_loop.py`
- Test: `sage-python/tests/test_metacognition.py`

**Context:** The current S2 validation does a single sandbox check and either passes or retries with a generic error message. The AVR (Act-Verify-Refine) pattern from NotebookLM MetaScaffold says: S2 should operate as an empirical scientist — execute code, capture stderr, feed it back as structured refinement prompt, repeat up to a budget. CGRS self-braking monitors the loop budget before escalating to S3. The existing `_prm_retries` counter and `S2_MAX_RETRIES_BEFORE_ESCALATION` constant provide the budget mechanism.

The key changes:
1. Each AVR iteration captures stdout+stderr and feeds structured refinement context
2. Successful execution captures stdout as verification evidence
3. The refinement prompt includes the iteration count and budget remaining
4. Emit AVR-specific events for dashboard visibility

**Step 1: Write the failing test**

Add to `sage-python/tests/test_metacognition.py`:

```python
def test_s2_avr_loop_constants():
    """S2 AVR loop has configurable constants."""
    from sage.agent_loop import S2_MAX_RETRIES_BEFORE_ESCALATION, S2_AVR_MAX_ITERATIONS
    assert S2_MAX_RETRIES_BEFORE_ESCALATION == 2
    assert S2_AVR_MAX_ITERATIONS == 3
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_metacognition.py::test_s2_avr_loop_constants -v`
Expected: FAIL (`S2_AVR_MAX_ITERATIONS` doesn't exist)

**Step 3: Add AVR constant and upgrade S2 validation in agent_loop.py**

Add constant at module level (near `S2_MAX_RETRIES_BEFORE_ESCALATION`):

```python
S2_AVR_MAX_ITERATIONS = 3  # Max Act-Verify-Refine iterations per code block
```

Replace the S2 sandbox validation block (lines ~250-302) with the AVR loop:

```python
            # System 2 validation (Empirical — AVR: Act-Verify-Refine)
            elif self.config.validation_level == 2 and content:
                code_blocks = _extract_code_blocks(content)

                if code_blocks and self.sandbox_manager:
                    # AVR loop: execute, verify, refine
                    sandbox = await self.sandbox_manager.create()
                    try:
                        result = await sandbox.execute(
                            f"python3 -c {_shell_quote(code_blocks[0])}"
                        )
                        if result.exit_code != 0:
                            self._prm_retries += 1
                            budget_left = self._max_prm_retries - self._prm_retries + 1
                            self._emit(LoopPhase.THINK,
                                       validation="s2_avr_fail",
                                       avr_iteration=self._prm_retries,
                                       avr_budget_left=budget_left,
                                       stderr=result.stderr[:200])
                            if self._prm_retries <= self._max_prm_retries:
                                log.info("S2 AVR fail (iteration %d/%d), refining.",
                                         self._prm_retries, self._max_prm_retries)
                                messages.append(Message(
                                    role=Role.USER,
                                    content=(
                                        f"SYSTEM [AVR iteration {self._prm_retries}/{self._max_prm_retries}]: "
                                        f"Code execution failed (exit code {result.exit_code}).\n"
                                        f"stderr:\n{result.stderr[:500]}\n\n"
                                        f"Refine your code to fix this error. "
                                        f"You have {budget_left} attempt(s) remaining before escalation to formal verification."
                                    ),
                                ))
                                continue
                        else:
                            self._emit(LoopPhase.THINK,
                                       validation="s2_avr_pass",
                                       stdout=result.stdout[:200])
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

                # S2 -> S3 escalation if max retries exhausted
                if self._prm_retries > self._max_prm_retries and self.config.validation_level == 2:
                    log.info("S2 AVR exhausted — escalating to S3 (formal verification).")
                    self.config.validation_level = 3
                    self._prm_retries = 0
                    self._emit(LoopPhase.THINK, escalation="s2_to_s3",
                               reason="AVR budget exhausted")
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
git commit -m "feat(s2): upgrade to AVR (Act-Verify-Refine) loop with budget monitoring"
```

---

## Summary

| Task | What | Research Source | Files |
|------|------|----------------|-------|
| 1 | MEM1 per-step Internal State | MEM1 paper (arxiv 2506.15841) | compressor.py, agent_loop.py |
| 2 | EpisodicMemory CRUD (update/delete) | AgeMem paper (arxiv 2601.01885) | episodic.py |
| 3 | AgeMem 7-tool memory interface | AgeMem + NotebookLM Technical | memory_tools.py |
| 4 | Wire all memory tools in boot.py | Integration | boot.py |
| 5 | S2 AVR loop with budget monitoring | NotebookLM MetaScaffold + MEM1 | agent_loop.py |
