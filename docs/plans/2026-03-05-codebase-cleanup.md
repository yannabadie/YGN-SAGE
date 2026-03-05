# Codebase Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Resolve all disconnected/dead/broken code identified in the codebase audit — informed by NotebookLM research (ExoCortex, Technical, Discover AI notebooks all recommend removing dead code).

**Architecture:** Remove dead modules, fix broken files, wire the eBPF evaluator into the evolution pipeline, update exports and dependencies. No new features — pure cleanup.

**Tech Stack:** Python 3.12, pytest, pyproject.toml, git

---

### Task 1: Remove dead LLM modules (anthropic.py, openai.py, registry.py)

**Files:**
- Delete: `sage-python/src/sage/llm/anthropic.py`
- Delete: `sage-python/src/sage/llm/openai.py`
- Delete: `sage-python/src/sage/llm/registry.py`
- Modify: `sage-python/src/sage/llm/__init__.py`

**Step 1: Delete the 3 dead files**

```bash
cd sage-python
rm src/sage/llm/anthropic.py src/sage/llm/openai.py src/sage/llm/registry.py
```

**Step 2: Update `llm/__init__.py` — remove LLMRegistry import**

Replace contents with:
```python
"""LLM abstraction layer with multi-provider support."""
from sage.llm.base import LLMConfig, Message, Role, ToolDef, ToolCall, LLMResponse

__all__ = [
    "LLMConfig",
    "Message",
    "Role",
    "ToolDef",
    "ToolCall",
    "LLMResponse",
]
```

**Step 3: Run tests to verify nothing breaks**

Run: `cd sage-python && python -m pytest tests/ -x -q`
Expected: All pass (these modules were never imported)

**Step 4: Commit**

```bash
git add -A sage-python/src/sage/llm/
git commit -m "refactor(llm): remove dead anthropic, openai providers and unused registry"
```

---

### Task 2: Remove dead memory modules (neo4j_driver.py, qdrant_driver.py, python_backend.py)

**Files:**
- Delete: `sage-python/src/sage/memory/neo4j_driver.py`
- Delete: `sage-python/src/sage/memory/qdrant_driver.py`
- Delete: `sage-python/src/sage/memory/python_backend.py`
- Modify: `sage-python/pyproject.toml` (drop `neo4j`, `qdrant-client` deps)

**Step 1: Delete the 3 dead files**

```bash
cd sage-python
rm src/sage/memory/neo4j_driver.py src/sage/memory/qdrant_driver.py src/sage/memory/python_backend.py
```

**Step 2: Update `pyproject.toml` — remove neo4j and qdrant-client from dependencies**

Change the `dependencies` list from:
```toml
dependencies = [
    "httpx>=0.28",
    "pydantic>=2.10",
    "rich>=13",
    "anyio>=4",
    "neo4j>=5.17",
    "qdrant-client>=1.8",
]
```
To:
```toml
dependencies = [
    "httpx>=0.28",
    "pydantic>=2.10",
    "rich>=13",
    "anyio>=4",
]
```

**Step 3: Run tests**

Run: `cd sage-python && python -m pytest tests/ -x -q`
Expected: All pass

**Step 4: Commit**

```bash
git add sage-python/src/sage/memory/ sage-python/pyproject.toml
git commit -m "refactor(memory): remove dead neo4j, qdrant, python_backend drivers and deps"
```

---

### Task 3: Remove dead evolution module (sandbox_evaluator.py)

**Files:**
- Delete: `sage-python/src/sage/evolution/sandbox_evaluator.py`

**Step 1: Delete the broken file**

```bash
rm sage-python/src/sage/evolution/sandbox_evaluator.py
```

**Step 2: Verify it's not imported anywhere**

```bash
grep -r "sandbox_evaluator" sage-python/src/ sage-python/tests/
```
Expected: No results

**Step 3: Run tests**

Run: `cd sage-python && python -m pytest tests/ -x -q`
Expected: All pass

**Step 4: Commit**

```bash
git add sage-python/src/sage/evolution/
git commit -m "refactor(evolution): remove broken sandbox_evaluator (superseded by ebpf_evaluator)"
```

---

### Task 4: Wire EbpfEvaluator as default evaluator stage in EvolutionEngine

**Files:**
- Modify: `sage-python/src/sage/evolution/engine.py`
- Test: `sage-python/tests/test_evolution.py`

**Step 1: Update EvolutionEngine.__init__ to add EbpfEvaluator as a default stage**

In `engine.py`, after the evaluator is created (`self._evaluator = evaluator or Evaluator()`), add:

```python
# Wire eBPF evaluator as default first stage (sub-ms execution)
try:
    from sage.evolution.ebpf_evaluator import EbpfEvaluator
    ebpf = EbpfEvaluator()
    self._evaluator.add_stage("ebpf_sandbox", ebpf, threshold=0.0, weight=1.0)
    log.info("eBPF evaluator wired as default evolution stage")
except Exception:
    log.debug("eBPF evaluator not available — evolution runs without hardware sandbox")
```

**Step 2: Run tests**

Run: `cd sage-python && python -m pytest tests/test_evolution.py tests/test_evolution_sota.py -x -v`
Expected: All pass (EbpfEvaluator gracefully mocks if sage_core unavailable)

**Step 3: Commit**

```bash
git add sage-python/src/sage/evolution/engine.py
git commit -m "feat(evolution): wire EbpfEvaluator as default stage in EvolutionEngine"
```

---

### Task 5: Fix broken debug files and remove obsolete scripts

**Files:**
- Modify: `debug/run_ygn_sage_agent.py` (remove broken imports, fix enforce_system3)
- Delete: `debug/sync_large_file.py` (broken, depends on nonexistent module)
- Delete: `scripts/list_notebooks.py` (broken, uses unofficial notebooklm lib)

**Step 1: Rewrite `debug/run_ygn_sage_agent.py` — remove broken imports, update to validation_level**

Replace lines 7-19 and line 64 with working code that removes the broken `query_research_nbs` and `notebooklm_agent_sync` imports and the `query_exocortex` tool, and replaces `enforce_system3=True` with `validation_level=3`.

**Step 2: Delete broken files**

```bash
rm debug/sync_large_file.py scripts/list_notebooks.py
```

**Step 3: Commit**

```bash
git add debug/ scripts/
git commit -m "fix: remove broken imports from debug scripts, delete obsolete list_notebooks"
```

---

### Task 6: Archive labs/ to Researches/experimental_hft/

**Files:**
- Move: `labs/*.py` → `Researches/experimental_hft/`

**Step 1: Move HFT experiments to archive**

```bash
mkdir -p Researches/experimental_hft
mv labs/commercial_hft_optimizer.py Researches/experimental_hft/
mv labs/deploy_autonomous_hft.py Researches/experimental_hft/
mv labs/gcp_hft_node.py Researches/experimental_hft/
rmdir labs
```

**Step 2: Commit**

```bash
git add labs/ Researches/experimental_hft/
git commit -m "refactor: archive HFT experiments from labs/ to Researches/experimental_hft/"
```

---

### Task 7: Update CLAUDE.md and audit doc, final test run

**Files:**
- Modify: `CLAUDE.md` (remove neo4j/qdrant from tech stack)
- Modify: `docs/plans/2026-03-05-codebase-audit.md` (mark resolved)

**Step 1: Update CLAUDE.md tech stack section**

Remove lines mentioning Neo4j and Qdrant from the tech stack since they are no longer dependencies.

**Step 2: Run full test suite**

Run: `cd sage-python && python -m pytest tests/ -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add CLAUDE.md docs/plans/
git commit -m "docs: update CLAUDE.md and audit report after codebase cleanup"
```
