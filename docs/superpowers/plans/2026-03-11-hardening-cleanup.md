# Hardening & Cleanup — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all P0-P2 issues from the March 11 audit: S-MMU mock degradation logging, dead code removal, shadow gate activation, bandit warm-start + persistence, LTL Python integration, and GraphDB/VectorDB stub cleanup.

**Architecture:** Six independent tasks touching Python SDK and boot wiring. No Rust changes needed (P0-1 sandbox and P0-2 subprocess are already fixed/documented). Each task is self-contained and can be committed independently.

**Tech Stack:** Python 3.12 (sage-python), sage_core PyO3 bindings

---

## File Structure

| File | Change | Responsibility |
|------|--------|---------------|
| `sage-python/src/sage/memory/working.py` | Modify | Add logging to mock methods |
| `sage-python/src/sage/boot.py` | Modify | Shadow gate check, bandit warm_start call |
| `sage-python/src/sage/routing/shadow.py` | Modify | Add load_existing_traces() for cross-session gate |
| `sage-python/src/sage/topology/ltl_bridge.py` | Create | Python helper wrapping Rust LtlVerifier |
| `sage-python/src/sage/evolution/ebpf_evaluator.py` | Delete | Dead code (eBPF abandoned) |
| `sage-python/src/sage/strategy/allocator.py` | Delete | Dead code (never imported) |
| `sage-python/src/sage/strategy/training.py` | Delete | Dead code (never imported) |
| `sage-python/src/sage/memory/compressor.py` | Modify | Remove GraphDB/VectorDB stub protocols |
| `sage-python/tests/test_working_memory_mock.py` | Create | Test mock logging |
| `sage-python/tests/test_ltl_bridge.py` | Create | Test LTL Python integration |

---

### Task 1: S-MMU Mock Degradation Logging (P0-3)

**Problem:** When `sage_core` is unavailable, mock methods silently return empty values. Boot.py line 532 has a warning, but individual mock method calls log nothing.

**Files:**
- Modify: `sage-python/src/sage/memory/working.py:37-78`
- Create: `sage-python/tests/test_working_memory_mock.py`

- [ ] **Step 1: Write the failing test**

```python
# sage-python/tests/test_working_memory_mock.py
"""Test that WorkingMemory mock logs warnings on S-MMU operations."""
import logging

def test_mock_smmu_operations_log_warnings(caplog):
    """Mock S-MMU methods should log warnings, not silently return empty."""
    # Force mock mode by patching
    import sage.memory.working as wm
    original = wm._has_rust
    wm._has_rust = False

    # Re-import to get mock class
    mock_wm = wm._PyWorkingMemory("test-agent")

    with caplog.at_level(logging.WARNING, logger="sage.memory.working"):
        mock_wm.compact_to_arrow_with_meta(["kw1"], None, None, "summary")
        mock_wm.retrieve_relevant_chunks(0, 2)
        mock_wm.get_chunk_summary(0)

    wm._has_rust = original

    # Each S-MMU operation should have logged a warning
    assert len(caplog.records) >= 3
    assert any("compact_to_arrow_with_meta" in r.message for r in caplog.records)
    assert any("retrieve_relevant_chunks" in r.message for r in caplog.records)
    assert any("get_chunk_summary" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd sage-python && python -m pytest tests/test_working_memory_mock.py -v --tb=short
```
Expected: FAIL (mock methods currently don't log)

- [ ] **Step 3: Add logging to mock S-MMU methods**

In `sage-python/src/sage/memory/working.py`, add a logger to `_PyWorkingMemory` and log warnings on S-MMU methods:

```python
# Add at top of _PyWorkingMemory class (line 45):
class _PyWorkingMemory:
    _mock_logger = logging.getLogger("sage.memory.working")
    _mock_warned = set()  # Only warn once per method to avoid spam

    def _warn_once(self, method: str) -> None:
        if method not in self._mock_warned:
            self._mock_logger.warning(
                "S-MMU mock: %s() returns dummy value — sage_core not compiled. "
                "Build with: cd sage-core && maturin develop",
                method,
            )
            self._mock_warned.add(method)

    # ... existing __init__ ...

    def compact_to_arrow(self):
        self._warn_once("compact_to_arrow")
        return 0

    def compact_to_arrow_with_meta(self, kw, emb=None, parent=None, summary=None):
        self._warn_once("compact_to_arrow_with_meta")
        return 0

    def retrieve_relevant_chunks(self, cid, hops, w=None):
        self._warn_once("retrieve_relevant_chunks")
        return []

    def get_page_out_candidates(self, cid, hops, budget):
        self._warn_once("get_page_out_candidates")
        return []

    def smmu_chunk_count(self):
        return 0  # No warning needed — just a counter

    def get_chunk_summary(self, chunk_id: int) -> str:
        self._warn_once("get_chunk_summary")
        return ""
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd sage-python && python -m pytest tests/test_working_memory_mock.py -v --tb=short
```
Expected: PASS

- [ ] **Step 5: Run full test suite**

```bash
cd sage-python && python -m pytest tests/ -v --tb=short -x -q
```
Expected: All existing tests still pass

- [ ] **Step 6: Commit**

```bash
git add sage-python/src/sage/memory/working.py sage-python/tests/test_working_memory_mock.py
git commit -m "fix(memory): log warnings on S-MMU mock method calls instead of silent empty returns"
```

---

### Task 2: Dead Code Removal (P2-11)

**Problem:** 3 modules are never imported: `ebpf_evaluator.py` (79 LOC), `allocator.py` (138 LOC), `training.py` (69 LOC). Total 286 LOC of dead code.

**Files:**
- Delete: `sage-python/src/sage/evolution/ebpf_evaluator.py`
- Delete: `sage-python/src/sage/strategy/allocator.py`
- Delete: `sage-python/src/sage/strategy/training.py`

- [ ] **Step 1: Verify no imports exist**

```bash
cd sage-python && grep -r "ebpf_evaluator\|EbpfEvaluator" src/ tests/ --include="*.py" -l
cd sage-python && grep -r "from sage.strategy.allocator\|from sage.strategy import allocator" src/ tests/ --include="*.py" -l
cd sage-python && grep -r "from sage.strategy.training\|from sage.strategy import training" src/ tests/ --include="*.py" -l
```
Expected: No matches (or only self-references)

- [ ] **Step 2: Delete the files**

```bash
rm sage-python/src/sage/evolution/ebpf_evaluator.py
rm sage-python/src/sage/strategy/allocator.py
rm sage-python/src/sage/strategy/training.py
```

- [ ] **Step 3: Run full test suite**

```bash
cd sage-python && python -m pytest tests/ -v --tb=short -x -q
```
Expected: All tests pass (these modules were never imported)

- [ ] **Step 4: Commit**

```bash
git add -u sage-python/src/sage/evolution/ebpf_evaluator.py sage-python/src/sage/strategy/allocator.py sage-python/src/sage/strategy/training.py
git commit -m "chore: remove dead code — ebpf_evaluator, allocator, training (286 LOC, 0 imports)"
```

---

### Task 3: Remove GraphDB/VectorDB Stubs (P2-10, YAGNI)

**Problem:** `compressor.py` defines `GraphDatabase` and `VectorDatabase` Protocol classes that are never implemented or called. Constructor accepts `graph_db`/`vector_db` args that are always None.

**Files:**
- Modify: `sage-python/src/sage/memory/compressor.py`

- [ ] **Step 1: Read the current file**

Read `sage-python/src/sage/memory/compressor.py` to find the stub protocols and constructor args.

- [ ] **Step 2: Remove stub protocols and unused constructor args**

Remove:
- `GraphDatabase` Protocol class definition
- `VectorDatabase` Protocol class definition
- `graph_db` and `vector_db` constructor parameters (keep default `None` if they have callers)
- Any imports only used by these stubs

- [ ] **Step 3: Run full test suite**

```bash
cd sage-python && python -m pytest tests/ -v --tb=short -x -q
```
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add sage-python/src/sage/memory/compressor.py
git commit -m "chore(memory): remove GraphDB/VectorDB stub protocols (YAGNI — zero implementations)"
```

---

### Task 4: Shadow Gate Activation (P1-6)

**Problem:** `ShadowRouter.is_phase5_ready()` exists but is never checked. Shadow traces accumulate to `~/.sage/shadow_traces.jsonl` but the gate never fires. Need: (1) load existing traces at boot for cross-session continuity, (2) log gate status at boot.

**Files:**
- Modify: `sage-python/src/sage/routing/shadow.py`
- Modify: `sage-python/src/sage/boot.py`

- [ ] **Step 1: Add `load_existing_traces()` to ShadowRouter**

In `sage-python/src/sage/routing/shadow.py`, add method to load trace counts from existing JSONL file:

```python
def load_existing_traces(self) -> None:
    """Load trace counts from existing JSONL file for cross-session gate continuity."""
    if not self._trace_path.exists():
        return
    try:
        total = 0
        mismatches = 0
        with open(self._trace_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                total += 1
                try:
                    record = json.loads(line)
                    if not record.get("match", True):
                        mismatches += 1
                except json.JSONDecodeError:
                    continue
        self.stats["total_comparisons"] = total
        self.stats["system_mismatches"] = mismatches
        _log.info(
            "Shadow: loaded %d existing traces (%.1f%% divergence)",
            total, self.divergence_rate() * 100,
        )
    except Exception as exc:
        _log.warning("Shadow: failed to load existing traces (%s)", exc)
```

- [ ] **Step 2: Wire gate check into boot.py**

In `sage-python/src/sage/boot.py`, after ShadowRouter creation, load traces and log gate status:

```python
# After shadow_router construction:
if shadow_router._shadow_active:
    shadow_router.load_existing_traces()
    if shadow_router.is_phase5_hard_ready():
        _log.info(
            "Shadow Phase 5 HARD gate passed (%d traces, %.1f%% divergence) — "
            "Python router can be safely removed",
            shadow_router.total, shadow_router.divergence_rate() * 100,
        )
    elif shadow_router.is_phase5_soft_ready():
        _log.info(
            "Shadow Phase 5 SOFT gate passed (%d traces, %.1f%% divergence) — "
            "Rust router preferred",
            shadow_router.total, shadow_router.divergence_rate() * 100,
        )
    else:
        _log.info(
            "Shadow Phase 5: %d/%d traces collected (need 1000, divergence=%.1f%%)",
            shadow_router.total, 1000, shadow_router.divergence_rate() * 100,
        )
```

- [ ] **Step 3: Run tests**

```bash
cd sage-python && python -m pytest tests/ -v --tb=short -x -q -k "shadow or boot"
```
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add sage-python/src/sage/routing/shadow.py sage-python/src/sage/boot.py
git commit -m "feat(routing): activate shadow Phase 5 gate — load cross-session traces, log gate status at boot"
```

---

### Task 5: Bandit Warm-Start from ModelCards (P1-7)

**Problem:** `ContextualBandit.warm_start()` API exists in Rust but is never called from Python. Bandit starts with uniform Beta(1,1) every session — no learning carried over. The `warm_start()` method seeds quality priors from card affinity scores.

**Files:**
- Modify: `sage-python/src/sage/boot.py:490-504`

- [ ] **Step 1: Add warm_start call after bandit creation**

In `sage-python/src/sage/boot.py`, after line 496 (`rust_router.set_bandit(rust_bandit)`), add warm-start from cards:

```python
# After set_bandit, seed bandit arms from ModelCards:
if rust_registry and rust_bandit:
    try:
        cards = rust_registry.all_models()
        templates = ["sequential", "avr", "parallel", "debate"]
        model_ids = [c.id for c in cards]
        affinities = [max(c.s1_affinity, c.s2_affinity, c.s3_affinity) for c in cards]
        rust_bandit.warm_start(model_ids, templates, affinities)
        _log.info(
            "Boot: Bandit warm-started with %d models x %d templates (%d arms)",
            len(model_ids), len(templates), len(model_ids) * len(templates),
        )
    except Exception as e:
        _log.debug("Boot: Bandit warm-start failed (%s)", e)
```

- [ ] **Step 2: Run tests**

```bash
cd sage-python && python -m pytest tests/ -v --tb=short -x -q -k "boot or bandit"
```
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add sage-python/src/sage/boot.py
git commit -m "feat(routing): warm-start bandit from ModelCard affinities at boot"
```

---

### Task 6: LTL Python Bridge (P2-9)

**Problem:** `LtlVerifier` and `LtlResult` are already exported from sage_core (lib.rs:66-67) but never called from Python. Need a thin bridge to verify topologies.

**Files:**
- Create: `sage-python/src/sage/topology/ltl_bridge.py`
- Create: `sage-python/tests/test_ltl_bridge.py`

- [ ] **Step 1: Write the failing test**

```python
# sage-python/tests/test_ltl_bridge.py
"""Test LTL temporal verification bridge."""
import pytest

def test_ltl_bridge_import():
    """LTL bridge module should be importable."""
    from sage.topology.ltl_bridge import verify_topology_ltl
    assert callable(verify_topology_ltl)

def test_ltl_bridge_returns_dict():
    """verify_topology_ltl should return a result dict even without sage_core."""
    from sage.topology.ltl_bridge import verify_topology_ltl
    result = verify_topology_ltl(None)  # None topology = skip
    assert isinstance(result, dict)
    assert "reachable" in result
    assert "safe" in result
    assert "live" in result
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd sage-python && python -m pytest tests/test_ltl_bridge.py -v --tb=short
```
Expected: FAIL (module doesn't exist)

- [ ] **Step 3: Create the bridge module**

```python
# sage-python/src/sage/topology/ltl_bridge.py
"""LTL temporal property verification bridge.

Wraps Rust LtlVerifier (always compiled in sage_core) to check
topology properties: reachability, safety, liveness, bounded liveness.
"""
from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)

try:
    from sage_core import LtlVerifier, LtlResult
    _HAS_LTL = True
except ImportError:
    _HAS_LTL = False


def verify_topology_ltl(
    topology: Any | None,
    max_depth: int = 20,
) -> dict[str, Any]:
    """Run LTL checks on a TopologyGraph.

    Returns a dict with keys: reachable, safe, live, bounded_live, warnings, errors.
    Returns all-True defaults if topology is None or LTL unavailable.
    """
    default = {
        "reachable": True,
        "safe": True,
        "live": True,
        "bounded_live": True,
        "warnings": [],
        "errors": [],
    }

    if topology is None:
        return default

    if not _HAS_LTL:
        _log.debug("LtlVerifier not available (sage_core not compiled)")
        return default

    try:
        verifier = LtlVerifier()
        result = verifier.verify(topology, max_depth)
        return {
            "reachable": result.reachable,
            "safe": result.safe,
            "live": result.live,
            "bounded_live": result.bounded_live,
            "warnings": result.warnings if hasattr(result, "warnings") else [],
            "errors": result.errors if hasattr(result, "errors") else [],
        }
    except Exception as exc:
        _log.warning("LTL verification failed (%s)", exc)
        default["warnings"].append(f"LTL check failed: {exc}")
        return default
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd sage-python && python -m pytest tests/test_ltl_bridge.py -v --tb=short
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sage-python/src/sage/topology/ltl_bridge.py sage-python/tests/test_ltl_bridge.py
git commit -m "feat(topology): add LTL Python bridge wrapping Rust LtlVerifier for topology temporal checks"
```

---

## Summary

| Task | Issue | LOC Changed | Risk |
|------|-------|-------------|------|
| 1. S-MMU Mock Logging | P0-3 | ~30 modify + ~30 test | Low |
| 2. Dead Code Removal | P2-11 | -286 delete | None |
| 3. GraphDB/VectorDB Stubs | P2-10 | ~-30 delete | None |
| 4. Shadow Gate Activation | P1-6 | ~40 add | Low |
| 5. Bandit Warm-Start | P1-7 | ~15 add | Low |
| 6. LTL Python Bridge | P2-9 | ~60 create + ~20 test | Low |

**Total:** ~-150 LOC net (removing dead code while adding new functionality).
