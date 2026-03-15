# Honest Foundation — P0→P4 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix documentation lies, resolve boot timeout, prove claims with real benchmarks, ship honest SDK.

**Architecture:** 5 priority tiers executed sequentially. P0 fixes false documentation. P1 resolves boot discovery (cache + revert unnecessary SSL bypass). P2 runs post-pipeline benchmarks. P3 proves multi-model. P4 ships SDK with honest numbers.

**Tech Stack:** Python 3.12+, pytest, google-genai, openai, bigcodebench, sage_core (Rust).

**Key correction:** Machine is NOT behind a corporate proxy. All `verify=False` hacks are unnecessary and must be reverted or made conditional via env var.

---

## Chunk 1: P0 — Fix Documentation Lies

### Task 1: Fix evolution proof contradiction in CLAUDE.md

**Files:**
- Modify: `CLAUDE.md:406`
- Reference: `sage-python/data/evolution_statistical_proof.json`

The JSON (canonical source) shows: full=99%, no_evo=100%, delta=-1pp, Cohen d=-0.3162, verdict "NO EFFECT DETECTED". CLAUDE.md line 406 says "-10pp (88% vs 98%), Cohen d=-1.41". These are irreconcilable.

- [ ] **Step 1: Read the JSON to confirm exact numbers**

Run: `cd sage-python && python -c "import json; d=json.load(open('data/evolution_statistical_proof.json')); print(f'full={d[\"full_mean_pass_rate\"]}, no_evo={d[\"no_evo_mean_pass_rate\"]}, delta={d[\"delta_pp\"]}pp, d={d[\"cohens_d\"]:.4f}, verdict={d[\"verdict\"]}')"`

- [ ] **Step 2: Fix CLAUDE.md line 406**

Replace:
```
| **Evolution proof (5 runs x 10)** | **-10pp** (88% vs 98%) | NEGATIVE: evo hurts on budget model. Cohen d=-1.41 |
```
With:
```
| **Evolution proof (10 runs x 10)** | **-1pp** (99% vs 100%) | INCONCLUSIVE: no significant effect. Cohen d=-0.32, Wilcoxon p=1.0 |
```

- [ ] **Step 3: Also fix the "SSL / Corporate Proxy" section**

The machine is NOT behind a corporate proxy. Replace the SSL section in CLAUDE.md with:

```markdown
### SSL / Network Configuration
Standard HTTPS — no proxy. If running behind a corporate proxy, set:
```bash
export REQUESTS_CA_BUNDLE=""             # Disable cert verification
export SAGE_SSL_VERIFY=false             # SAGE-specific SSL bypass
```
The `verify=False` pattern in provider code is controlled by `SAGE_SSL_VERIFY` env var.
```

- [ ] **Step 4: Mark pre-pipeline benchmarks honestly**

Add a note after the benchmark results table:
```markdown
> **Note:** HumanEval+ 84.1% and MBPP+ 75.1% were measured on 2026-03-10 with routing layer only (pre-pipeline). The full 5-stage pipeline (merged 2026-03-14) has not been benchmarked yet.
```

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: fix evolution proof (-1pp not -10pp), remove false proxy claim, date benchmarks"
```

---

### Task 2: Run and save kNN routing benchmark

**Files:**
- Modify: `sage-python/src/sage/bench/routing_ground_truth.py`
- Reference: `sage-python/config/routing_ground_truth.json`

The kNN router claims 92% on 50 GT tasks but NO result file exists. The benchmark function returns data but doesn't save it.

- [ ] **Step 1: Run kNN routing GT benchmark and capture output**

Run: `cd sage-python && python -c "
import json
from sage.strategy.knn_router import KnnRouter
from sage.strategy.metacognition import ComplexityRouter
from sage.bench.routing_ground_truth import run_routing_gt, GT_PATH

# kNN router
knn = KnnRouter()
if knn.is_ready:
    result_knn = run_routing_gt(knn, verbose=True)
    print(f'kNN: {result_knn.correct}/{result_knn.total} = {result_knn.accuracy:.1%}')
else:
    print('kNN not ready')

# Heuristic baseline
heuristic = ComplexityRouter()
result_h = run_routing_gt(heuristic, verbose=True)
print(f'Heuristic: {result_h.correct}/{result_h.total} = {result_h.accuracy:.1%}')
"`

- [ ] **Step 2: Save results to JSON**

Add result-saving to the routing_ground_truth.py or run a script that saves:

```python
import json, dataclasses
from pathlib import Path
from datetime import datetime, timezone

results = {
    "benchmark": "routing_ground_truth",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "routers": {
        "knn": dataclasses.asdict(result_knn) if knn.is_ready else None,
        "heuristic": dataclasses.asdict(result_h),
    }
}
out = Path("docs/benchmarks") / f"{datetime.now().strftime('%Y-%m-%d')}-routing-gt.json"
out.write_text(json.dumps(results, indent=2, default=str))
```

- [ ] **Step 3: Commit**

```bash
git add docs/benchmarks/
git commit -m "bench: save kNN and heuristic routing GT results (50 tasks)"
```

---

## Chunk 2: P1 — Fix Boot Discovery

### Task 3: Add discovery cache with 24h TTL

**Files:**
- Modify: `sage-python/src/sage/providers/connector.py`
- Create: `sage-python/tests/test_discovery_cache.py`

- [ ] **Step 1: Write the test**

Create `sage-python/tests/test_discovery_cache.py`:

```python
"""Tests for provider discovery caching."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest


class TestDiscoveryCache:
    def test_write_and_read_cache(self, tmp_path):
        from sage.providers.connector import _write_cache, _read_cache

        models = [{"id": "gemini-2.5-flash", "provider": "google"}]
        _write_cache("google", models, cache_dir=tmp_path)

        cached = _read_cache("google", cache_dir=tmp_path)
        assert cached is not None
        assert len(cached) == 1
        assert cached[0]["id"] == "gemini-2.5-flash"

    def test_cache_expires_after_ttl(self, tmp_path):
        from sage.providers.connector import _write_cache, _read_cache

        models = [{"id": "test-model", "provider": "test"}]
        _write_cache("test", models, cache_dir=tmp_path)

        # Manually expire the cache
        cache_file = tmp_path / "test_models.json"
        data = json.loads(cache_file.read_text())
        data["timestamp"] = time.time() - 90000  # 25 hours ago
        cache_file.write_text(json.dumps(data))

        cached = _read_cache("test", cache_dir=tmp_path)
        assert cached is None  # Expired

    def test_missing_cache_returns_none(self, tmp_path):
        from sage.providers.connector import _read_cache

        cached = _read_cache("nonexistent", cache_dir=tmp_path)
        assert cached is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_discovery_cache.py -v`
Expected: FAIL — `_write_cache` and `_read_cache` don't exist.

- [ ] **Step 3: Implement cache functions in connector.py**

Add after line 14 (after `logger = logging.getLogger(__name__)`):

```python
import json as _json
import time as _time

_CACHE_DIR = Path.home() / ".sage" / "discovery_cache"
_CACHE_TTL_SECONDS = 24 * 3600  # 24 hours


def _cache_path(provider: str, cache_dir: Path | None = None) -> Path:
    return (cache_dir or _CACHE_DIR) / f"{provider}_models.json"


def _read_cache(
    provider: str, cache_dir: Path | None = None, ttl: int = _CACHE_TTL_SECONDS
) -> list[dict[str, Any]] | None:
    path = _cache_path(provider, cache_dir)
    if not path.exists():
        return None
    try:
        data = _json.loads(path.read_text(encoding="utf-8"))
        if _time.time() - data.get("timestamp", 0) > ttl:
            return None
        return data.get("models", [])
    except (ValueError, OSError):
        return None


def _write_cache(
    provider: str, models: list[dict[str, Any]], cache_dir: Path | None = None
) -> None:
    d = cache_dir or _CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": _time.time(), "models": models}
    _cache_path(provider, d).write_text(
        _json.dumps(payload, indent=2), encoding="utf-8"
    )
```

- [ ] **Step 4: Wire cache into _discover_google()**

In `_discover_google()`, after the method signature, add cache check:

```python
    async def _discover_google(self, api_key: str) -> list[DiscoveredModel]:
        # Check disk cache first
        cached = _read_cache("google")
        if cached is not None:
            logger.info("Using cached Google model list (%d models)", len(cached))
            return [DiscoveredModel(**m) for m in cached]

        # ... existing discovery code ...

        # After building models list, cache it
        _write_cache("google", [
            {"id": m.id, "provider": m.provider,
             "context_window": m.context_window,
             "max_output_tokens": m.max_output_tokens,
             "supports_thinking": m.supports_thinking}
            for m in models
        ])
        return models
```

- [ ] **Step 5: Wire cache into _discover_openai_compat()**

Same pattern — cache check at start, write after successful discovery:

```python
    def _discover_openai_compat(self, cfg, api_key):
        provider_name = cfg["provider"]
        cached = _read_cache(provider_name)
        if cached is not None:
            logger.info("Using cached %s model list (%d models)", provider_name, len(cached))
            return [DiscoveredModel(**m) for m in cached]

        # ... existing code ...

        _write_cache(provider_name, [
            {"id": m.id, "provider": m.provider} for m in models
        ])
        return models
```

- [ ] **Step 6: Run tests**

Run: `cd sage-python && python -m pytest tests/test_discovery_cache.py -v`
Expected: 3/3 PASS.

- [ ] **Step 7: Commit**

```bash
git add sage-python/src/sage/providers/connector.py sage-python/tests/test_discovery_cache.py
git commit -m "feat: add 24h TTL disk cache for provider discovery (~/.sage/discovery_cache/)"
```

---

### Task 4: Make SSL bypass conditional via env var

**Files:**
- Modify: `sage-python/src/sage/providers/connector.py`
- Modify: `sage-python/src/sage/providers/openai_compat.py`
- Modify: `sage-python/src/sage/boot.py`

The `verify=False` hacks added today are unnecessary (no proxy). Make them conditional.

- [ ] **Step 1: Create SSL helper**

Add to `sage-python/src/sage/llm/_ssl.py` (or check if it exists):

```python
import os

def ssl_verify() -> bool:
    """Return False only if SAGE_SSL_VERIFY=false is explicitly set."""
    return os.environ.get("SAGE_SSL_VERIFY", "true").lower() != "false"
```

- [ ] **Step 2: Fix connector.py — use ssl_verify()**

In `_discover_google()`, replace `httpx.Client(verify=False, timeout=30)` with:
```python
from sage.llm._ssl import ssl_verify
import httpx
client._api_client._httpx_client = httpx.Client(verify=ssl_verify(), timeout=30)
```

In `_discover_openai_compat()`, replace `httpx.Client(verify=False, timeout=15)` with:
```python
from sage.llm._ssl import ssl_verify
import httpx
http_client = httpx.Client(verify=ssl_verify(), timeout=15)
```

- [ ] **Step 3: Fix openai_compat.py — use ssl_verify()**

Replace `httpx.AsyncClient(verify=False, timeout=60)` with:
```python
from sage.llm._ssl import ssl_verify
client_kwargs["http_client"] = httpx.AsyncClient(verify=ssl_verify(), timeout=60)
```

- [ ] **Step 4: Run full test suite**

Run: `cd sage-python && python -m pytest tests/ -q --ignore=tests/test_a2a_server.py`
Expected: 1444+ passed, 0 failed.

- [ ] **Step 5: Commit**

```bash
git add sage-python/src/sage/llm/_ssl.py sage-python/src/sage/providers/connector.py sage-python/src/sage/providers/openai_compat.py
git commit -m "fix: make SSL bypass conditional via SAGE_SSL_VERIFY env var (default: verify=True)"
```

---

### Task 5: Test boot with discovery cache

**Files:**
- No new files.

- [ ] **Step 1: Clear any existing cache**

Run: `rm -rf ~/.sage/discovery_cache/`

- [ ] **Step 2: Time first boot (cold cache — hits APIs)**

Run: `cd sage-python && time python -c "
from sage.boot import boot_agent_system
from sage.events.bus import EventBus
s = boot_agent_system(use_mock_llm=False, llm_tier='fast', event_bus=EventBus())
pool = getattr(getattr(s, 'pipeline', None), '_provider_pool', None)
if pool:
    print(f'Providers: {len(pool._providers)} — {list(pool._providers.keys())}')
print('Boot OK')
"`
Expected: First boot succeeds (may take 30-60s for API calls). Cache files created in `~/.sage/discovery_cache/`.

- [ ] **Step 3: Time second boot (warm cache — no API calls)**

Run same command again.
Expected: Boot completes in <5s (cached discovery).

- [ ] **Step 4: Verify cache files exist**

Run: `ls -la ~/.sage/discovery_cache/`
Expected: `google_models.json`, and possibly `openai_models.json`, `deepseek_models.json`, etc.

- [ ] **Step 5: Commit cache TTL proof**

No code changes needed. Just verify it works.

---

## Chunk 3: P2 — Prove Pipeline with Real Benchmarks

### Task 6: Run HumanEval+ with current pipeline

**Files:**
- No new code.
- Output: `docs/benchmarks/2026-03-15-evalplus-humaneval-pipeline.json`

- [ ] **Step 1: Run 20-task smoke test**

Run: `cd sage-python && python -m sage.bench --type evalplus --dataset humaneval --limit 20`
Expected: Results printed. Pass rate should be reported.

- [ ] **Step 2: Run full 164-task benchmark**

Run: `cd sage-python && python -m sage.bench --type evalplus --dataset humaneval`
Expected: Full results. Compare against 84.1% (pre-pipeline baseline from March 10).

- [ ] **Step 3: Record delta**

Note the exact numbers. If pipeline helps: great. If pipeline hurts: document honestly.

- [ ] **Step 4: Commit results**

```bash
git add docs/benchmarks/
git commit -m "bench: HumanEval+ with full pipeline (post Phase B+C)"
```

---

### Task 7: Run BigCodeBench hard (first non-saturated benchmark)

**Files:**
- No new code.
- Output: `docs/benchmarks/2026-03-15-bigcodebench-hard-instruct.json`

- [ ] **Step 1: Smoke test (5 tasks)**

Run: `cd sage-python && python -m sage.bench --type bigcodebench --subset hard --split instruct --limit 5`
Expected: Results printed. Verify adapter works end-to-end.

- [ ] **Step 2: Full hard subset (~150 tasks)**

Run: `cd sage-python && python -m sage.bench --type bigcodebench --subset hard --split instruct`
Expected: pass@1 result. This is the first non-saturated benchmark for SAGE.

- [ ] **Step 3: Commit results**

```bash
git add docs/benchmarks/
git commit -m "bench: BigCodeBench hard (150 tasks, first non-saturated benchmark)"
```

---

### Task 8: Run ablation at N=50 (minimum for significance)

**Files:**
- No new code.
- Output: `docs/benchmarks/2026-03-15-ablation-50.json`

- [ ] **Step 1: Run ablation with 50 tasks**

Run: `cd sage-python && python -m sage.bench --type ablation --limit 50`
Expected: 6 configs × 50 tasks. Per-pillar deltas with better statistical power.

- [ ] **Step 2: Document per-pillar contribution**

Note which pillars show measurable delta and which don't.

- [ ] **Step 3: Commit**

```bash
git add docs/benchmarks/
git commit -m "bench: ablation N=50 (6 configs, per-pillar attribution)"
```

---

## Chunk 4: P3 — Prove Multi-Model

### Task 9: Verify per-node provider resolution

**Files:**
- No new code.

- [ ] **Step 1: Boot and check providers**

After Task 5 proves boot works with cache, run:
```python
from sage.boot import boot_agent_system
from sage.events.bus import EventBus
s = boot_agent_system(use_mock_llm=False, llm_tier='fast', event_bus=EventBus())
pool = s.pipeline._provider_pool

# Test resolution for different providers
for model_id in ["gemini-2.5-flash", "deepseek-chat", "grok-3", "kimi-k2.5"]:
    p, c = pool.resolve(model_id)
    print(f"{model_id} → {type(p).__name__} (provider={c.provider})")
```
Expected: Different provider types for different model_ids.

- [ ] **Step 2: Run a real task through pipeline with multi-model**

```python
result = await s.run("Write a Python function to merge two sorted lists")
```
Check logs for which models were actually used per node.

---

### Task 10: BigCodeBench with heterogeneous models (if providers work)

**Files:**
- No new code — uses existing TopologyBench script.

- [ ] **Step 1: Run BigCodeBench hard with default pipeline**

Already done in Task 7.

- [ ] **Step 2: Compare single-model vs pipeline**

Use ablation framework: baseline mode (single model) vs full pipeline (multi-model assignment).

- [ ] **Step 3: Document results honestly**

If multi-model helps: quantify. If it doesn't: document that too.

---

## Chunk 5: P4 — Ship Honest SDK

### Task 11: Update CLAUDE.md with real post-pipeline numbers

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update benchmark table with new results from Tasks 6-8**

Add rows with dates:
```
| **HumanEval+ pipeline (164)** | XX.X% pass@1 | Post Phase B+C, 2026-03-15 |
| **BigCodeBench hard (150)** | XX.X% pass@1 | First non-saturated benchmark |
| **Ablation N=50** | +XXpp full vs baseline | Per-pillar: routing +Xpp, memory +Xpp, ... |
```

- [ ] **Step 2: Update test counts**

Update the test count line with current numbers.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update benchmarks with post-pipeline results (honest numbers)"
```

---

### Task 12: Update memory with final state

**Files:**
- Modify: `~/.claude/projects/C--Code-YGN-SAGE/memory/project_phase_c_complete.md`
- Modify: `~/.claude/projects/C--Code-YGN-SAGE/memory/MEMORY.md`

- [ ] **Step 1: Update memory with honest status and real numbers**
- [ ] **Step 2: Push everything**

```bash
git push origin dev
```

---

## Summary

| Priority | Tasks | Time | Deliverable |
|----------|-------|------|-------------|
| **P0** | 1-2 | 30min | Honest docs, kNN benchmark saved |
| **P1** | 3-5 | 1-2h | Discovery cache, SSL conditional, boot <5s |
| **P2** | 6-8 | 2-4h | HumanEval+ pipeline, BigCodeBench hard, ablation N=50 |
| **P3** | 9-10 | 1-2h | Multi-model verified, heterogeneous comparison |
| **P4** | 11-12 | 30min | CLAUDE.md updated, pushed |

**Total: 12 tasks, ~6-10h.** P0+P1 are blocking. P2 requires API credits. P3 requires P1 working. P4 is documentation.
