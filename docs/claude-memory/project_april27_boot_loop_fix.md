---
name: April 27 boot pipeline + test pollution fix
description: Two pre-existing issues fixed — swebench-induced asyncio policy pollution (18 NotImplementedError) + asyncio.run() loop loss in boot_pipeline (3 grpc errors)
type: project
originSessionId: bf130342-a1fc-4ba3-9819-62c0d87a6b87
---
**Date:** 2026-04-27. Commits `0b9c2464` (fix) + `20bb93b1` (constraints path normalize).

**Two pre-existing issues** that surfaced when running the full pytest suite with `-k "not test_e2e and not test_pydantic_ai_integration and not test_live_multiprovider and not test_swebench"`. Both pre-dated this session — they had been hidden by either previous infra bugs (collection-time ImportError before maturin recipe fix) or by run-in-isolation test patterns. Neither is "new bug introduced by recent work."

**Issue 1: Test pollution (18 NotImplementedError)**
- Symptom: 18 tests in `test_execution.py`/`test_sandbox*.py` fail with `NotImplementedError` only when run as part of full suite, pass in isolation.
- Root cause: `pytest.importorskip("swebench")` at top of `test_swebench_ca_patch.py` triggers swebench's package init, which calls `asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())` on Windows. SelectorEventLoop has no subprocess support, so every later `asyncio.create_subprocess_exec()` raises `NotImplementedError`. pytest collection imports every test module BEFORE any filter runs, so `-k "not test_swebench"` does not prevent the swebench import.
- Fix: 3-layer defensive Proactor restore in `sage-python/tests/conftest.py`:
  1. `pytest_configure` — before collection
  2. `pytest_collection_modifyitems` — after collection (covers swebench)
  3. `_restore_windows_proactor_policy` autouse fixture — before each test (belt + braces)
- Win32-only; Linux/macOS unaffected.

**Issue 2: Boot pipeline loop loss (3 grpc.aio errors)**
- Symptom: `RuntimeError: There is no current event loop in thread 'MainThread'` deep inside `boot_agent_system()` → xAI/Google grpc.aio.Channel construction.
- Root cause: `boot_pipeline._discover_models()` calls `asyncio.run(...)`. `asyncio.run()`'s cleanup calls `events.set_event_loop(None)`. Subsequent provider construction calls `asyncio.get_event_loop()` and finds `_set_called=True, _loop=None`, raising on Python 3.12+.
- Fix: snapshot the caller's loop BEFORE asyncio.run(), restore it AFTER. If no prior loop, set a fresh one. In `sage-python/src/sage/boot_pipeline.py:36-62`.
- Why this didn't surface before: `boot_agent_system()` is rarely called from sync code with no pre-existing loop. Tests that hit this path either skip on missing GOOGLE_API_KEY or run inside pytest-asyncio's loop. The new test_provider_pool_wiring class is the first sync fixture that boots the system on Python 3.13.

**Why:** the user explicitly tasked "résoudre définitivement... toute les Pending" — the 18+3 was the last Pending item from cgpro's post-cycle playbook ("Test pollution / order-dependence — pre-existing not A14"). Both issues were correctly diagnosed by cgpro as pre-existing and required structural fixes, not skips.

**How to apply:**
- Don't try to set loop manually in test fixtures — fix the producer (boot_pipeline) instead. Fixture-level workarounds got tangled with pytest-asyncio's auto-mode loop management.
- For asyncio test pollution on Windows, the `pytest_configure + pytest_collection_modifyitems + autouse fixture` pattern is the canonical defense. Single-layer protection isn't enough because pollution can happen at multiple stages.
- After the fix: 2350 passed, 36 skipped, 0 failures (on `-k "not test_e2e and not test_pydantic_ai_integration and not test_live_multiprovider and not test_swebench"`). mypy 0 errors / 184 files. ruff clean. type:ignore ceiling held at 45.

**A27-followup status — DONE** (already shipped as commit `20bb93b1`): the absolute WSL path `/mnt/c/Code/YGN-SAGE/sage-python/constraints.txt` in compile_python_constraints.sh was switched to relative `../sage-python/constraints.txt` so the comment-trail in `sage-discover/constraints.txt` is portable across machines.
