---
name: cli_gaps stages A-D shipped 2026-05-07 — sage run --jsonl v0 protocol gaps closed
description: Cycle-13 K post-Phase-2.2 cli_gaps stage chain. 4 NYI gaps closed (final_seq, set_budget, cli_progress, cancel) + Stage E doc closure. cgpro `cgpro_cli_protocol_gaps_20260507`.
type: project
originSessionId: ae15e41f-58ed-438b-8f62-6e3feb79131b
---
Cycle-13 K post-Phase-2.2 cli_gaps stage chain. New cgpro conv `cgpro_cli_protocol_gaps_20260507` (in-project YGN-SAGE, no --resume per Yann directive 2026-05-06). Closes the 4 NYI v0 protocol gaps documented in `cli/run.py:21-29` at cycle-12 prelude.

**Why:** Cycle-12 prelude shipped `sage run --jsonl` as the pi-mono backend with 4 known gaps (`final_seq` not populated, `set_budget` not wired, `cli_progress` not heartbeating, `cancel` not emitting terminal failure frame). Each gap blocks a piece of the cycle-13 SWE-bench Pro 4-arm ablation observability contract. Per cgpro DESIGN E trap Q5 the gaps were locked for a focused stage chain BEFORE cycle-13 main run wiring (so the predictions stream isn't telemetry-blind).

**How to apply:**
- Active cgpro conv `cgpro_cli_protocol_gaps_20260507` is now closed; future cycle-13 main run wiring / cycle-14 façade rewrite Phase C / SWE-bench Pro arms wiring starts a NEW conv (no --resume, auto-routes to YGN-SAGE project).
- 4 stages A-D + closure E. Each stage shipped through pre-commit cgpro VERIFY → SHIP loop. Some stages took 2-3 EDIT_REQUIRED rounds.
- `pipeline.py` UNTOUCHED across all 4 stages (299 LOC, < 300 HARD GATE met since Phase 2.2 Stage F). cgpro hard-stop on touching it. CLI state attached to pipeline at runtime via `setattr(pipeline, "_cli_progress_state", state)` instead of declared class attribute.
- Final test count: 3290 Python (+49 across stages A-D) / 555 Rust / 100 sage-discover.
- mypy 0 / ruff clean / claims_audit OK / narrative_guard PASS at every stage.

**cli_gaps stage commit chain:**
- Stage A `2d557b15` — unify stdout seq + populate cli_complete.final_seq. `_StdoutSeqCounter.last` is the SOLE source of truth for `cli_complete.payload.final_seq`; mirror's `last_stdout_seq` is debug only (cli_tool_request CAN fire after the last mirrored runtime event, mirror tracker would miss it). Forensic file kept byte-identical with its own internal seq domain. 3 cgpro rounds.
- Stage B `7bd48c17` — wire tightening-only set_budget command. `CostTracker.tighten_remaining_budget(new_remaining_usd)` root guard with `BudgetUpdateResult` frozen dataclass. Reason codes: `budget_before_prompt` / `budget_invalid_value` (zero rejected because `budget_usd <= 0` is the unlimited sentinel — cgpro round-2 trap) / `budget_loosen_rejected`. `failure(kind="cli_command")` is non-terminal per Stage B spec amendment. Always-on `ctx.cost_tracker` (even unlimited) so tighten-from-infinity works; P9 byte-identical preserved. 4 cgpro rounds.
- Stage C `2ce3c877` — cli_progress idle heartbeat. Timer-based 5s cadence with 10s idle guard, NOT piggyback. CLI-owned `_CliProgressState` dataclass, two timestamps (`last_non_progress_frame_at` + `last_progress_frame_at`). Load-bearing trap: `cli_progress` does NOT update `last_non_progress_frame_at` (would degrade cadence to 10s). 7 canonical stage labels (boot/classify/decompose/select_topology/assign_models/execute/learn). `_set_cli_progress_stage(pipeline, stage)` helper in `pipeline_v2.orchestrator` called UNCONDITIONALLY before each of 6 stages. 3 cgpro rounds.
- Stage D `d0bfea2b` — cancel hardening + cooperative v0 cancellation lock. Emits exactly one `failure(kind="cli_cancel", error_type="cancelled", message=<reason>)` BEFORE terminal `cli_complete(outcome="cancelled", exit_code=130)`. Idempotent at stream level. `_drive_cancel_run` test helper patches `sage.boot.boot_agent_system` to return a fake System with hanging Pipeline.run (direct asyncio, no subprocess, no real sleeps). 3 cgpro rounds.
- Stage E `b4956396` — protocol doc cleanup (`_stage_learn` / `_stage_select_topology` → `sage.pipeline_v2.<X>.<X>` module-function refs) + status counter sync (3241 → 3290) + final gates closure. cgpro round-2 EDIT_REQUIRED: status header + status changes + cancel snapshot (FLAT shape) + invariant 9 backport promise (factual: ledger has it since cycle-12 `f647c5ae`); round-3 GO_PUSH.

**Lessons applied during the cycle:**
1. **Runtime failure frame shape is FLAT-redacted on stdout, NOT nested under `payload`.** Stage D test discovery: `RuntimeEventLog.emit_failure(kind, error_type, message)` produces a stdout frame with `kind` / `error_type` / `node_id` at TOP LEVEL (cycle-7 R6.1c redacted form). The `message` is hashed into `payload_hash`; full payload preserved in the forensic file under `trace_dir`. CLI envelope events (`cli_started` / `cli_progress` / `cli_tool_request` / `cli_complete`) keep nested `payload` shape. Tests assert `f.get("kind") == "cli_cancel"` (top-level) for runtime failure and `f["payload"]["outcome"]` for CLI envelope. Stage B's spec wording `failure.payload.kind` was a documentation bug closed by Stage D round-2 EDIT_REQUIRED.
2. **Two-timestamp idle clock**: tracking `last_non_progress_frame_at` + `last_progress_frame_at` separately is the canonical pattern for emit-on-idle heartbeats. Single-timestamp would forever reset and degrade cadence.
3. **Pipeline.py LOC discipline post Phase 2.2**: 299 LOC is the working margin. Stage B added `tighten_budget` method + `_active_context` annotation by trimming docstrings (`run_with_frame` 6 → 1 line, `run_with_bench_evaluator` 9 → 2 lines, `_run_internal` 4 → 1 line). Stage C did NOT touch pipeline.py at all (CLI-owned state attached via `setattr`). For future cycles: any new method on `CognitiveOrchestrationPipeline` requires same-commit docstring trim or HARD_STOP.
4. **`_StdoutSeqCounter.last` is canonical for cli_complete.final_seq, NOT mirror tracker**: subtle. cgpro Stage A round-2 caught: cli_tool_request can fire AFTER the last mirrored runtime event (mid-tool-call cancel path), so the mirror tracker is a subset, not the truth. Counter's `last` is the universal source.
5. **Spec doc fixes propagated retroactively**: when Stage D revealed Stage B's `failure.payload.kind` wording was wrong, fix it in Stage D rather than reopening Stage B. cgpro lock: "Stage D should fix the protocol's failure-field shape" — done in same commit.
6. **`tests/runtime/` directory was missing**: Stage B created it (mkdir + new test file). Phase 2.2 didn't have a `tests/runtime/` because the pipeline_v2 tests live at `tests/test_pipeline_v2_*.py`. The new directory is for future runtime/* tests (CostTracker is the seed).

**cgpro browser session stability**: cgpro Chromium crashed mid-session twice today (Phase 2.2 E3 once, cli_gaps Stage B once). Fix each time: user runs `cgpro adopt` from terminal to re-import ChatGPT desktop app session. Same incident pattern as 2026-05-06. Session reports "healthy" after adopt, ask command resumes. Browser timeout `page.goto: Timeout 60000ms exceeded` is transient; retry usually works.

**Open follow-ups (not cli_gaps scope):**
- Stage E commit (TBD this session) — closure with status counter sync.
- Cycle-13 main run wiring — arms A/B/C/D, $240-460 budget, 3-5 days, REQUIRES cli_gaps stages closed (they are now).
- ALIRE3.md pre-existing working-tree deletion still uncommitted.
- Wider `~/.sage/` test-pollution flake class (cycle-11 issue, separate).
- Deeper runtime cancellation (Rust / provider in-flight interrupt) — explicitly v0 limitation, future cycle scope.

**HEAD post-cli_gaps closure:** `b4956396` (Stage E shipped 2026-05-07). cgpro conv `cgpro_cli_protocol_gaps_20260507` closed.
