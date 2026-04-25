---
name: April 20 (session 1) — Rust-First plan 1.1 done (max_steps singleton)
description: Plan item 1.1 landed (commit b7ced9d) — singleton max_steps now scales 5/10/20 per system tier on the bypass path, closing the H7 singleton asymmetry
type: project
originSessionId: e6496ce0-f81e-4f1f-bc19-bd2fd75b67ef
---
**Session 1 of ~8 on the Rust-First plan.** Closed plan item 1.1 only
(one-commit-per-session rule per plan §"Fresh session" bootstrap).

**What changed.** `sage-python/src/sage/pipeline.py` single-agent bypass
branch (~line 1019, after the validation_level block) now scales
`self._agent_loop.config.max_steps = {1:5, 2:10, 3:20}.get(ctx.system, 10)`.
Mirrors `agent_loop_factory.py:132-137`. Before this commit, the
singleton inherited `MAX_AGENT_STEPS=20` from `boot.py:279` regardless
of system tier → S1 tasks on the bypass path ran at 4x the factory-
intended step budget before the loop could exit.

**Why this works.** `agent_loop.py:424` reads `self.config.max_steps`
directly at each step — same live-read contract as `validation_level`
which was already mirrored. No need to touch AgentLoop itself.

**Regression test.** `tests/test_pipeline.py::test_pipeline_single_agent_scales_max_steps_by_system`
uses `SAGE_ABLATION_NO_TOPOLOGY=1` (via monkeypatch) to force the
bypass path for S2/S3 tiers — otherwise `hint="sequential"` would
route through `_build_topology_from_hint` and never touch the
singleton. Asserts config.max_steps == {1:5, 2:10, 3:20} per tier.

**Tests.** 36/36 `test_pipeline.py` green. Full suite 1927 passed;
the 5 errors + 1 failure (pydantic_ai_integration, e2e_campaign,
provider_pool_wiring) are a pre-existing "no current event loop"
asyncio-fixture pollution — tests pass in isolation, not introduced
by this change.

**Plan state.** 1.1 `[x]` (commit b7ced9d, session 1). Next = 1.2
stall_cap singleton audit (depends on 1.1 — needs max_steps set to
compute cap). Advisor-verified the H4-pattern risk before writing:
`agent_loop.py:424` reads `self.config.max_steps` per iteration, so
the mutation is observable; the plan's spy-mock test approach is
valid for this field.

**User feedback during session.**
- "N'utilise pas litellm nous l'avons remplacé par pydanticAI" → saved
  as `feedback_no_litellm.md`; the repo confirms migration happened
  2026-04-18 (boot_pipeline.py:74 comment, openai_compat.py deprecation).
  No provider code touched in this commit.
- "Tu devrais lire doc et le vault obsidian" → read the plan, spec,
  bypass-patterns.md, ADR-011 (placeholder awaiting 1.1-1.3), and
  Changelog-Apr9-20. No contradictions with 1.1 implementation.

**Obsidian updates deferred.** ADR-011 body is a placeholder that
will be finalized after 1.2 + 1.3 commits (per its own draft text).
Changelog-Apr9-20.md will get a Bloc 10 line when Phase 1 completes.
Per plan §Session Close Routine: "Obsidian touch (only when a major
milestone lands)" — a single item is not a milestone.
