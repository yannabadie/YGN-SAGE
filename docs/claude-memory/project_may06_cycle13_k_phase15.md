---
name: May 6 — Cycle-13 K Phase 1.5 ToolPolicy capability manifest (4 commits, cgpro DESIGN_LOCKED one-shot)
description: Cycle-13 K Phase 1.5 shipped end-to-end after Phase 0 closure. cgpro DESIGN_LOCKED one-shot in NEW conv `cgpro_phase15_toolpolicy_20260506` (in YGN-SAGE project per Yann directive 2026-05-06). 4 commits proven 3-actor recipe (cgpro DESIGN → Claude IMPLEMENT — codex skipped because scope was locked tight enough). 15 tests T1-T15. Ledger 9 → 10 invariants. Audit CLI shipped. claims_audit --strict GREEN with 20 claims.
type: project
originSessionId: 88857be6-7048-463a-8ee4-cb3b4cca20fd
---
## Context

Cycle-13 K Phase 0 had closed (12 commits `7cda0d9f..5863bb06`, narrative guard 27 docs × 14 patterns). Per cgpro round-2 verdict: **ToolPolicy capability manifest doit passer AVANT toute publication PyPI utilisable** parce qu'un research-preview multi-agent sans default-deny tool manifest ouvre une surface trop large.

Yann gave "go" → I drafted Phase 1.5 DESIGN with locked-spec 8-question schema + allowed/forbidden files + format de réponse, sent to cgpro NEW conv `cgpro_phase15_toolpolicy_20260506` (no `--resume` so auto-routes to YGN-SAGE project).

cgpro responded **DESIGN_LOCKED one-shot**. Then 4 commits implemented in single session.

## What shipped (4 commits, all on origin/main, `5863bb06..6497c427`)

| Commit | SHA | Subject |
|---|---|---|
| 1 | `218a695e` | feat(policy): sage.policy schema (ToolCapability enum + ToolPolicy + errors + manifest + env/TOML loading) |
| 2 | `dc01ee61` | feat(tools,policy): Tool dataclass extended + Tool.execute last-resort gate + Registry.register strict resolution + 31-entry built-in manifest |
| 3 | `5a4cfd1e` | feat(agent,policy): boot.py Stage-0 sets ToolPolicy.from_environment() — ContextVar propagation handles bypass automatically (PEP 567) |
| 4 | `6497c427` | test(policy,docs): 15 tests T1-T15 + sage.ops.toolpolicy_audit CLI + ledger invariant 10 + security.tool_policy_default_deny claim + README/AI-ARCH/CLAUDE.md updates |

## Capability vocabulary (locked)

`ToolCapability` enum: `pure / read_local / write_local / network / subprocess / dangerous`. **Single label per tool**. Multi-effect tools classify as `dangerous`. **No hierarchical implicit closure** — granting `dangerous` does NOT also grant `network`, granting `network` does NOT grant `write_local`, etc.

## Key contracts (cgpro DESIGN_LOCKED)

- **Default effective policy = `{pure}` only** (NOT pure+read_local — strictness over convenience for the research-preview PyPI install path).
- **Hybrid grant** : `policy.grant("network")` (programmatic) + `SAGE_TOOL_GRANTS=read_local,network` (env) + `~/.sage/tool_policy.toml` (TOML).
- **Failure semantics** : Registration → `ToolPolicyDeclarationError` (HARD-FAIL, no default-tag-dangerous fallback because that creates illusion of security). Runtime → `ToolPolicyDenied` raised inside `Tool.execute`, caught and returned as `ToolResult(is_error=True)` so OracleStack reads `trainable=False`.
- **Tool.execute is the last-resort gate** because `agent_loop_execution.execute_tool_call` can re-lookup ToolForge-synthesised tools after AgentLoop pre-check.
- **AgentTool default = `dangerous`** via `_CLASS_CAPABILITY_DEFAULTS["AgentTool"]` because it delegates to arbitrary `agent.run(...)`.
- **ContextVar propagation** : `_CURRENT_POLICY` ContextVar means `asyncio.create_task` (used by `pipeline_v2/execute.py:create_bypass_agent_loop`) inherits parent's policy automatically per PEP 567 — no manual snapshot/restore needed. T14 attests.
- **Audit CLI** does NOT execute handlers (per cgpro DESIGN trap 7).

## Public API surface (Q6)

```python
from sage.policy import (
    ToolCapability,
    ToolPolicy,
    ToolPolicyError,
    ToolPolicyDenied,
    ToolPolicyDeclarationError,
    get_effective_tool_policy,
    set_current_tool_policy,
)
```

Internals NOT re-exported: `_BUILTIN_TOOL_CAPABILITIES`, `_CLASS_CAPABILITY_DEFAULTS`, ContextVar plumbing.

## Built-in tool manifest (31 entries at ship)

By I/O profile: 0 pure (manifest), 9 read_local (read_file, list_files, search_files, search_repo, git_diff, search_memory, retrieve_context, summarize_context, filter_context, search_causal_chain, list_active_agents), 5 write_local (write_file, edit_file, create_file, delete_file, apply_patch, store_memory, update_memory, delete_memory), 3 subprocess (run_tests, execute_python, execute_code), 6 network (search_exocortex, refresh_knowledge, lookup_library_docs, context7_query, web_search, web_fetch), 9 dangerous (bash, execute_bash, execute_raw, sage_recurse, create_agent, call_agent, create_python_tool, create_bash_tool).

## Ledger invariant 10

"Tool capability declaration & grant enforcement". **Declaration AND enforcement**, NOT verified behavior — a `pure` tool can still lie at runtime; the contract is "declared capability is the maximum-safe summary, gates block at registration and at Tool.execute boundary". Behavior verification is future work (sandbox/WASI/AST confinement).

Heading "## The 9 invariants" → "## The 10 invariants" in `runtime-integrity-ledger.md`. README + AI-ARCHITECTURE.md + CLAUDE.md propagated via `sync_doc_counters.py` (CLAUDE.md line 16 manually updated since it's intentionally excluded from sync).

## Tests T1-T15 (locked exactly per cgpro DESIGN)

T1 pure_allowed_without_explicit_grant
T2 read_local_denied_by_default
T3 read_local_allowed_with_grant
T4 write_local_denied_by_default
T5 write_local_allowed_with_grant
T6 network_denied_by_default
T7 network_allowed_with_env_grant (SAGE_TOOL_GRANTS=network ≠ write_local — no implicit closure)
T8 subprocess_denied_by_default
T9 subprocess_allowed_with_grant
T10 dangerous_denied_without_exact_dangerous_grant (granting 4 tiers ≠ dangerous)
T11 dangerous_allowed_with_exact_grant
T12 registry_resolution_allows_manifest_and_rejects_unknown_unlabeled (uses local strict-resolver patch to bypass autouse permissive resolver)
T13 tool_execute_returns_error_result_on_policy_denial (handler NOT invoked when denied)
T14 bypass_factory_inherits_effective_policy_via_contextvar (asyncio.create_task PEP 567 inheritance)
T15 toolpolicy_audit_cli_lists_capabilities_and_effective_grants

## conftest.py autouse fixture (cgpro trap 8)

`_grant_all_tool_capabilities_in_tests` autouse fixture in `sage-python/tests/conftest.py`:
1. Sets ContextVar to `ToolPolicy(grants=frozenset(ToolCapability))` for the test scope (all 6 tiers granted).
2. Monkeypatches `sage.policy.manifest.resolve_tool_capability` to fall back to PURE on declaration error (lets ad-hoc test fixtures register without manual classification).

Production stays strict — only the test surface gets the relaxation. Operator env-vars (SAGE_TOOL_GRANTS, ~/.sage/tool_policy.toml) untouched per cgpro trap 8 "doit être isolé par monkeypatch".

## Final stats

- 11 commits Phase 0 (`7cda0d9f..5863bb06`) + **4 commits Phase 1.5** (`5863bb06..6497c427`) = 15 commits in cycle-13 K so far.
- 3089 → **3179** Python tests (+90 cycle-13 K-specific = 75 Phase 0 + 15 Phase 1.5).
- **20 claims** (was 19, +1 for `security.tool_policy_default_deny`): 8 delivered + 4 default-on + 4 evidence_pending + 2 opt-in + 2 planned + 1 retired.
- **10 invariants** in runtime-integrity-ledger.md (was 9, +1 for "Tool capability declaration & grant enforcement").
- **27 narrative-grade docs guarded** at PR-time (Phase 0 final).
- claims_audit --strict GREEN. sync_doc_counters --check OK. regenerate_claims_index --check OK. ruff clean.

## cgpro conversation continuity

Active conv: `cgpro_phase15_toolpolicy_20260506` (in YGN-SAGE project gizmoId `g-p-69ed9637e63c8191b61c9741b50d1c01`). DESIGN_LOCKED one-shot at session start. VERIFY round in flight at session end (BG `bffbtfgew`). Resume via `--resume cgpro_phase15_toolpolicy_20260506` for any future Phase 1.5 follow-up.

Older threads (kept for cycle continuity, NOT migrated to project):
- `Analyse approfondie de repo` (id `69fb0d11-...`) — ALIRE.md remediation Phase 0/0.6/0.6b/0.6c/0.6d/0.6e/0.6f. Closed for cycle-13 K Phase 0.
- `cgpro_pi_mono_pivot_20260505` — cycle-12 pi-mono pivot strategic thread.

## Recipe used (and lessons)

cgpro DESIGN → Claude IMPLEMENT (no codex used this time — scope was clear and ~600 LOC, single-actor implementation faster than 3-actor coordination). Claude VERIFY (smoke + tests + gates). cgpro VERIFY (post-push, in flight). SHIP done at commit/push.

If Phase 2.1 facade rewrite uses 3-actor recipe (codex for repetitive mechanical refactor of ~1500 LOC), the proven pattern is:
1. cgpro DESIGN → Claude validates 2-3 smallest stubs → codex implements 4-N → Claude verifies + commits → cgpro VERIFY → SHIP.

For Phase 1.5 single-actor was the right call. For Phase 2.1 multi-actor will be.
