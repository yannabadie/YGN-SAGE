# C2b validation smoke — N=10 SWE-bench Lite, 2026-04-21

## TL;DR

**C2b is empirically falsified.** The prompt-level fix (naming `search_exocortex` in the tool menu + dedicating a workflow step to it) did **not** cause the model to call the tool even once across 10 tasks and 315 total tool invocations.

Root cause: the `search_exocortex` tool's **own description** (in `sage.tools.exocortex_tools`) advertises scope = `"MARL, cognitive architectures, formal verification, evolutionary computation, memory systems"`. No SWE-bench task is in those research domains, so the LLM correctly skips the tool — the prompt text can suggest it, but the function-calling schema is authoritative.

## What we ran

```
python -m sage.bench --type swebench --dataset lite --limit 10 \
  --output docs/benchmarks/2026-04-21-swebench-c2b-smoke.json
```

With `SAGE_EXOCORTEX_STORE=fileSearchStores/ygnsageresearch-wii7kwkqozrd` loaded from `.env`. Gen-only (no Docker eval — the hypothesis is about tool-usage counts, not pass-rate).

Instrumented logging added on the same branch: `sage.agent_loop_execution.execute_tool_call` now emits `tool.call name=<X> args_keys=[...] output_len=N` per invocation so we can attribute calls by tool name (previously only a per-node aggregate `tool_calls=N` was logged — useless for this audit).

## Results — tool attribution

| Tool | Calls | % |
|------|------:|--:|
| `execute_bash` | 315 | 100.0 % |
| `search_exocortex` | **0** | **0.0 %** |
| `refresh_knowledge` | 0 | 0.0 % |
| `sage_recurse` | 0 | 0.0 % |
| (anything else) | 0 | 0.0 % |

## Results — generation outcomes

| Outcome | Count | Task IDs |
|---------|-------|----------|
| PATCH | 5 | astropy-12907, astropy-14995, astropy-6938, astropy-7746, django-11001 |
| EMPTY | 3 | astropy-14182, astropy-14365, django-10914 |
| ERR (timeout_300s) | 2 | django-10924, django-11019 |

Mix: 6 astropy + 4 django tasks — the typical first-10 of SWE-bench Lite.

Comparison with earlier smokes on the same slice:

| Smoke | Gen rate | search_exocortex calls | prompt mentions search_exocortex? |
|-------|---------:|-----------------------:|:---------------------------------:|
| v13 (pre-C2b) | 50 % | 0 | No |
| v17 (pre-C2b) | 60 % | 0 | No |
| N=50 partial (pre-C2b) | 63 % | 0 | No |
| **C2b (this run)** | **50 %** | **0** | **Yes (menu + MUST clause + Step 4)** |

The gen-rate drift between runs is well inside the ±10 pp per-task-flip noise band we established in the N=10 → N=50 session on 2026-04-21; no signal there.

## Why the prompt change didn't move behavior

Every tool the LLM sees is described by TWO things:
1. **Prompt text** — what C2b edits (menu + MUST clause + Step 4).
2. **JSON Schema description** — what the function-calling API sends to the model ALONGSIDE the prompt.

`sage-python/src/sage/tools/exocortex_tools.py:15-20` ships schema description:

> *"Search the ExoCortex knowledge store for research papers and insights. Use when you need specific research knowledge about MARL, cognitive architectures, formal verification, evolutionary computation, or memory systems."*

From the model's point of view, the schema description is the authoritative "when to call this" hint. The SWE-bench tasks we hand it — *"Set default FILE_UPLOAD_PERMISSION to 0o644"*, *"astropy WCS serialization raises on round-trip"* — match **none** of those domains. The model correctly decides the tool is out of scope and reaches only for `execute_bash`.

C2b's addition in the system prompt said "use search_exocortex for library API contracts" — which directly contradicts the tool's own schema description. When the two disagree, the model trusts the schema (a documented LLM behavior, cf. OpenAI function-calling spec).

## Follow-up: C2c options

Three realistic paths — pick one.

### Option A — Honest re-framing (smallest, principled)

Revert the "library API contracts" claim in `sage.input.swebench.SWEBENCH_SYSTEM_TEMPLATE` and `sage.tools.exocortex_tools` to match each other. Both should say the same thing: `search_exocortex` is for **research context** (formal verification, evolutionary computation, etc.), not library docs. This stops the prompt from lying about the tool.

Pros: 10 LOC, no new integrations, ships today.
Cons: SWE-bench tool-usage mix stays at 100 % bash forever (the real need — library docs — has no tool to serve it).

### Option B — Broaden the ExoCortex store

Teach `exocortex.query()` to also surface library documentation (django/astropy/flask/requests/sqlalchemy). Would need a separate ingestion pipeline feeding official docs into the same File Search store.

Pros: Keeps one tool surface.
Cons: Weeks of work; conflates "research" and "docs" retrieval with different ranking needs; storage cost.

### Option C — New tool: `lookup_library_docs` backed by Context7 MCP

Context7 MCP (already listed under `claude.ai Context7` in this environment's MCP catalog) is *designed* for on-demand library docs. Wire a sage-side `Tool` that hits Context7's `resolve-library-id` + `query-docs` flow, register it at boot next to `search_exocortex`, and update C2b's prompt to point at the new tool for library contracts while keeping `search_exocortex` for research-paper context.

Pros: Real capability that actually serves the 2026-04-21 audit's finding; principled separation.
Cons: MCP wiring is new infrastructure for `sage-python` — ~200 LOC + tests; Context7 availability in non-Claude-Code contexts (when users run sage outside this environment) is TBD.

### Recommendation

**A now, B never, C as a deliberate design discussion.** Ship A as C2c to stop over-promising, then open a separate spec on whether a Context7-backed library-docs tool is the right abstraction (it probably is — but ADR-013 worthy, not a drive-by addition).

## Artifacts

* Log: [`2026-04-21-swebench-c2b-smoke.log`](2026-04-21-swebench-c2b-smoke.log) (1625 lines, 315 `tool.call` events).
* Predictions JSON: written at end of run — attached if present.
* Instrumentation commit: `tool.call name=X args_keys=Y output_len=Z` now emitted per invocation in `sage.agent_loop_execution` — stays wired for every future smoke, so per-tool attribution is permanent.
