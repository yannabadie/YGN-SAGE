---
title: ADR-019 EvidenceProducers v0 (Cycle 6, R6.1a)
type: adr
status: shipped
date: 2026-04-29
commits: ["38c0da4e", "19ea317c"]
tags: [runtime, oracle, evidence, deterministic, cycle-7-prep]
---

# ADR-019 — EvidenceProducers v0 (Cycle 6, R6.1a)

## Context

R9 OracleStack v0 (cycle 5, ADR-018) shipped the verdict hierarchy + Stage 6 trainable gate behind `SAGE_ORACLE=1`, but Tool/Formal oracles were always-None placeholders and `_spec_oracle` triggered on substring scanning of `final_output` for invalidated assumption IDs (the R9 cgpro VERIFY caught this lexical-fallback failure mode and bolted on `_INVALIDATION_MARKERS` as a guard).

The non-negotiable R9 invariant ("NO bandit / MAP-Elites / online-evolution / training-memory promotion update unless `OracleVerdict.trainable=True`") therefore held only for the Exact path (bench harness `bench_result["passed"]`). Every other source of evidence either Abstained (Tool/Formal v0) or fired on lexical heuristics (Spec). For cycle-7 default-on flip, this needed the deterministic delta substrate that R9's design called the "v1 insertion point".

cgpro 2026-04-29 cycle-5-reassess locked R6.1a as cycle 6: typed `RuntimeDelta` records emitted by 6 producers (tool / test / diff / formal / code-node / planner), surfaced through `RunFrame.runtime_deltas`, consumed by promoted Tool/Formal v1 + structured Spec.

## Decision

NEW module `sage/runtime/evidence/` (11 files, ~1200 LOC src + 666 LOC test):

- **delta.py**: public `RuntimeDelta` frozen+slots dataclass, `ProducerName` + `DeltaPolarity` `Literal` aliases, `PRODUCERS` / `POLARITIES` tuple constants, `RUNTIME_DELTA_SCHEMA_VERSION="0"`. `__post_init__` enforces:
  - `(producer, delta_kind)` ∈ `_DELTA_KIND_TABLE` (6 producers × 4-6 kinds = 30 valid pairs).
  - `polarity` ∈ `_POLARITY_RULES[producer][delta_kind]` (per-kind polarity, e.g. `test_parser/tests_failed` ⇒ `{"negative"}`, `formal_verifier/verifier_unavailable` ⇒ `{"neutral"}`).
  - `confidence` ∈ [0, 1], `run_id` non-empty, `event_seq` ≥ 0 or None.
  - `payload` validated against per-(producer, delta_kind) schema (`PAYLOAD_ALLOWED_KEYS`), then deep-frozen via `MappingProxyType` of recursively-frozen nested mappings/sequences.
  - `evidence_hash = sha256(canonical_json({schema_version, producer, delta_kind, polarity, source_id, payload}))` — **excludes** `run_id`/`node_run_id`/`event_seq`/`confidence` (cgpro Q1.c lock: collision-free across re-emission, dedup-friendly across run boundaries).
- **errors.py**: `EvidenceError(RuntimeError)`.
- **payloads.py**: `PAYLOAD_ALLOWED_KEYS` per (producer, delta_kind), `PAYLOAD_MAX_STRING_LENGTHS`, `FORBIDDEN_KEYS` (defense against payload-injected schema fields), `validate_payload`, `compute_evidence_hash`, `deep_freeze_payload`, `canonical_json`.
- **producers/tool.py**: `produce_tool_execution_deltas` + `produce_test_parser_deltas`.
- **producers/formal.py**: `produce_formal_verifier_deltas` (obligation_proved / obligation_refuted / counterexample_found / obligation_unknown / verifier_unavailable / assumption_invalidated). Q5.b lock: requires `obligation_id` + `verifier_id` + `encoding` — refuses to fire on raw solver SAT/UNSAT.
- **producers/code_node.py**: `produce_code_node_structured_return_deltas` (validates payload against declared schema).
- **producers/diff.py**: `produce_diff_deltas` (clean / hunk_header_mismatch / repair_accepted / repair_rejected / context_mismatch / patch_applied / patch_failed) extending swebench diff verifier outcomes.
- **producers/parsers.py**: pytest summary-line regex parser (passed/failed/error counts).
- **producers/planner.py**: `produce_assumption_invalidated_deltas` — replaces R9 spec oracle's substring scan trigger.

### Hierarchy v1: Tool / Formal / Spec promoted from None placeholders

```
exact > tool > formal > spec > llm_judge > abstain
```

- **Tool** (`_tool_oracle`): consumes `test_parser` / `tool_execution` deltas.
  - All `tests_passed` deltas, no fatal: `trainable=True, label="pass", score=1.0, confidence=1.0`.
  - Negative deltas (`tests_failed`/`tests_partial`):
    - If any `tests_partial` AND `passed_count + failed_count + error_count > 0`: `score = passed/total`, `label="partial"`. Q5.a lock: partial wins ONLY on deterministic counts; ambiguous → fail not partial.
    - Else: `score=0.0`, `label="fail"`.
  - `fatal_failure` deltas: trainable fail ONLY when `payload["fatal_scope"] == "claimed_task_output"` (cgpro R6.1a verify push-back). Generic agent-loop tool exceptions (`fatal_scope="incidental_tool_call"`) and unscoped fatals fall through to abstain. Code-node executor in `topology/runner.py` tags `claimed_task_output`; agent_loop_execution tags `incidental_tool_call`.
- **Formal** (`_formal_oracle`): consumes `formal_verifier` deltas.
  - Trainable kinds (`obligation_proved` / `obligation_refuted` / `counterexample_found`) require COMPLETE obligation semantics enforced at BOTH producer and oracle layers (cgpro R6.1a verify defense-in-depth):
    - `obligation_id` + `verifier_id` + `encoding` + `solver_status` all required.
    - `obligation_proved` ⇒ `solver_status="unsat"` (proof of no counterexample).
    - `obligation_refuted` / `counterexample_found` ⇒ `solver_status="sat"` (refutation/witness).
    - Producer rejects mismatched `solver_status`; oracle re-validates via `_formal_delta_is_complete` before constructing a verdict — incomplete deltas force abstain.
  - `obligation_unknown` / `verifier_unavailable`: Abstain (no completeness requirement; producer keeps lenient contract).
- **Spec** (`_spec_oracle`): structured-only stub in v1 (cgpro R6.1a verify push-back).
  - `_INVALIDATION_MARKERS` constant + substring scan REMOVED entirely.
  - For v1, spec oracle ALWAYS abstains — `StateFrame.invalidated_assumptions` alone OR `formal_verifier/assumption_invalidated` alone are insufficient evidence per cgpro lock. We need a structured claim-dependency channel proving the final output reasserts the invalidated fact. That channel ships in cycle 7+; until then, spec oracle is effectively a placeholder that always returns None.
  - Hierarchy still falls through to LLMJudge stub → Abstain when spec abstains.

### NEW field on RunFrame: `runtime_deltas: tuple[RuntimeDelta, ...]`

Q4.a lock — name `runtime_deltas` (NOT `evidence_deltas`, distinguishes from R6 `StateDelta`). Stable ordering: insertion order from the per-run `_RunFrameBuilder.emit_evidence_delta(...)` (Q4.b — chronological by emission, not sorted by node_run_id; deltas are append-only and the seq encoding makes `parent_event_id` chains durable).

### Pipeline integration (3 producers live, 3 scaffolded)

Q3.b lock — R6.1a ships LIVE emission for:
- `agent_loop_execution.py:174` — tool execution deltas (gated `SAGE_ORACLE=1`).
- `bench/swebench_diff_verifier.py:384` — diff verifier deltas.
- `bench/swebench_patch_repair.py:364` — repair outcome deltas.

Scaffolded (schemas + tests via fixtures, no live emission yet — cycle-7 / R6.1b):
- `producers/code_node.py` — code node structured return.
- `producers/formal.py` — formal verifier (Q5.b infra ready, awaits Z3/OxiZ wire-up).
- `producers/planner.py` — planner_decision (topology_selected / decomposition_applied).

### OFF-mode preservation (Q6.a lock)

`SAGE_ORACLE` flag gated at all 8 emission sites:
- `agent_loop_execution.py:174`
- `bench/swebench_diff_verifier.py:384`
- `bench/swebench_patch_repair.py:364`
- `pipeline.py:765, 842, 1661, 2369, 2393, 2494`
- `topology/runner.py:1095`
- `run_frame/builder.py:39` (env capture allowlist)

When `SAGE_ORACLE` unset/0: no producer fires, `RunFrame.runtime_deltas == ()`, byte-identical to R9 OFF mode.

### Test fixtures: 22 round-trip JSON pairs

`tests/fixtures/runtime_evidence/{tool_execution_*,pytest_*,diff_*,formal_*,code_node_*}.{input,expected}.json` — input is the producer call payload, expected is the resulting `RuntimeDelta` serialized. `test_runtime_evidence.py` round-trips: load input → call producer → serialize delta → assert equality with expected. Catches schema drift across all 6 producers.

## R6.1a verify push-back — closeout (`19ea317c`)

cgpro 2026-04-29 R6.1a VERIFY round 1 produced **PUSH BACK** with 3 blocking findings against `38c0da4e`. All three closed in `19ea317c` (10 files, +349/-134):

1. **Spec oracle lexical removal**: `_INVALIDATION_MARKERS` deleted, substring scan deleted, spec oracle becomes structured-only (effectively abstain in v1 until claim-dep channel ships).
2. **Formal completeness gate**: producer rejects trainable formal deltas missing `verifier_id`/`encoding`/`solver_status` or with status inconsistent with kind direction. Oracle re-validates via `_formal_delta_is_complete` defense-in-depth.
3. **Tool fatal_scope gate**: `fatal_failure` deltas now require `fatal_scope ∈ {"claimed_task_output", "incidental_tool_call", "unknown"}` payload. ToolOracle trains fail only on `claimed_task_output`. Agent-loop tool exceptions tagged `incidental_tool_call` → abstain. Code-node executor tagged `claimed_task_output` → trainable fail.

10 Gate A regression tests added (`TestFormalCompletenessGate` 6 tests + `TestToolFatalScopeGate` 4 tests) per cgpro recommendation.

R9 lexical reassertion test deleted (`test_spec_oracle_state_contradiction_returns_trainable_negative`); R9 mention-vs-reassertion test rewritten as `test_spec_oracle_lexical_substring_does_not_train` covering both reassertion AND mention phrasings (both abstain in v1).

## Consequences

- **47 + 10 = 57 R6.1a acceptance tests** (37 base + 10 Gate A + 10 oracle integration) + 22 fixture pairs.
- **3339 LOC across 64 files** total (initial 38c0da4e: 2990+/-23, fixup 19ea317c: 349+/-134 across 10 files).
- **mypy 0/32** R6.1a-touched files post-fixup.
- **ruff clean** on all R6.1a-touched files.
- **Verify-local regression** on R5/R6/R7/R9 cycle suites: exit 0, ~138 cumulative tests passing post-fixup.
- **Cycle-7 readiness**: R6.1a is the gate. Default-on flip of `SAGE_ORACLE=1` requires (a) R6.1a ship + push-back closeout (DONE), (b) Gate B full local suite green (DONE), (c) Gate C no-spend synthetic ON smoke covering 7 oracle scenarios (TODO), (d) Gate D paid SWE-bench N=10 with throwaway bandit DB (TODO, separate ticket).
- **A14 incident class**: Stage 6 learning gate now backed by structured deterministic evidence at 3 live emission points (agent_loop_execution, swebench_diff_verifier, swebench_patch_repair) AND defended by 3 fatal-evidence gates (formal completeness, tool scope, spec structured-only). No code path remains where lexical heuristics can produce a trainable verdict.

## Related

- [[ADR-014-RuntimeContracts-Cycle1]] — R2 `_run_core` event seqs are what builder records on delta emission.
- [[ADR-015-RuntimeEventLog-Cycle2]] — `evidence_delta` is the 14th event type extending the R5 taxonomy.
- [[ADR-016-StateCore-Cycle3]] — `_spec_oracle` consumes `view.state_frames` + R6.1a `assumption_invalidated` deltas (dual source).
- [[ADR-017-RunFrame-Cycle4]] — `RunFrame.runtime_deltas` is the new field on the typed snapshot.
- [[ADR-018-OracleStack-Cycle5]] — Tool/Formal v1 + structured Spec replace v0 None / lexical placeholders.
- `docs/contracts/runtime-event-log.md` — evidence_delta event row (cycle-7 doc update).
- `roadmap.md` — cycle-7 default-on gate criteria.
- `tests/golden/runtime_events/evidence_delta.json` — golden fixture (cycle-7 add).
- `.tmp/cgpro_r6_1a_design_locked_spec.md` — locked spec from cgpro 2026-04-29 design round.
