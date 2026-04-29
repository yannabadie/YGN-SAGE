---
title: ADR-018 OracleStack v0 (Cycle 5, R9)
type: adr
status: shipped
date: 2026-04-29
commits: ["276cc7d4", "84b77f7e"]
tags: [runtime, oracle, training-gate, evidence, learning]
---

# ADR-018 — OracleStack v0 (Cycle 5, R9 + R9.0.1)

## Context

After cycles 1-4 shipped (RuntimeContracts → RuntimeEventLog → StateCore → RunFrame), Stage 6 learning was still vulnerable to the **A14 incident class**: bandit posteriors / MAP-Elites archive / online evolution / episodic memory consolidation all updated from outputs whose quality was inferred by `QualityEstimator`'s lexical heuristics OR by silent fallback when no real verifier was available. The 2026-04-26 cgpro audit of the bandit chain found two compounded bugs (Stage 0 `route()` was off-policy, Stage 5 `record()` was unreachable due to PyO3 name mismatch) that combined meant the bandit recorded NOTHING from production traffic — but its lexical-fallback path was still firing on edge cases.

R7 RunFrame just shipped the typed evidence surface (NodeRunRecord with provider/model + event seqs + state versions/deltas + cost/quality snapshots). The natural next move: **gate Stage 6 learning** so it consumes only verified evidence.

cgpro 2026-04-29 cycle-4-reassess locked R9 OracleStack as cycle 5. Hard invariant: **NO bandit / MAP-Elites / online-evolution / training-memory promotion update unless OracleVerdict.trainable=True**. Behind `SAGE_ORACLE=1` (opt-in v0 ; cycle 7 default-on flip after R6.1a + smoke validation).

## Decision

NEW module `sage/runtime/oracle/` with 6 files:

- **errors.py**: `OracleUnavailable(RuntimeError)`.
- **verdict.py**: public `OracleVerdict` + `EvidenceRef` frozen+slots dataclasses, `Literal` type aliases (`VerdictSource`, `QualityLabel`), tuple constants (`VERDICT_SOURCES`, `QUALITY_LABELS`), `ORACLE_VERDICT_SCHEMA_VERSION="0"`, full `__post_init__` validation.
- **config.py**: `OracleConfig` (min_confidence, enable_llm_judge, enable_lexical_fallback, timeout_per_oracle_sec, allowed_sources) + post-init validation.
- **_oracles.py**: private `_exact_oracle`, `_tool_oracle` (None v0; R6.1a v1), `_formal_oracle` (None v0; R6.1a v1), `_spec_oracle` (StateCore contradiction with cgpro VERIFY guard), `_llm_judge_oracle` (always-Abstain stub).
- **stack.py**: public `evaluate(view, *, final_output, bench_result, config)` — walks hierarchy, first-confident-trainable wins, else Abstain.
- **__init__.py**: lazy-imported `evaluate` to break circular with `run_frame` (RunFrame.oracle_verdict is forward-ref).

### OracleVerdict invariants

```python
@dataclass(frozen=True, slots=True)
class OracleVerdict:
    trainable: bool
    verdict_source: VerdictSource  # exact | tool | formal | spec | llm_judge | abstain
    quality_label: QualityLabel    # pass | fail | partial | unknown
    score: float | None            # 0.0..1.0, None for abstain
    confidence: float              # 0.0..1.0
    reason_codes: tuple[str, ...]
    evidence: tuple[EvidenceRef, ...]
    schema_version: Literal["0"] = "0"
```

`__post_init__` enforces:
- `trainable=True` requires `verdict_source != "abstain"` AND `score is not None` AND `0.0 <= score <= 1.0` AND `len(reason_codes) >= 1` AND `len(evidence) >= 1`.
- `verdict_source == "abstain"` requires `trainable=False` AND `quality_label="unknown"` AND `score is None`.
- **All `trainable=False` collapse to `verdict_source="abstain"`** — no ambiguous non-abstain non-trainable verdicts.
- `0.0 <= confidence <= 1.0`.

### Hierarchy: first-confident-trainable wins, else Abstain

```
exact > tool > formal > spec > llm_judge > abstain
```

- **Exact**: `bench_result["passed"]` from harness. Pass → `trainable=True, score=1.0`. **Fail → `trainable=True, label="fail", score=0.0`** (NOT abstain ; failure is high-value negative evidence).
- **Tool**: v0 always None (placeholder for R6.1a Tool/Formal v1 with deterministic evidence producers).
- **Formal**: v0 always None.
- **Spec**: StateCore contradiction detection. cgpro VERIFY round-trip caught lexical-fallback failure mode where merely DISCUSSING that an assumption was invalidated would train a negative reward. Fixed with `_INVALIDATION_MARKERS` guard list (14 negation phrases: "was wrong", "no longer holds", "was retracted", "was invalidated", "found to be incorrect", etc.). Only mention WITHOUT invalidation markers fires as `trainable=True` negative.
- **LLMJudge**: always-Abstain stub in v0 (real impl deferred to R9.1 — needs trust framework against citation gaming + sycophancy).
- **Abstain**: `trainable=False, verdict_source="abstain"`. Default fallback when hierarchy exhausted or all candidates below `min_confidence=0.7`.

### CRITICAL pipeline reorder when SAGE_ORACLE=1

Pre-R9 pipeline order:
```
execute → _record_to_memory → _stage_learn → emit_final_result
```

R9 reorders (only when `SAGE_ORACLE=1`):
```
execute → emit_final_result → oracle.evaluate → emit_oracle_verdict 
       → _record_to_memory(is_training_evidence=verdict.trainable)
       → _stage_learn (gated: each training sink wrapped in if verdict.trainable)
```

OFF mode preserves legacy order byte-identical.

### NEW event type: `oracle_verdict` (13th)

Emitted between `final_result` and `run_frame_summary`. `parent_event_id == final_result.seq`. Payload = `verdict.to_dict()` with no raw output. RunFrame extended with `oracle_verdict: OracleVerdict | None = None` field.

NEW `RunFrameView` read-only dataclass (safe subset, no raw output) — `_RunFrameBuilder.snapshot_view() -> RunFrameView` exposes the in-flight builder to `evaluate()` without breaching private internals.

### ALL training sinks gated (not just bandit)

Per cgpro DESIGN: bandit-only gating is insufficient. When `verdict.trainable=False`:
- `bandit.record_outcome` / `pipeline._record_bandit_outcome_checked` — skipped
- `engine.record_outcome` (MAP-Elites archive) — skipped
- `engine.should_evolve` / `engine.evolve` — skipped
- semantic / consolidation promotion as training evidence — skipped
- Episodic memory MAY still record but tagged `is_training_evidence=False`

### `SAGE_ORACLE` added to `_ALLOWED_FEATURE_FLAGS` (8 keys total in R7's allowlist)

## R9.0.1 — Evidence-starved Abstain pin (`84b77f7e`)

cgpro recommended pre-cycle-6 follow-up: explicit test documenting v0 hierarchy fallthrough state.

`test_oracle_v0_evidence_starved_default_falls_through_to_abstain` pins: with `SAGE_ORACLE=1` but no bench_result + Tool/Formal returning None v0 + no StateCore contradiction + LLMJudge stubbed, the hierarchy MUST fall through to Abstain. Documents the cycle 5 → 6 handoff state and the intended R6.1a insertion point (when delta producers ship, Tool/Formal will start returning real verdicts and runs that previously hit Abstain may become trainable).

NEW `roadmap.md` "Current operational gates" section listing all 4 strategic feature flags + cycle-7 default-on gate criteria + explicit "Do NOT flip SAGE_ORACLE default-on until R6.1a + smoke validation".

## Consequences

- 17 R9 acceptance tests (14 base + 2 cgpro mandatory + 1 cgpro VERIFY round-trip mention-vs-reassertion) + 1 R9.0.1 placeholder test.
- 1528 LOC across 30 files in R9 (substantial pipeline reorder cascade through 5 src files + 8 test files) + 65 in R9.0.1.
- mypy 0/204 (was 198, +6 for new oracle module).
- 2447/2447 full Python suite passing (excluding API-key-gated). +90 tests vs R0 baseline 2357.
- Sets the architectural foundation for R6.1a (deterministic evidence producers) and R9.1 (real LLMJudge) and cycle-7 default-on rollout.
- A14 incident class fixed at the architectural level for the first time: Stage 6 learning only updates from verified evidence.

## Related

- [[ADR-014-RuntimeContracts-Cycle1]] — R2 unification was prerequisite
- [[ADR-015-RuntimeEventLog-Cycle2]] — oracle_verdict extends the R5 event taxonomy
- [[ADR-016-StateCore-Cycle3]] — `_spec_oracle` consumes `view.state_frames` for contradictions
- [[ADR-017-RunFrame-Cycle4]] — `RunFrameView` is the safe input to `evaluate()`
- `docs/contracts/runtime-event-log.md` — oracle_verdict event row + SAGE_ORACLE mode-aware contract row
- `roadmap.md` — cycle-7 default-on gate criteria
- `tests/golden/runtime_events/oracle_verdict.json` — golden fixture
- A14 incident: `docs/migrations/2026-04-27-a14-reset.md`
