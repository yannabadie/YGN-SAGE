---
name: May 6 — Cycle-13 K Phase 0 + 0.6 closure (ALIRE.md remediation, 7 commits, cgpro 2 rounds VERIFY)
description: Cycle-13 K Phase 0 (truthfulness foundation) shipped over 7 commits responding to ALIRE.md. cgpro on conv `Analyse approfondie de repo` (id 69fb0d11-...) ran 2 EDIT_REQUIRED rounds (initial plan + post-Phase-0 patch). Claims registry, doc-counter sync, strict CI gates, evidence anchor pinning, Path 6 default-off non-regression all shipped. Total 47 tests Phase-0-specific, 3136 Python tests global. 0 regressions.
type: project
originSessionId: 88857be6-7048-463a-8ee4-cb3b4cca20fd
---
## Context

ALIRE.md (cgpro analysis 2026-05-06) flagged YGN-SAGE as ambitious lab but NOT an "imbattable agent" yet. Three structural problems:
(1) too many claims unanchored to artefact;
(2) external validation insufficient;
(3) god-object pipeline central.

Yann instructed: read ALIRE → submit plan to cgpro on new conv `Analyse approfondie de repo` (id `69fb0d11-9bd8-8390-a074-edb6826f8cb6`) with explicit GitHub-verification request → integrate corrections → enter plan mode → apply.

## What shipped (12 commits, all on origin/main, `7cda0d9f..5863bb06`)

| Phase | Commit | Subject |
|---|---|---|
| 0.1a | `6e8ac732` | Minimal claims registry (19 claims, 7 categories) + audit CLI non-blocking |
| 0.2 | `18f9c65e` | Sync test counters + CI gate `doc-counters-coherence.yml` |
| 0.3 | `2fdd8ae5` | Invariant count SoT (AI-ARCHITECTURE.md 8→9) + ledger heading-vs-table parser |
| 0.1b | `5619b3cf` | Strict current.json gate (schema v2: snapshot+generated_for_commit_sha, codified ≤1-commit grace, no implicit) |
| 0.4 | `763d1fc1` | Pin evidence anchors + loose-OR contract (test OR benchmark) + claims_audit --strict enforcing |
| 0.5 | `ba6078f4` | Path 6 default-off non-regression test + Phase 0 closure |
| **0.6** | `aadaf6d6` | **Truthfulness coherence patch (cgpro EDIT_REQUIRED post-Phase-0)** |
| **0.6b** | `70c836e2` | **Narrative drift sweep + guard test 4 docs (cgpro round-3 EDIT_REQUIRED)** |
| **0.6c** | `4aa46eb1` | **Guard scope 4→10 docs + DEAD CODE retired + Pillar-5 archive banner (round-4 EDIT_REQUIRED)** |
| **0.6d** | `88a67449` | **Final sweep: paren-attribution patterns + 3 archive docs in guard (round-5 EDIT_REQUIRED)** |
| **0.6e** | `1b8d6d9f` | **Long-tail Papers/Benchmarks/Decisions sweep + 12 docs added (round-6 EDIT_REQUIRED)** |
| **0.6f** | `5863bb06` | **Local repo-wide grep closeout: 2 more docs + adaptive_router.py docstring (cgpro round-7 UI flaky)** |

**Final stats**: 19 claims (7 delivered + 3 default-on + 4 evidence_pending + 2 opt-in + 2 planned + 1 retired). `python -m sage.ops.claims_audit --strict` exit 0. 3089 → **3162** Python tests (+73 Phase-0-specific). mypy 0 / ruff clean.

**Narrative guard final (Phase 0.6e)** : `tests/test_narrative_guard.py` parametrized over **25 narrative-grade docs** with **14 forbidden patterns**:
- 4 originaux Phase 0.6b: README, AI-ARCHITECTURE.md, CLAUDE.md, .claude/rules/architecture.md
- +6 Phase 0.6c: .claude/rules/critical-directives.md, research-decisions.md, ui/README.md, ui/app.py, sage-python/src/sage/routing/README.md, sage-core/src/README.md
- +3 Phase 0.6d: YGN-SAGE/Architecture/Pillar-5-Strategy.md, Pipeline.md, 00-Architecture-MOC.md
- **+12 Phase 0.6e**: YGN-SAGE/Papers/{kNN-Routing.md, 00-Papers-MOC.md}, YGN-SAGE/Benchmarks/{Routing-GT.md, 00-Benchmarks-MOC.md}, YGN-SAGE/Decisions/{ADR-001-kNN-over-Heuristic.md, 00-Decisions-MOC.md}, docs/papers/{paper1_knn_routing.md, paper2_sage_system.md}, docs/benchmarks/results.md, sage-python/src/sage/pipeline_v2/classify.py (code-side docstring!), sage-python/README.md, sage-python/src/sage/bench/README.md.

Patterns: `92% GT` / `88% GT` / `34% GT` / `93.3% GT` / `kNN 92%` / `SystemRouter 88%` / `Path 6: Learned` / `Path 6 (learned` / `OR Path 6` / `kNN (...92%)` / `SystemRouter (...88%)` / `ComplexityRouter (...34%)` / `DEAD CODE` / `kNN [≤160 chars] 92%` / `kNN [≤160 chars] 93.3%` / `SystemRouter [≤160 chars] 88%` / `56/60` / `46/50` / `ComplexityRouter [≤160 chars] 45%`. Same-line caveat allowed: `evidence_pending`, `CLAIMS.yaml`, `routing.knn_92pct`, `routing.system_router_88pct`, `historically`, `historic`, `non-autoritative`, `Priority-3`, `emergency fallback`, `AUDIT2`, `corrected`, `NOT dead code`.

## cgpro 2-round VERIFY summary

### Round 1 (pre-Phase-0): EDIT_REQUIRED
GitHub-verified at `db304bc6`. Confirmed:
- pipeline.py 1800 LOC unchanged (god-object class fully verified)
- `pipeline_v2/__init__.py` says `"Not a façade"` — Phase C open
- README at db304bc6 STILL said "8 invariants" (drift on README too, not just AI-ARCHITECTURE)
- current.json embarked `git.commit_sha = 32d39bdf` (a commit BEFORE its claimed HEAD)
- `smmu.rs + smmu_bridge.rs` have NO save_state/load_state
- prompt_injection wired only in agent_loop.py, not pipeline.py
- Wheels matrix already partial (Linux x86_64 + Windows x86_64 + macOS arm64 × Py 3.12/3.13), missing Linux aarch64 + macOS x86_64 + tag/release trigger + TestPyPI

Sub-order imposed: `0.1a → 0.2+0.3 → 0.1b → 0.4 → 0.5`. Approach details (loose-OR audit, codified explicit grace, doc-counter propagation regex strategy) all from cgpro answer.

### Round 2 (post-Phase-0): EDIT_REQUIRED Phase 0.6
4 traps + 1 bonus:
1. **README ↔ CLAIMS drift**: capability state table said `delivered` for kNN 92% / topology 6-paths but registry said `evidence_pending`. → Split rows into capability (delivered) vs metric (evidence_pending).
2. **"Path 6" collision**: Rust enum says path 6 = TemplateFallback, registry says path6_learned = learned policy, README listed 7 entries. → README "How a task flows" lists 6 engine paths with template fallback as #6 + separate "Optional learned-policy path (NOT counted in 6)". Registry description disambiguates. Env-var SAGE_ENABLE_PATH6 unchanged for backward compat.
3. **Path 6 test over-promise**: test promised `path6_usage_count == 0` but proved only env-var/source guards. → Docstring rewritten as EXPLICIT NON-PROMISE; runtime-counter check deferred to Phase 2.1/2.2.
4. **sync_doc_counters docstring sur-vendait CI**: claimed `pytest --collect-only` gate that doesn't exist. → Docstring corrected to "COUNTER-PROPAGATION GATE" only. (Choice: corrected docstring instead of adding ~30min CI gate.)
5. **Bonus loose-OR prefix**: evidence_benchmark must be under `docs/benchmarks/` or `docs/audits/`. → Added `_evidence_benchmark_prefix_ok()` with backslash normalization + 4 tests.

Phase 0.6 also added AI-ARCHITECTURE.md callout pointing readers at CLAIMS.yaml.

## Key code/contracts shipped

- **`docs/claims/{routing,topology,memory,packaging,benchmarks,security,runtime}.yaml`** — single source of truth for capability/metric status.
- **`docs/CLAIMS.yaml`** — autogenerated index (sorted by id, do not edit directly).
- **`sage-python/src/sage/ops/claims_audit.py`** — AUDIT (default) / STRICT (--strict) modes. Loose-OR contract: delivered/default-on requires (test OR benchmark anchor) + evidence_commit. evidence_benchmark MUST be under docs/benchmarks/ or docs/audits/.
- **`scripts/regenerate_claims_index.py`** — write + --check modes; volatile-line stripping for stable check.
- **`scripts/sync_doc_counters.py`** — read current.json → propagate to README badge + AI-ARCHITECTURE.md table + .claude/rules/architecture.md project-structure block. `--bump-commit-sha SHA` writes both `snapshot_commit_sha` and `generated_for_commit_sha` (schema v2). Invariant counter loaded from ledger heading + table-row count check.
- **`.github/workflows/doc-counters-coherence.yml`** — propagation gate: sync --check + regenerate --check + claims_audit --strict.
- **`.github/workflows/strict-current-json-coherence.yml`** — schema v2 + snapshot==generated SHA + git rev-list distance ≤ 1 (codified grace, NOT implicit).
- **`sage-python/tests/test_path6_default_off.py`** — 3 tests: claim status opt-in, RunFrameBuilder allowlist, source-no-implicit-truthy-set.

## Methodology validated this cycle

- **cgpro 2-round VERIFY pattern proven**: pre-implementation review (DESIGN), post-Phase-N closure (POST_PUSH report). Both rounds caught real consistency drifts that local exploration missed.
- **GitHub-verified vs local-state divergence**: cgpro reads GitHub HEAD; local devs read working tree. Drift at the boundary is frequent and ALWAYS worth a re-verification call.
- **"Distance ≤ 1 codified explicit grace"** is the right escape from the chicken-egg "current.json describes the commit it's IN" problem. Cgpro accepted it explicitly.
- **Loose-OR audit contract** (test OR benchmark anchor) is more truthful than strict-AND for benchmark claims where the artefact IS the evidence.
- **CLAUDE.md INTENTIONALLY excluded from sync** for invariant-count propagation because it has historical-timeline references ("5 invariants at cycle-8 closure") that must NOT be auto-rewritten.
- **Phase ordering "véridicité avant ambition"**: claims registry → doc alignment → strict CI gates → evidence pinning → closure. Each phase ships green CI before next starts.

## Next blocks (cgpro queue, awaiting GO_NEXT_BLOCK on conv 69fb0d11)

1. **Phase 1.5 ToolPolicy capability manifest** (cgpro priority — BEFORE PyPI publish): pure / read_local / write_local / network / subprocess / dangerous capability labels. Default-deny per tier. Hooks in AgentLoop.run() + bypass factory. 12-15 tests.
2. **Phase 2.1 ADR-015 Phase C facade rewrite**: pipeline.py 1800 LOC → < 300 LOC facade. Prereq: 25 P9 phase-1 characterization tests byte-identical before/after. Recipe codex 3-actor.
3. **Phase 1.1 Wheels release-gating**: complete matrix (add Linux aarch64 + macOS x86_64) + tag/release trigger + TestPyPI dry-run + post-publish sentinel.
4. **Phase 1.2 SAGE_CLI_PROTOCOL v0 closure**: cli_progress / set_budget / cancel / cli_complete.final_seq.
5. **Phase 1.3 Issue #14 + Dependabot triage**: 8 PRs + Python dependency drift.
6. **Phase 2.2 prompt_injection wiring complet**: propagate to pipeline.py + TopologyRunner + bench harness.
7. **Phase 2.4 replay déterministe minimal**: `python -m sage.ops.replay --trace <path> --task <id>`.
8. **Phase 2.5 S-MMU persistence**: decide ship Rust persistence OR document volatile cache contract.
9. **Phase 3.1 SWE-bench Pro 4-arms N≥50 cloud** (with $240-460 API budget).

## Files to consult for cycle-13 K continuation

- `.tmp/cgpro_alire_plan_draft.md` — original plan submitted to cgpro
- `.tmp/cgpro_alire_phase0_post_push.md` — Phase 0 closure report (POST_PUSH)
- `.tmp/cgpro_alire_phase06_post_push.md` — Phase 0.6 closure report (POST_PUSH)
- `~/.claude/plans/transient-napping-brooks.md` — full approved plan
- `docs/claims/*.yaml` — registry source of truth
- `docs/CLAIMS.yaml` — autogenerated index

## Active cgpro conversation

`Analyse approfondie de repo` (id `69fb0d11-9bd8-8390-a074-edb6826f8cb6`). Use `cgpro ask --resume "69fb0d11-9bd8-8390-a074-edb6826f8cb6"` for continuity. Per CGPRO.md: pre-commit DESIGN + post-push REPORT pattern.
