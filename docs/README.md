# Documentation

Central documentation directory for YGN-SAGE architecture, plans, audits, and benchmark results.

## Directory Structure

### `plans/` -- Architecture Designs and Implementation Plans

Chronologically named design documents and implementation plans for each development phase:

- `2026-03-02-ygn-sage-architecture-design.md` -- Original architecture design
- `2026-03-02-ygn-sage-implementation.md` -- Initial implementation plan
- `2026-03-05-*` -- Phase 1-2 designs: knowledge pipeline, cognitive routing, Z3/SIMD/S-MMU integration
- `2026-03-06-*` -- V2 rebuild: evidence-first design, convergence, hardening, cognitive orchestrator
- `2026-03-07-*` -- Audit response plan, Rust embedder plan, S-MMU wiring plan
- `comprehensive_knowledge_transfer.md` -- Full project knowledge transfer document

### `audits/` -- Audit Reports and Verification

External and internal audit results from three independent reviewers:

- `2026-03-06-opus-v3-self-audit.md` -- Opus 4.6 self-audit (V3)
- `2026-03-06-gpt54-design-review.md` -- GPT-5.4 design review
- `2026-03-06-gpt54-external-audit.md` -- GPT-5.4 external audit
- `2026-03-07-audit-verification.md` -- Verification of all 20 audit findings and fixes

### `benchmarks/` -- Benchmark Results

Machine-readable benchmark outputs with timestamps:

- `*-humaneval.json` -- HumanEval pass@1 results (summary)
- `*-humaneval.jsonl` -- HumanEval per-problem traces (truth pack)
- `*-routing.json` -- Routing accuracy results (30-task classification)
- `*-routing.jsonl` -- Routing per-task traces
- `*-summary.json` -- Aggregated summary reports

### `ADR-cognitive-routing-system2.md`

Architecture Decision Record for the S2 cognitive routing system design.
