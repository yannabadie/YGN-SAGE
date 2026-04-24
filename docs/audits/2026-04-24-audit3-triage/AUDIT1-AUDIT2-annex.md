# AUDIT.md + AUDIT2.md — triage annex (retroactive)
**Date:** 2026-04-24 · **Protocol breach acknowledgement + close**

## Pourquoi cet annex existe

PROMPT.md §1.1 dit : `ls AUDIT*.md` + "Lis chaque fichier en entier". Le
glob matche `AUDIT.md`, `AUDIT2.md`, `AUDIT3.md` — j'ai triagé
uniquement AUDIT3 en séance principale. Violation protocole §1.1 +
gate 1.

**Date reconstructions:**
- `AUDIT.md` — 68KB, écrit 2026-04-24 à 11:23
- `AUDIT2.md` — 18KB, écrit 2026-04-24 à 11:25
- `AUDIT3.md` — 18KB, écrit 2026-04-24 à 11:41

Tous les 3 cible commit `11b58c51b213e0d5375b8f35bd52ecfd689bce7c`
(avant mes 13 commits AUDIT3 de aujourd'hui).

## Stratégie de résolution

Given pace constraints + ~80% overlap between AUDIT/AUDIT2/AUDIT3,
this annex:
1. Cross-references each AUDIT.md/AUDIT2.md claim to existing triage
   artifacts (already-addressed / AUDIT3-ticketed / new)
2. Identifies **NEW actionable items** not covered anywhere
3. Proposes disposition per new item (quick-fix / ticket / no-fix)

## AUDIT.md claim mapping

| AUDIT.md claim (§3 table) | Status | Reference |
|---|---|---|
| "Agent Development Kit" | ✅ Partially supported | Self-admission; same as AUDIT3 #1 |
| "S1/S2/S3 routing" | ✅ Partially | Same as AUDIT3 #1 |
| "Multi-agent topology" | ✅ Partially | Same as AUDIT3 #1 |
| "Learns which topology to use" | ✅ Partially/overbroad | Same as AUDIT3 #1 (pipeline learn stage) |
| "Assigns models from 7 providers" | ✅ Partially | Same as AUDIT3 #1 |
| "Executes with formal verification" | ✅ Materially overclaimed | Same as AUDIT3 #3 + #8 (Fix 1 addresses naming) |
| "Formal verification includes LTL" | ✅ Partial / unsafe default | **Same as AUDIT2 "LTL model checking" + AUDIT3 #8 — Fix 1 addresses** `835eced0` |
| "4-tier memory / S-MMU" | ✅ Partially | Same as AUDIT3 §4 implicit |
| "Tool sandbox safety" | ✅ Partially | Addressed by ADR-013 §5 flip + Fix 2 HITL |
| "HITL, streaming, circuit breaker" | ✅ Partially | Hooks exist, cancellation not full |
| "Benchmarks show meaningful advantage" | ❌ Unsupported as SOTA | **NEW: doc caveat update needed** |
| "Training / learned policy exists" | ❌ Not verified | Same as CLAUDE.md — training parked, on HF |
| "ONNX / DistilBERT quality estimator" | ❌ Unsupported | **Already addressed by A0d `bf220e0` 2026-04-23** |
| "CI validates framework" | ✅ Partially | **NEW: inconsistent test counts** |
| "Protocol support: MCP/A2A" | ✅ Partially | **NEW: auth/conformance not verified** |
| "Production deployment ready" | ❌ Unsupported | **NEW: docs should align with alpha framing** |
| "Open-source maturity" | ✅ Partially | alpha self-declared |

**AUDIT.md §6 Security register (NEW claims not in AUDIT3):**

| Risk | Status | Notes |
|---|---|---|
| Dynamic tool creation registered by default | ✅ **Addressed by Fix 2 HITL** `3bdf9c43` + `SAGE_DANGEROUS_TOOLS=False` default |
| Sandbox downgrade / fallback execution | ✅ Addressed by ADR-013 §5 `c2113d8` |
| Repository mutation and test execution | ⚠️ partial — typed_repo.py has `apply_patch`/`run_tests` (path-jailed); no per-run sandbox checkout |
| Prompt/tool injection through retrieved context | ⚠️ **A13 already ticketed** (same as AUDIT3 #10) |
| **Secret leakage into logs or memory** | ⚠️ **NEW** — event bus stores payloads, episodic memory logs content prefixes; no redaction layer |
| Fail-open verification | ✅ **Addressed by A0b** `2bd966c` SAGE_STRICT_GOVERNANCE |
| Shared mutable runtime state | ✅ **Addressed by A0a** `9067be5` |
| **Supply-chain exposure** | ⚠️ **NEW** — Python deps unpinned; no cargo-audit/cargo-deny/pip-audit/Semgrep/Dependabot |
| **Protocol service exposure** | ⚠️ **NEW** — MCP/A2A gateway auth not verified |
| Cost explosion | ✅ **Addressed by Fix 3** `55a393c1` |

## AUDIT2.md claim mapping

| AUDIT2 claim (§3 table) | Status | Notes |
|---|---|---|
| "5 pillars" | ✅ Partially | Same as AUDIT3 #1 |
| "Rust + Python + discover" | ✅ Supported | Same as AUDIT3 |
| "kNN router 92% GT" | ✅ Partially | ground-truth has **60 tasks** (AUDIT2 correct); README/docs may say 50 — doc drift |
| **"Heuristic router dead code" CONTRADICTED** | ✅ **AUDIT2 evidence correct** — `pipeline.py:477` has `# Priority 3: AdaptiveRouter heuristic` fallback. CLAUDE.md calls it "emergency fallback only" which aligns with being-present-but-not-primary. **Rename in CLAUDE.md from "DEAD CODE" to "emergency fallback" would close the contradiction** |
| "6-path topology engine" | ✅ Partially | Same as AUDIT3 |
| "7 providers / 19 models" | ✅ Partially | cards.toml is authoritative |
| "Formal verification" | ✅ Partially / overclaimed | Fix 1 rename partial mitigation |
| "LTL model checking" | ✅ **Fix 1 addresses** — rename to GraphPropertyChecker `835eced0` |
| "3-layer sandbox" | ✅ Partially | Same as AUDIT3 #5 + #9 — post-ADR-013 §5 |
| "4-tier memory" | ✅ Partially | MemoryAgent graph-DB is "planned" label |
| "Learns from every run" | ✅ Partially | Same as AUDIT.md/AUDIT3 |
| "BigCodeBench 45.9% above SOTA" | ❌ Unsupported as SOTA | **NEW: doc caveat** |
| "SWE-bench claims" | ❌ Not verifiable as serious | **NEW: doc caveat** |
| **"Tool safety: traceback leaks"** | ✅ **Addressed by A0c** `684bb17` 2026-04-23 |

**AUDIT2.md §6 Risk register (NEW):**

| Risk | Status | Notes |
|---|---|---|
| Prompt/memory injection | ⚠️ Already ticketed as A13 |
| Tool traceback leakage | ✅ A0c `684bb17` |
| **Dynamic tool validation downgrade** | ⚠️ **NEW** — `forge.py:352-354` has `ast.parse()` fallback when Rust validator fails; fails open. Should fail-closed. |
| **Host execution fallback** | ⚠️ partial — `isolated_executor.py` has non-Linux subprocess path; **orphaned post ADR-013 §5** (not on default path) |
| Raw shell / repo mutation | ⚠️ partial — `execute_bash` default-off via `SAGE_DANGEROUS_TOOLS=False`; `apply_patch` in typed_repo path-jailed but not per-run |
| **Provider fail-open** | ⚠️ **NEW** — health check exception can leave providers as-alive |
| **Supply chain (PyPI trust / SHA-pinned actions)** | ⚠️ **NEW** (also in AUDIT.md) |
| **Search tool portability (rg unavailable)** | ⚠️ **NEW LOW** — targeted fallback missing |

## NEW actionable items (not in AUDIT3 + not already shipped)

Severity-scored per PROMPT.md §3.1:

| ID | Claim | Severity | Criterion | Effort |
|---|---|---|---|---|
| **N1** | Secret leakage in event bus + episodic memory (no redact layer) | **HIGH** | blast=system-wide logs/traces; exploit=any logged payload; freq=every event | ~1 week (design + impl) |
| **N2** | Supply-chain (unpinned deps, no audit tools, no SHA-pinned actions) | **HIGH** | blast=RCE via dep compromise; exploit=attacker-controlled upstream; freq=every build | ~3-5 days |
| **N3** | Dynamic tool validation downgrade (forge.py:354 AST-only fallback) | **HIGH** | blast=unsafe tool registered; exploit=Rust validator unavailable; freq=cold-start | ~1 day (fail-closed pattern) |
| **N4** | Protocol service exposure (MCP/A2A auth not verified) | **MEDIUM** | depends on if user exposes gateway publicly | ~1 week (auth + docs) |
| **N5** | Provider fail-open on health-check failure | **MEDIUM** | wrong-provider routing vs hard-fail | ~4 hours |
| **N6** | BigCodeBench/SWE-bench SOTA framing unsupported | **LOW** | credibility only | ~30 min docs |
| **N7** | Heuristic router "DEAD CODE" framing contradicts code (priority 3 fallback exists) | **LOW** | CLAUDE.md/README doc drift | ~15 min |
| **N8** | routing_ground_truth.json has 60 tasks, docs may say 50 | **LOW** | doc drift | ~15 min |
| **N9** | Repository mutation (apply_patch/run_tests per-run sandbox) | **MEDIUM** | blast=workspace; exploit=agent malice; freq=agent runs | ~3 days (per-run checkout) |
| **N10** | rg portability fallback missing | **LOW** | dev UX only | ~1 hour |

## Proposed roadmap additions

- **A16** — Centralized redaction layer for logs/events/memory (N1) — HIGH, ~1 week
- **A17** — Supply-chain security CI gates: pip-audit, cargo-audit, cargo-deny, Dependabot, SHA-pin actions (N2) — HIGH, ~3-5 days
- **A18** — Dynamic tool validation fail-closed (N3) — HIGH, ~1 day
- **A19** — Protocol gateway auth (N4) — MEDIUM, ~1 week
- Minor doc fixes (N6, N7, N8, N10) — can be done inline later

## Quick wins doable now (~30-60 min total)

1. **N7+N8 — doc drift fixes:** update CLAUDE.md + README references for heuristic router and routing_ground_truth task count
2. **N6 — benchmark framing caveat:** add "not SOTA on public leaderboard" disclaimer next to BCB 45.9% / SWE-bench claims

## Disposition final

**5 HIGH-severity NEW items** (N1-N3, N2) → add to roadmap as A16/A17/A18 tickets (multi-day each).

**3 LOW-severity doc fixes** (N6, N7, N8) → opportunistic inline fixes if session budget allows.

**1 LOW infra item** (N10) → roadmap minor backlog.

**Meta-audit of AUDIT.md + AUDIT2.md:**

- AUDIT.md is the MOST comprehensive of the 3 today's audits (68KB vs 18KB). It has broader scope but most specific claims overlap with AUDIT2/AUDIT3.
- AUDIT2 is tightest and focuses on security + verification.
- AUDIT3 is mostly a compressed version.

**For future AUDIT4:** header with commit-sha basis + explicit dedup against prior audit claims would save triage effort.
