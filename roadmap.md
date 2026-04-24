# YGN-SAGE roadmap

**Last updated:** 2026-04-23
**Scope:** forward-looking work surfaced by the 2026-04-23 session (Track 2+3
close-out, wasm JIT cache, pre-emission diff-context verifier, ALIRE audit
triage). Not a long-term strategy doc — a living backlog grouped by
expected time horizon. Priorities inside each horizon ordered by
impact-over-effort.

Reference frames:
- `ALIRE.md` — external audit of commit `44a157c` (2026-04-22). Several of
  its critical items landed this week (dangerous_tools flip, subprocess-
  fallback docs sweep, ADR-013); the remaining items are in this roadmap.
- `docs/benchmarks/2026-04-23-track3-closeout.md` — the Track 3 breadcrumbs
  pinned there map to items below.
- `docs/superpowers/specs/2026-04-23-diff-context-verifier-design.md` —
  spec that drove the 2026-04-23 observe-mode ship; repair-mode
  implementation is in the short-term section below.

---

## Horizon A — short-term (next 1-2 weeks)

### A0 — ALIRE2 triage remediations (2026-04-23)

The 2026-04-23 ALIRE2 verification (`docs/audits/2026-04-23-alire-verification.md`)
confirmed 4 high-severity live gaps on main. All 4 were shipped the
same session as small, reviewable commits — kept here so the audit
trail is explicit.

| Item | Shipped commit | Note |
|---|---|---|
| **A0a** Restore all 10 mutated AgentLoop fields in the pipeline bypass `finally`. Targeted fix; full refactor = B9. | `9067be5` | `pipeline.py` snapshots `_orig_bypass_state` dict pre-mutation; finally restores all 10 (happy + exception). +2 regression tests. |
| **A0b** `SAGE_STRICT_GOVERNANCE=1` turns the write-gate-init-failure and verification-failed paths into hard aborts (emit `EXECUTE_HALTED_UNVERIFIED`). Default off; byte-identical. | `2bd966c` | `pipeline.py` + `_is_strict_governance()` helper. +16 regression tests. |
| **A0c** Redact raw traceback from `Tool.execute()` model-visible output (info-leak fix, ALIRE2 §6). | `684bb17` | `tools/base.py` returns type:message; `log.exception` keeps operator-side visibility. +3 regression tests. |
| **A0d** Caveat the "DistilBERT ONNX QualityEstimator" claim in 6 docs — active backend is Z3 QualityLabeler + abstention; ONNX artefact NOT shipped. | `bf220e0` | README.md, pillars.md, results.md, methodology.md, Obsidian Papers/CascadeRouting.md + Architecture/Pipeline.md. |

### A1. Accumulate observe-mode data across opportunistic SWE-bench smokes
**Why:** repair-mode flip needs ≥10 flagged + ≥10 clean observations to
discriminate signal from false positives at a meaningful SE. Dedicated
smokes cost $5-10 apiece; instead, **every** future SWE-bench run opts
into observe mode passively.

**Progress (2026-04-24, N=20 cumulative):**

| Smoke | N | PATCH | EMPTY | Flagged | Verifier-missed fails |
|---|---|---|---|---|---|
| 2026-04-23 | 10 | 2 | 8 | 2/2 | 0 |
| 2026-04-24 | 10 | 2 | 8 | 1/2 | 1 (malformed hunk header) |
| **Total** | **20** | **4** | **16** | **3** | **1** |

**New finding (2026-04-24):** astropy-6938 emitted a patch that Docker
rejected with `malformed patch at line 15` because the hunk header's
old/new line counts don't match the diff body. The current verifier
walks context lines INSIDE well-formed hunks — it can't run the content
check if `patch` aborts at header parse. Adding hunk-header arithmetic
(old-line-count vs number of ` `/`-` lines; new-line-count vs ` `/`+`)
as a pre-check would close this blind spot with ~20 LOC.

**Concrete action:** update smoke invocations in docs + scripts (already
done in `.claude/rules/development.md` + `CLAUDE.md`) so new runs include
`SAGE_DIFF_VERIFIER_MODE=observe`. Two more passive smokes → N=40 with
~8 PATCHes. Consider verifier extension for the second failure class.

**Done when:** ≥10 PATCH entries observed AND at least one Docker-clean
PATCH exists in the sample. **Currently stuck at 0 clean PATCHes after
N=20** — the coder-role model emits unapplicable diffs deterministically.
This is an orthogonal concern to repair-mode coverage and may need its
own sub-item (see A6 below).

### A2. Fast-abort root cause — ✅ DIAGNOSED 2026-04-24 (Kimi k2.5 tool-call)
**Why:** astropy-14182 (~58 s) and astropy-7746 (~72 s) both aborted
with 0-ish tool calls in the 2026-04-23 AND 2026-04-24 smokes —
deterministic (N=2/2). Investigation via gen-log dive:

**Root cause chain (both tasks, identical signature):**
1. Rust `ModelAssigner` filters `needs_tools && !card.supports_tools`
   (`model_assigner.rs:289`). `needs_tools` comes from
   `node.required_capabilities.contains("tools")`.
2. In `sage-core/src/topology/templates.rs`, only the `coder` node
   declares `"tools"`; `planner`, `worker_N`, `synthesizer`, `source`,
   `mixer`, `dispatcher` etc. use `["text_processing"]` or
   `["reasoning"]` without `"tools"`.
3. Rust assigner picks kimi-k2.5 for a non-coder node (scoring +
   diversity penalty + affinity).
4. Python `AgentLoop` factory grants tools to EVERY node regardless
   of declared capabilities at execution time.
5. Kimi's Moonshot API rejects the 4th+ assistant message carrying
   tool_calls with `HTTP 400 "thinking is enabled but
   reasoning_content is missing in assistant tool call message at
   index 3"`. `cards.toml:598-609` already documents this (F9 audit
   fix marked `supports_tools=false`) — but the filter only fires
   when `needs_tools=true` from the node caps.
6. Stage 4 multi-agent fails → single-agent fallback to
   gemini-3.1-flash-lite-preview returns empty content → task
   aborts as EMPTY.

**Fix ticketed as A7** (next entry); the diagnosis itself is done.

**Impact:** closes the 20% silent-budget-leak once A7 lands. All
2026-04-23 and 2026-04-24 smokes' per-task signal was masked by this
bug.

### A7. Template capability hygiene (fix for A2) — ✅ SHIPPED 2026-04-24
**Why:** A2's Option A. Prevent kimi-k2.5 from being assigned to
tool-using nodes by declaring `"tools"` in every non-sink template
role's `required_capabilities`.

**Action taken:**
- Added `"tools"` to 23 nodes across all 12 templates in
  `sage-core/src/topology/templates.rs`. Each edit carries an
  inline comment pointing to A7's rationale and the model_assigner
  filter site.
- Sink-prompted roles (SINK_NODE_PROMPT) deliberately kept
  tool-free:
  - `synthesizer` (sequential, brainstorming)
  - `aggregator` (parallel, horizon_pipeline, parallel_fanout)
  - `mixer` (selfmoa)
  - `judge` (debate)
  - `verifier` in `robust` only (AVR's `verifier` gets `"tools"` —
    it's not sink-prompted)
  - `solver` (formal_solver — deterministic Rust, not LLM)
- New regression test `test_a7_tool_capability_hygiene_all_templates`
  in `templates.rs`. Iterates every template; asserts non-sink
  nodes declare `"tools"` AND sink nodes do NOT. Fires with a
  pointer to A7's rationale if a future template adds a role
  without `"tools"` or regresses an existing one.

**Validation:** Rust `cargo test --features smt --lib topology::templates`
→ 20/20 PASS (+1 new A7 test). The Rust-level filter at
`model_assigner.rs:289` (`needs_tools && !card.supports_tools`)
now fires on every template-built non-sink node, excluding kimi-k2.5
from the candidate pool. No Python-side change needed — the existing
`cards.toml:610 supports_tools = false` on kimi is the other half
of the contract.

**Option B (correct long-term):** make AgentLoop honour the node's
`required_capabilities` — tool-free roles get a tool-free agent
variant. Matches F9's original intent. Wider refactor
(agent_loop_factory + phases/act.py). Do at B9 (AgentLoop
concurrency refactor) — until then, declaring `"tools"` universally
(except SINK_NODE_PROMPT roles) is the pragmatic close-out.

**Expected impact:** closes A2's 20% silent-budget-leak on
template-built multi-agent topologies. Next opportunistic observe
smoke should see astropy-14182 / astropy-7746 no longer fast-abort.
A1's patch-rate gate becomes achievable since every task gets real
tool-call attempts rather than HTTP 400s on turn 4.

**Empirical verification (2026-04-24, N=6 gen-only smoke on the same
task IDs as the pre-A7 smoke).** See
`docs/benchmarks/2026-04-24-a7-verification/findings.md`.

- **Zero kimi-k2.5 HTTP 400 occurrences** in the 82 KB gen log (vs 2
  in yesterday's pre-A7 smoke).
- **astropy-7746** (canonical fast-abort): `EMPTY 72 s` → `PATCH 634 chars` ✅
- **astropy-12907**: `EMPTY 266 s` → `PATCH 4879 chars` ✅
- **astropy-14995**: `EMPTY 174 s` → `PATCH 474 chars` ✅
- **astropy-14182**: still 0 chars, but now `Generation timed out`,
  not fast-abort. kimi-400 path closed; residual failure is a
  different class (agent-loop convergence / gen-timeout). Separate
  investigation — NOT an A7 regression.
- **Patch rate: 4/6 = 67%** (vs 20% pre-A7 on the same N=10 superset).

Secondary side effect (documented, not a regression): running the
smoke in parallel with the rustpython wasm source build triggered
`WinError 1455` (Windows paging file exhaustion) on astropy-7746's
`git apply --check`. The patch was still written to predictions.jsonl
(634 chars survived); only the repair-validator phase was
short-circuited. Future large parallel runs should stagger wasm
builds vs benches.

### A8. Migrate kimi-k2.5 → kimi-k2.6 in cards.toml — ✅ SHIPPED 2026-04-24 (Phase 1: model id + specs)
**Why:** user flagged that Moonshot's latest is **kimi-k2.6**, not
k2.5. cards.toml's `kimi-k2.5` entry carries `supports_tools = false`
from the F9 audit — the documented reason is Moonshot's
`reasoning_content` API requirement on the 4th assistant tool-call
turn with `thinking` enabled (the exact symptom that caused A7 to
exist). If k2.6 resolved that contract, we could flip
`supports_tools` back to `true` after migration and regain kimi as a
tool-capable routing option. A7 stays load-bearing either way (it's a
belt-and-suspenders capability hygiene fix) but the kimi-k2.6 move is
a potential net win on routing diversity + cost.

**Action taken (Phase 1):**

1. Fetched https://platform.kimi.ai/docs/guide/kimi-k2-6-quickstart
   (Directive #6 — live docs, not training data).
2. Verified via same doc: k2.6 inherits k2.5's `reasoning_content`
   requirement on multi-turn tool calls when thinking is enabled
   (which is the default). Same HTTP 400 class as before.
3. Updated `sage-core/config/cards.toml` + mirror in
   `sage-python/config/cards.toml` (no longer a symlink on Windows —
   must sync manually):
   - id: `kimi-k2.5` → `kimi-k2.6`
   - family: `kimi-k2.5` → `kimi-k2`
   - context_window: `128000` → `262144` (256K per docs; old was
     stale pre-k2.5 carryover)
   - affinity scores bumped +0.02 (code/reasoning/tools/math)
     reflecting "Kimi's latest and most intelligent" positioning —
     subject to ablation
4. `supports_tools = false` STAYS until the reasoning_content
   plumbing is fixed. Path forward (still open, ticketed below):
   - (a) plumb `thinking: {type: "disabled"}` in our PydanticAI
     wrapper whenever the request carries tools. k2.6 supports the
     disable toggle per its quickstart.
   - (b) OR fix our wrapper to preserve reasoning_content across
     turns. PydanticAI has a `MoonshotAIProvider` class
     (https://pydantic.dev/docs/ai/api/pydantic-ai/providers/) but
     the docs don't confirm reasoning_content passthrough — needs
     a direct code inspection.
5. Updated substring quirks in `openai_compat.py`: temperature+top_p
   hard-strip now matches `k2.6` / `k2-6` / `k2-thinking` variants
   alongside existing k2.5 patterns (same quirk, documented by the
   k2.6 quickstart "temperature and top_p fixed when thinking
   enabled").
6. Updated 6 call-sites with hardcoded `kimi-k2.5`: connector.py
   default_model, 2 test fixtures, test_provider_quirks cases
   (renamed `test_kimi_k25_strips_temperature` →
   `test_kimi_k26_strips_temperature_and_top_p` + kept a
   back-compat test for callers pinning legacy id), models.toml
   comment, test_pydantic_ai_integration
   (`test_kimi_k2_5_supports_tools_is_false` →
   `test_kimi_k2_6_supports_tools_is_false`).

**Validation:** 30/30 Python quirks + provider-integration tests
PASS; 20/20 Rust template tests PASS.

**Phase 2 (pending, tracked as sub-items):**
- Investigate PydanticAI's `MoonshotAIProvider` source to determine
  if reasoning_content is preserved across turns. If yes, SAGE's
  wrapper is the bug; swap wrapper → flip `supports_tools = true`.
- If no, plumb `thinking: {type: "disabled"}` in our provider layer
  for kimi-k2.6 whenever tools are present. Ablation smoke required
  to confirm disabling thinking doesn't regress reasoning quality.

**Note on user's concern ("PydanticAI should handle this"):**
The user's intuition is sound — PydanticAI has a dedicated Moonshot
provider. The observed failure was through our SAGE `PydanticAIProvider`
wrapper. The A8 Phase 1 change is compatible with either future
resolution (wrapper fix OR thinking-disable plumbing); the cards.toml
flag gates tool use until Phase 2 lands.

### A10. search_repo size-cap against MemoryError — ✅ SHIPPED 2026-04-24
**Why:** astropy-14182's post-A7 `Generation timed out` failure
(investigated 2026-04-24) was caused by `search_repo`'s Python
fallback calling `p.read_text(encoding="utf-8", errors="ignore")` on
every file in the repo — including a big data file that exhausted
process memory with `MemoryError`. The exception propagated up to
the agent loop as a tool error; the agent retried; D8 stall cap
fired; task timed out with 0 chars. Orthogonal to A7 (kimi fast-
abort) — this is a large-file-in-repo class.

**Action taken:**
- `sage-python/src/sage/tools/typed_repo.py:_build_search_repo_tool`
  Python fallback path now: (1) `_MAX_FILE_BYTES = 1 MiB` size gate
  via `p.stat().st_size` before reading, (2) `MemoryError` added to
  the per-file `except` clause so a single pathological file can't
  abort the whole scan.
- Regression test `test_search_repo_skips_large_files` — creates a
  1.5 MiB file containing the search term + a small file containing
  the same term, asserts small file IS found AND large file is
  skipped.

**Validation:** 53/53 typed_repo tool tests PASS.

**Not touched:** the ripgrep fast-path (the primary happy path for
any machine that has rg installed) doesn't hit this bug — it uses
memory-mapped search and streams lines. The fix is specifically for
the Python fallback.

### A11. DeepSeek v4 migration — ✅ SHIPPED 2026-04-24 (Phase 1: cards + default_model)
**Why:** https://api-docs.deepseek.com/ (fetched 2026-04-24) announces
**deprecation of `deepseek-chat` and `deepseek-reasoner` on
2026-07-24** (3 months from today). They become the non-thinking and
thinking modes of `deepseek-v4-flash`. Also ships
`deepseek-v4-pro` as the high-accuracy thinking variant.

**Action taken:**
- `sage-core/config/cards.toml` + `sage-python/config/cards.toml`
  (sync'd manually — Windows symlink broken):
  - **NEW** `deepseek-v4-flash`: non-thinking, 1M context, $0.14
    cache-miss / $0.28 output per 1M. `supports_tools=true`.
  - **NEW** `deepseek-v4-pro`: thinking-mode, 1M context, $1.74
    cache-miss / $3.48 output per 1M. `supports_tools=true`
    provisionally (thinking-mode reasoning_content quirk is the
    same class as kimi-k2.5/k2.6; if a production smoke hits 400s,
    flip to false like we did for kimi — same template filter
    handles it via A7).
  - **LEGACY** `deepseek-chat` + `deepseek-reasoner`: kept until
    sunset (2026-07-24) so bandit history + pinned fixtures keep
    working. Block comments mark them LEGACY.
- `connector.py::default_model` for deepseek: `deepseek-chat` →
  `deepseek-v4-flash`.
- `test_live_multiprovider.py` matrix: added v4-flash + v4-pro
  entries, kept legacy rows with LEGACY comment.
- Pricing source: https://api-docs.deepseek.com/quick_start/pricing
  (fetched 2026-04-24). Affinities seeded from the matching legacy
  entry + research-backed bumps; ablation-subject.

**Validation:** 30/30 provider + quirks + assigner tests PASS.
cards load cleanly via PyO3 ModelRegistry.from_toml_file; 7 models
spot-checked (cost + context + tools flags match live docs).

**Phase 2 (pending, tracked here):**
- **v4-pro thinking quirk first-run triage.** First production
  smoke that assigns v4-pro to a tool-using node will reveal
  whether PydanticAI's MoonshotAIProvider-style
  `send_back_thinking_parts='field'` contract covers DeepSeek too,
  or if we need the same plumbing we're ticketing for kimi
  (A8 Phase 2). If 400s appear: flip `supports_tools=false` on
  v4-pro, A7 filter excludes it from tool-using nodes, plumbing
  work becomes the unblocker.
- **Pre-sunset flip.** Before 2026-07-24, swap test fixtures that
  pin `deepseek-chat` → `deepseek-v4-flash`, remove LEGACY blocks
  from cards.toml, and kill the deepseek-chat / deepseek-reasoner
  entries from live-provider matrix. Could schedule a /schedule
  agent to open that cleanup PR around 2026-07-01.

### A9. Investigate gpt-5.5 — ✅ SHIPPED 2026-04-24 (Phase 1: cards)
**Why:** user flagged OpenAI released gpt-5.5 as a new model. Our
cards.toml ships gpt-5.4, gpt-5.4-pro, gpt-5.4-mini, gpt-5.4-nano,
gpt-5.3-codex. If gpt-5.5 is a measurable improvement over 5.4 on
reasoning/code, adding it (or replacing 5.4 as the default reasoner
tier) could materially improve SWE-bench and BCB results.

**Action taken (Phase 1):**
1. WebSearch-verified against live OpenAI sources (2026-04-24):
   - https://openai.com/index/introducing-gpt-5-5/ (announcement)
   - https://developers.openai.com/api/docs/models/gpt-5.5 (API ref)
   - https://developers.openai.com/api/docs/pricing (pricing)
   - Wikipedia gpt-5 page also lists 5.5 as a successor, confirming.
2. Added to cards.toml (both sage-core + sage-python mirrors):
   - **gpt-5.5**: $5/$30 per 1M tokens, 1M context, supports tools +
     vision + json + responses API. Affinities +0.01 above gpt-5.4
     on reasoning/code — conservative, ablation-subject.
   - **gpt-5.5-pro**: $30/$180 per 1M, 1M context, pro-tier
     affinities. S2 affinity lowered to 0.60 to prevent pro-rate
     assignment on mid-tier tasks.
   - gpt-5.4 + gpt-5.4-pro kept intact; bandit-managed over time.
3. `test_live_multiprovider.py` matrix: gpt-5.5 + gpt-5.5-pro rows
   added before gpt-5.4 entries.

**Validation:** Cards load cleanly via PyO3 ModelRegistry
(cost_in=$5.0, ctx=1M for gpt-5.5; $30.0/1M for pro). 30/30 provider
+ quirks tests PASS.

**Phase 2 (tracked here, not shipped):**
- Live-call smoke against gpt-5.5 once OPENAI_API_KEY holder has
  access (model availability at API level not verified in-session
  — only via docs).
- Ablation smoke (N=10) comparing gpt-5.5 vs gpt-5.4 on BigCodeBench
  Hard / SWE-bench to validate affinity seed values.
- Bandit calibration sweep (N=50 routing_gt) to confirm bandit
  doesn't over-select 5.5 on tasks where 5.4 remains adequate.

### A3. Repair-mode implementation — ✅ SHIPPED 2026-04-24

### A3. Repair-mode implementation — ✅ SHIPPED 2026-04-24
**Why:** spec § "Validation plan" — repair mode feeds the diff-verifier
mismatch diagnostic to an LLM one-shot repair. The 2026-04-23 observe
smoke confirmed the mismatch signal is clean (zero false positives on
two patches); 2026-04-24 A7 verification accumulated more observe data
(4 PATCH with verifier-flagged failures + 0 clean).

**Action taken:**

- `sage-python/src/sage/bench/swebench_diff_verifier.py`: new
  `repair_with_verifier_feedback(llm, problem_statement, broken_patch,
  mismatches, instance_id, timeout) -> (new_patch, stage)` async
  function. Builds a structured repair prompt that shows per-hunk
  EXPECTED-vs-ACTUAL line text (truncated to 20 lines × 200 chars
  per section to keep the prompt bounded). Returns stages
  `"verifier_repair"`, `"verifier_repair_empty"`, or
  `"verifier_repair_skipped"`.
- `_get_diff_verifier_mode`: "repair" no longer downgrades. Invalid
  modes still fall back to "off" with a WARN.
- `generate_patches`: when `verifier_mode == "repair"` AND
  `mismatches` is non-empty AND an LLM handle exists, the verifier-
  repair call runs BEFORE `try_repair_patch`. If it succeeds
  (`stage == "verifier_repair"`), the corrected diff replaces the
  original patch; `try_repair_patch` then sees the corrected input
  (and may still apply CRLF normalization / counts-fix / git-apply
  LLM-repair on top). The combined repair_stage is serialised as
  `"verifier_repair+<downstream>"` so operators can see the full
  chain.
- Observe-mode behaviour unchanged — annotations are populated in
  both observe AND repair modes, so the existing bucket-analysis
  scripts work on repair-mode predictions without modification.

**Tests (new):**
- `test_diff_verifier_repair_calls_llm_with_mismatch_feedback`
  in `test_swebench_emission_wiring.py` — end-to-end wire test:
  builds a real git repo, serves a canned agent emission with
  wrong context lines, installs a spy LLM that returns a corrected
  diff. Asserts: (a) observe-like annotation, (b) LLM called
  exactly once with a prompt naming the mismatched file AND the
  actual file contents, (c) the corrected patch propagates to the
  prediction dict.

**Test suite:** 48/48 verifier + repair + emission-wiring tests
PASS. The legacy `test_diff_verifier_repair_warns_and_downgrades_to_observe`
was replaced by the new wire test (same contract, updated
behaviour).

**Next (tracked here, not yet done):**
- **Paired N=50 smoke** (observe-only vs observe+repair) per the
  spec § "Validation plan". Measures whether repair-mode actually
  improves Docker pass-rate on flagged patches, or just generates
  new kinds of invalid diffs. Cost: 2 × 30-minute N=50 runs ≈
  $15-20 total. Gated on: stable A1 observe data (currently at
  N=20 cumulative, 4 PATCH — enough to run the pair if user wants
  signal quickly, but larger-N paired runs give more statistical
  power).
- **Repair-mode coverage of the malformed-header class (A6).**
  CRLF-normalized diffs with correct content but wrong counts are
  handled by `try_repair_patch`'s programmatic-counts stage.
  Content-mismatch is handled by A3's verifier_repair. Both
  classes are now covered; no residual is known.

### A6. CRLF normalization in patch emission — ✅ SHIPPED 2026-04-24
**Why (initial framing):** astropy-6938 in the 2026-04-24 smoke was
rejected by Docker `patch` at line 18. Surface reading suggested a
malformed hunk header (`@@ -1541,10 +1541,4 @@`) where the header
counts didn't match the body.

**Actual root cause (byte-level inspection):** the patch file had
`\r\n` line endings throughout. `git apply` and GNU `patch` both
reject `\r` bytes in diff bodies as "corrupt patch". The repair
pipeline (`swebench_patch_repair.try_repair_patch`) DID run on this
patch — both stages failed with the same CRLF-corruption error. The
existing `_fix_hunk_header_counts` works on whole lines and preserves
the embedded `\r`, and the LLM repair stage also returned a CRLF
patch (the prompt carried the CRLF in the example body). So the
two-stage repair was insufficient to fix a CRLF-only bug.

Source of the CRLF: Windows `open(path, "w")` text-mode translates
every emitted `\n` into `\r\n`. The chain is:
1. Agent emits LF patch → `_extract_patch_from_response` normalizes
   CRLF → LF at line 216 ✓
2. `write_predictions` writes with default `open(...,"w")` → CRLF
   line terminator between JSON records, inside each record the
   patch stays JSON-escaped as `\\n` ✗ (the fault)
3. SWE-bench harness reads predictions.jsonl, extracts
   `model_patch` string (LF only in memory)
4. Harness writes `patch.diff` inside Docker context via text-mode
   open → CRLF again ✗
5. Docker invokes `patch` which aborts

**Shipped fix:**
- `swebench_patch_repair.py`: new `_normalize_line_endings()` helper;
  `try_repair_patch` gains a Stage 0 CRLF normalization before
  validation. New `"crlf_normalized"` stage value for telemetry.
  Belt-and-suspenders — even if upstream writes CRLF, the repair
  pipeline now starts from LF-only bytes.
- `swebench_bench.write_predictions`: now uses
  `open(path, "w", encoding="utf-8", newline="")` so line terminators
  are bare LF on all platforms. JSON-escaped patch body stays LF by
  construction.

**Tests (4 new, all passing):**
- `test_normalize_crlf_to_lf`
- `test_normalize_bare_cr_to_lf`
- `test_normalize_noop_on_lf_only`
- `test_repair_crlf_normalization_resolves` (end-to-end through
  `try_repair_patch` with a real git repo)
- `test_write_predictions_writes_lf_only_line_endings` (byte-level
  assertion on emitted JSONL)

**Impact:** closes the astropy-6938 failure class. Future observe
smokes should see repair-stage breakdown including
`crlf_normalized` for Windows-emitted patches. A3 repair-mode
implementation no longer needs a separate malformed-header class
— the existing `content_mismatch` covers all true hallucination
cases once CRLF noise is stripped.

**NB:** the original A6 framing ("extend the verifier to detect
malformed hunk headers") was wrong. The verifier's docstring
explicitly defers structural issues to `swebench_patch_repair`;
that deferral was correct. The real gap was upstream — a platform-
emission hygiene bug, not a verifier coverage gap.

### A4. Public-claims reconciliation (README ↔ PyPI)
**Why:** ALIRE flagged a three-way divergence (README/commits/PyPI) on
test counts. README has been refreshed (commit `be2d3fc`), but the
published PyPI `ygn-sage 0.1.0` still carries the March description.
Publishing a new PyPI patch release (`v0.1.1`) would sync, at the cost
of a minor release cycle.

**Concrete action:** bump patch version, refresh classifiers, push to
PyPI via the existing release workflow if one exists (grep for it)
or add a brief `docs/RELEASE.md`.

### A5. Fix the `ss.F.` anomaly captured in `tasks/bsc5mma67.output`
**Why:** the test-ignore ceiling drifted below tests and landed (commit
`5efdd42`). The original failing run wasn't labelled and ceased to exist
in the task runtime — attribute the output to a session context for
next time, or wrap long-running pytest invocations in a `TaskCreate` so
the task ID survives.

**Cost:** trivial, mostly a workflow hygiene task.

---

## Horizon B — medium-term (next 1-2 months)

### B1. OpenTelemetry GenAI span integration
**Why:** EventBus is in-process and lossy under backpressure (ALIRE
finding). OpenTelemetry published a GenAI semantic-conventions spec
(Development stability as of 2026-Q1) covering LLM request/response
attributes (`gen_ai.provider.name`, `gen_ai.operation.name`,
`gen_ai.request.model`, `gen_ai.usage.{input,output}_tokens`,
`gen_ai.response.finish_reasons`). That's the standard contract we'd
emit.

**Concrete action:** pick an OTel Python SDK (opentelemetry-sdk +
opentelemetry-exporter-otlp). Wire provider-side emission in
`sage.providers.pydantic_ai_provider` (and deprecated openai_compat
while it lives). Initial target: every LLM call emits a span with the
minimum-required attributes above. Deferred: tool-call spans, MCP
conventions. Docker harness span path is separate.

**Dependencies:** none blocking. Would benefit from B2 (durable store)
to land first, but they can proceed in parallel.

### B2. Durable trace + deterministic replay harness
**Why:** ALIRE high-severity item, and it's the prerequisite for any
serious formal-verification work (ALIRE's "runtime assurance" direction).
Every LLM call + tool call + topology decision + memory write gets a
trace ID and payload hashed; replay reconstructs the exact decision
sequence under mocked providers/tools.

**Concrete action:** start with an in-memory JSONL trace schema (one
line per event, typed). Ship the writer first; deserialisation + replay
later. Target: SWE-bench bench can reproduce a prior run's decisions
when replayed on mocks.

**Cost:** significant (~2-3 weeks). Defer if A1-A3 uncover a smaller
lever.

### B3. ToolPolicy capability manifest
**Why:** ALIRE critical item. Current `ToolRegistry` exposes
register/list/describe but no capability labels, side-effect contracts,
or approval requirements. That's a prompt-injection exposure.

**Concrete action:** new `ToolSpec` field `capabilities:
list[Capability]` + `data_access: DataScope` + `approval_required: bool`.
Policy check inserted in `execute_tool_call` before the tool runs.
Migrate all builtin tools (`execute_bash`, `read_file`, `search_repo`,
`apply_patch`, etc.) to declare capabilities. Default-deny on new tools
without a manifest.

**Dependencies:** none. Can ship incrementally (typed tools first, then
generated ToolForge tools).

### B4. Platform wheels with Rust core on PyPI
**Why:** ALIRE high severity. Current PyPI wheel is pure-Python;
`pip install ygn-sage` doesn't get the Rust extension. Either fix the
install story or clearly scope the PyPI package as "Python bindings —
build Rust from source".

**Concrete action:** extend the release workflow with a `maturin build
--release` matrix (Linux/macOS/Windows) and upload via `twine`. Include
the `rustpython.wasm` artefact (or build it in CI) so `SAGE_REQUIRE_WASM=1`
passes. Decide whether the wasm artefact goes in the wheel (37 MB bloat)
or a separate package (`ygn-sage-sandbox`).

**Cost:** 1 week. Mostly CI work.

### B5. CI job that builds wasm + enforces `SAGE_REQUIRE_WASM=1`
**Why:** the 2026-04-23 `SAGE_REQUIRE_WASM` gate (commit `cf188df`)
turns missing `rustpython.wasm` into a build error, but no CI job
exercises that path. A release pipeline that doesn't verify the wasm
artefact is the same failure mode in a nicer suit.

**Concrete action:** add a `.github/workflows/sandbox-build.yml` job
that clones RustPython, builds `wasm32-wasip1`, caches it, then runs
`SAGE_REQUIRE_WASM=1 cargo build --features sandbox`. ~20 min added to
CI (cacheable via `actions/cache`).

### B6. Fast-abort root cause fix (depends on A2 findings)
**Why:** whatever A2 surfaces, fixing the 20% fast-abort rate has
proportional payoff on every SWE-bench smoke.

### B8. Finish or de-scope the Rust controller orchestration entry — ✅ CLOSED 2026-04-23 (de-scope)
**Why:** ALIRE2 verification (A0a companion, 2026-04-23) confirmed
`RustTopologyController::evaluate_and_decide` returned `None` (scaffold
stub). Per-path methods (`check_empty_error_reroute`,
`check_quality_cascade`, `check_parallel_inconsistency`,
`check_importance_prune`, `is_in_gate_band`,
`should_trigger_emergent_spawn`) ARE populated and ARE called from
Python's `TopologyController`. "Rust-primary since 2026-04-20" (ADR-012)
was accurate for each decision path, but misleading if read as
"complete Rust control plane".

**Decision (2026-04-23):** de-scope + delete. Advisor (built-in
reviewer) and codex `gpt-5.4` at `model_reasoning_effort=high` both
converged independently:

- The orchestration cascade depends on Python-resident subsystems
  (embedder, SmtVerifier access, topology-graph `predecessors`, gate
  management, upgrade-model resolution). A Rust top-level wrapper
  would be a thin PyO3 shim back into Python — no measurable win.
- No current or near-term scenario (2026 horizon) shows the
  Python↔Rust crossing as a hot path. Benchmarks back this up.
- Renaming the stub preserves a misleading artifact; deletion makes
  the Rust/Python boundary explicit and matches current behaviour.

**Action taken:**

- **Deleted** `RustTopologyController::evaluate_and_decide` from
  `sage-core/src/topology/controller.rs`.
- **Updated** the module doc comment to state "Rust primitives that
  back the Python `TopologyController` … Python orchestrates which
  primitive to invoke".
- **Updated** ADR-012 with an amendment replacing "Rust-primary
  controller" with: `RustTopologyController` is Rust-primary for
  adaptation state and per-path decision primitives; orchestration
  remains Python-owned where decisions depend on Python-resident
  subsystems (embedder, SMT feedback access, topology graph, gate
  management, upgrade-model resolution).
- **Test contract**: `test_rust_topology_controller_exposes_per_path_primitives`
  (replacing `test_rust_topology_controller_evaluate_scaffold_returns_none`)
  asserts per-path primitives exist AND `evaluate_and_decide` is
  deliberately absent — regression guard against future resurrection.

**Effort:** ~2h (estimate matched).

### B9. Per-run immutable execution context (AgentLoop refactor)
**Why:** A0a (shipped `9067be5`) fixes incomplete restoration of the
10 mutated fields in the bypass path's `finally`. That closes
state-bleed under serial nested calls, but NOT under concurrent ones —
two `run()` coroutines on the same singleton AgentLoop would still
race on the shared fields between the snapshot and the `finally`.

The architectural fix is stop mutating a shared object entirely: build
a fresh, immutable execution context per run. References ALIRE2 §4.

**Estimated effort:** 1-2 weeks. Don't start until A1/A3 (observe +
repair mode) stabilise; this is a big refactor and interleaving it
with Track 3 work multiplies risk.

### B7. Test count drift gate automation
**Why:** the `test_mypy_count.py` ceiling drifted by 5 between commits
over weeks (caught in this session). The ceiling-bump flow is manual
and error-prone.

**Concrete action:** add a pre-commit hook or CI step that, on ceiling
drift, auto-generates a commit that lists the new ignores and their
source commits for review. Or: replace the ceiling test with a
per-file baseline (count per module is allowed; new modules must
declare).

---

## Horizon C — longer-term (3-6 months)

### C1. Semantic LLM verifier (option C from the diff-context spec)
**Gated on:** diff-verifier repair mode shipping (A3) and NOT moving
resolved rate ≥ 4 pp at N=50.

**Why:** the 2026-04-23 Track 3.1b analysis found 3 orthogonal
semantic-miss failure modes (context hallucination, over-aggressive
scope, wrong-layer coercion). The content-verifier addresses mode 1.
Modes 2 and 3 would need an LLM pass asking "does this diff actually
fix the problem". Expensive (+ $0.02-0.10/task) but the only
prompt-level remediation for mode 3.

### C2. Domain-knowledge injection (Track 3 breadcrumb #2)
**Why:** django-10924's `FilePathField.formfield` wrong-layer fix
requires knowing about Django's migration autodetector semantics. Not
a library-API contract (so `lookup_library_docs` doesn't help),
not a research-paper question (so `search_exocortex` doesn't help).
Could be:
- domain-specific knowledge bases seeded at boot (Django, Flask,
  numpy, etc.)
- or a pre-emission "is there framework-specific convention I'm
  missing?" LLM meta-question

Open research question. Keep a backlog note; implement only if observed
more than once in tracers.

### C3. External benchmark harness (GAIA, AgentBench, τ-bench, SWE-bench-Live)
**Why:** ALIRE high severity. Internal benchmark docs are candid but
don't substitute for external baselines. GAIA, AgentBench, and
τ-bench are the credible frames for agent evaluation; SWE-bench-Live
is the contamination-resistant next generation of SWE-bench-Verified.

**Cost:** multi-week. Likely need a dedicated harness module + artefact
store.

### C4. Runtime assurance layer (ALIRE "highest-leverage change" tier)
**Why:** pre/post-conditions on every tool call, data-flow constraints,
policy enforcement before side effects. Builds on B3 (ToolPolicy) and
B2 (durable trace).

### C5. Model-checked orchestration specs (TLA+/Alloy)
**Why:** ALIRE "world-class" tier. Multi-agent coordination,
cancellation, retry semantics captured in a formal spec. Tests verify
the runtime traces match the spec.

**Cost:** very high. Consider only once B2 + B3 land; speculative
otherwise.

---

## Open questions (breadcrumbs)

These are design ambiguities without a decisive experiment yet. Kept
here so they don't get lost.

1. **Is 0.95 really the right fuzzy-threshold for the diff-context
   verifier?** Spec correction `3c3fc27` narrowed the fuzzy branch to
   whitespace-only equivalence, with 0.95 retained as an observability
   hint. If the observe bucket accumulates fuzzy-below-threshold cases
   that correlate with whitespace drift we didn't anticipate, the
   threshold may need tuning or the branch may need another narrowing.

2. **Is `search_exocortex` actually useful in SWE-bench?** Prompt
   hygiene commit `29987bc` reframed it from anti-affordance to
   positive use case, but a follow-up audit hasn't measured whether
   usage went from 0 calls to non-zero. Worth a one-line grep on the
   next SWE-bench gen log: does `tool.call name=search_exocortex`
   appear anywhere?

3. **Module::serialize cache vs wasmtime built-in cache.** The
   2026-04-23 JIT cache (commit `50b4ee8`) hand-rolls a single-file
   .cwasm artefact. wasmtime's `Config::cache_config_load_default()`
   is the official built-in alternative. If we find ourselves adding
   cache-management complexity (invalidation rules, concurrent-writer
   safety beyond the current tempfile+rename, multi-module caches),
   consider migrating to the built-in.

4. **Why 8/10 EMPTY is the norm on SWE-bench Lite smokes.** Covered
   by A2 above but worth repeating as a meta-question: is this a
   model/capability ceiling, an infra issue (circuit breaker), or a
   prompt issue? The answer shapes which roadmap items are
   prioritised next.

5. **Do we need a dedicated prompt-injection red-team corpus?** The
   2026-04-22 wasm sandbox red-team (40/40 passing) validated the
   capability layer; nothing equivalent exists for prompt injection
   via the agent's tool-call path. ALIRE flagged this as high; hasn't
   surfaced as a concrete incident yet.

---

## Dropped / superseded directions

* **SWE-bench parity smoke at N=50** — the `±2 pp` statistical gate was
  below the N=50 SE ceiling (noise floor ≈ 10 pp/task; combined arm-gap
  SE ≈ 2 pp at N=50). Replaced by the functional criterion (typed-only
  produces patches) which landed 2026-04-22.
* **Track 3.2 "read test files first" prompt addition** — invalidated
  by the Track 3.1 finding that agents DO read test files on all three
  tracers.
* **Track 3.5 dedicated N=50 paired smoke** — deferred at the 2026-04-23
  close-out; no prompt-level lever had a lift hypothesis worth the $30-50
  spend.
* **Noise-floor calibration via paired identical-config runs** —
  advisor correctly flagged this as "measurement theater"; noise floor
  can be computed post-hoc from any future N=50+ smoke via resampling.

* **ALIRE2 §6 sandbox claims E/F/G (subprocess-based execution paths)**
  — accurate-as-described in code but **orphaned** post ADR-013 §5
  flip (`c2113d8`, 2026-04-22). `sandbox/manager.py` +
  `sandbox/isolated_executor.py` remain in the tree as legacy fallback
  that is unreachable on the default path (`allow_local=False`, Wasm
  first, hard-fail if Wasm absent). `sage-core/src/sandbox/subprocess.rs`
  gated behind `SAGE_UNSAFE_RAW_EXEC=1`. Deletion of the Python-side
  orphan files is tidy-but-not-urgent — ticketed as a separate B-tier
  cleanup; see `docs/audits/2026-04-23-alire-verification.md` for the
  divergence note.

* **ALIRE2 §3 "heuristic extraction in QualityEstimator" (claim I)**
  — refuted by code inspection. No regex / length / keyword signals in
  `quality_estimator.py`. Path is Z3 + (optional ONNX) + None
  abstention. The ONNX-not-shipped half of the claim IS true (handled
  by A0d docs sweep, `bf220e0`).

---

## Horizon pacing

Roughly: two Horizon A items per week; one Horizon B item per month;
Horizon C items tracked but not worked on until a B item forces the
issue. Do not lump A and B in the same commit; ship them separately.
