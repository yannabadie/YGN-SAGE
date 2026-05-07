---
name: codex IMPLEMENT delegation recipe — proven 2026-05-05 on cycle-12 Phase B
description: When doing repetitive mechanical refactor work (e.g., moving stage bodies, renaming APIs across many files), delegate to codex 0.128 xhigh fast via cgpro-derived prompts after Claude validates the recipe on the smallest cases first. Reserves Claude's context for orchestration + verification.
type: feedback
originSessionId: 3b36b883-89b8-4bbd-a2fe-cc9d305be971
---
**Rule**: For multi-step mechanical refactors with N similar instances (e.g., move 6 stage bodies, rename N call sites, add N type annotations), use this 3-actor split:

1. **cgpro DESIGN** — locks the spec + traps in advance. Q1-Q6 + 8-trap pattern.
2. **Claude IMPLEMENTS the smallest 2-3 instances** to validate the recipe + establish the commit message pattern.
3. **codex 0.128 xhigh fast IMPLEMENTS instances 4-N** via prompts derived from cgpro template.
4. **Claude VERIFIES + COMMITS each codex output** from the parent session.
5. **cgpro VERIFY** before final push.

**Why this works** (validated 2026-05-05 cycle-12 Phase B, 6 stage moves, ~2050 lines, 0 regressions):

- cgpro is good at strategic decisions + trap identification. It's NOT good at long-context mechanical work.
- codex 0.128 xhigh fast is excellent at mechanical body moves with crystal-clear constraints. It's not good at ambiguous strategic choices.
- Claude is good at orchestration, verification, commit message authoring, and recovery when codex output is slightly off. Burns Claude's context on the heavy mechanical work, leaving no room for the strategic course-corrections the session may need.

**Token economy** (rough):
- cgpro DESIGN: 1 round, ~5KB prompt + ~10KB response.
- Claude on smallest instances: ~3 commits × ~10K context each.
- codex on larger instances: 6 commits × ~40K context each (parsed by Claude on return, ~3K of the response is the diff Claude verifies).
- cgpro VERIFY: 1 round, ~5KB prompt + ~1KB GO_COMMIT_PUSH.

vs. Claude doing ALL the moves himself: ~6 × ~80K context = saturates the working context fast.

**When to apply this recipe**:
- N ≥ 4 similar instances (the codex setup overhead is amortized).
- Pure mechanical work (codex sandbox can't make architectural decisions).
- Each instance is independent (codex can't see what the previous instance did unless explicitly told).
- Tests are the verification gate (codex output must be runnable + passing).

**When NOT to apply this recipe**:
- Behavior-changing work (codex doesn't reliably preserve invariants across complex semantic changes).
- Cross-instance coordination (codex sees one instance at a time).
- Decision points requiring judgment (cgpro is the right delegate).
- Small N (≤3 instances — Claude is faster than the setup overhead).

**Setup specifics 2026-05-05**:
- codex CLI 0.128 on `C:\Program Files\nodejs\codex.cmd`. Working without `codex:rescue` plugin.
- codex sandbox runs as `mas_d0z9tb4\codexsandboxoffline` user — **CAN'T write to `.git/`** on this Windows setup (ACL-blocked). Workflow: codex modifies files + verifies tests, Claude commits from parent session.
- Codex prompt template stored at `.tmp/codex_<task>_move.md` per move — preserved for cycle retrospective.
- Each codex prompt MUST include: hard constraints, async-vs-sync explicit, module globals to import explicitly (cgpro DESIGN trap #2), local imports to keep verbatim, helpers that stay on the class, verification commands, commit-message pattern reference. Without these, codex hallucinates dead-code noqa comments or breaks invariants.
- Common codex foot-fault: adds `# noqa: F401` for imports it thinks are needed but actually aren't (e.g., `import asyncio` in a body that doesn't use `asyncio` directly because the lock is created via a class method). Claude catches this in the verification pass and removes them.

**Prompt template** (proven on 6 stage moves):
```
You are refactoring YGN-SAGE. This is pure code motion. ABSOLUTELY no logic
changes.

Repo: C:/Code/YGN-SAGE.
Baseline commit: <SHA>.

## Task
<one paragraph describing the move + line range>

## Pattern (already validated N times)
<list of recent commits showing the exact pattern>

## Hard constraints (cgpro DESIGN lock)
<the 10-12 specific traps from cgpro>

## Async signature [if applicable]
<exact code template the delegator must follow>

## Module globals to import explicitly
<what's used unqualified in the body>

## Helpers that stay on the class
<list with `self.<helper>(...)` patterns>

## Imports inside the body STAY where they are
<list of local imports — DO NOT lift them>

## Phase A wrapper test update
<old test name → new test name + the simplest deterministic path>

## Verification
<exact pytest + ruff commands>

## After verification
<commit message reference + "do not push, parent session will commit">
```

**Reference incident**: 2026-05-05 cycle-12 Phase B. 6 stage moves: decompose (Claude, ~24 lines, recipe validation), classify (Claude, ~114), assign_models (Claude, ~84), select_topology (codex, ~539), learn (codex, ~319 async), execute (codex, ~603, with bypass mutation intact). Total ~2050 lines moved across 7 commits in ~3 hours of wall-clock. Each codex output verified locally + had 1 minor cleanup (unused noqa imports). Recipe ZERO logic regressions across 25 P9 phase 1 byte-identical tests + 110-126 broader regression sweeps at every commit.
