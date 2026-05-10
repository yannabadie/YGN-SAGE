# YGN-SAGE Claude Code Project Memory Refresh - 2026-05-10

This file records the 2026-05-10 refresh of Claude Code project memory into
repo-local Codex-readable files.

## Source And Destination

Claude Code auto-memory source:
`C:\Users\yann.abadie\.claude\projects\C--Code-YGN-SAGE\memory`

Repo copy:
`docs/claude-memory`

The source tree was left intact except for the new 2026-05-10 project-memory
entry and the `MEMORY.md` index update. The repo copy was refreshed by copying
the source files into `docs/claude-memory`.

## Verification

After refresh:

- Source files: 75
- Destination files: 75
- Missing in destination: 0
- Extra destination files: 0
- Hash mismatches: 0
- Newest file in both trees: `MEMORY.md`, last write `2026-05-10 16:37:20`
  local filesystem time

## New Claude Memory Files Added To The Repo Copy Since 2026-05-07

- `project_may07_block_d_diff_verifier_budget.md`
- `project_may09_provider_status.md`
- `project_may10_provider_model_catalog_refresh.md`

## Current Claude Transcript Handles

Newest top-level Claude Code JSONL transcripts under
`C:\Users\yann.abadie\.claude\projects\C--Code-YGN-SAGE` at refresh time:

- `adec9fba-4188-4b8d-83fd-eec5e9006586.jsonl`, last write
  `2026-05-10 15:51:15`, length `15237`
- `98c5d292-7098-4166-9e24-6d083e057a81.jsonl`, last write
  `2026-05-10 15:50:52`, length `9499485`
- `ea52b6a4-977f-4cac-83f6-f491b3dad1c2.jsonl`, last write
  `2026-05-07 18:17:51`, length `3247906`

`docs/codex-memory/claude-last-two-conversations.md` was refreshed to summarize
the two newest transcript handles above.

## Current Read Order

1. `CLAUDE.md`
2. `docs/status/2026-05-10-current-state.md`
3. `docs/claude-memory/MEMORY.md`
4. `docs/codex-memory/CLAUDE_PROJECT_IMPORT_2026-05-10.md`
5. `docs/codex-memory/CODEX_PROJECT_REFRESH_2026-05-10.md`
6. Task-specific copied Claude memory files under `docs/claude-memory/`
7. Raw JSONL transcripts only when exact evidence is required

## Trust Rules

- Current repo state, tests, branch, and Git status beat copied memory.
- Claude memory is directional context, not proof of current behavior.
- The 2026-05-10 provider/model-catalog memory supersedes the 2026-05-09
  provider-status note where they conflict.
- `docs/codex-memory/memory-bank/` remains low-trust scaffold unless current
  files corroborate it.
