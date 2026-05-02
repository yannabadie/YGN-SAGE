# YGN-SAGE Codex Memory Import - 2026-05-01

This file is the Codex-facing entrypoint for the Claude Code project memory import requested on 2026-05-01.

## Why this exists

Codex can reliably load repo-local instruction context through `AGENTS.md`. Claude Code stores project instructions and auto memory through `CLAUDE.md`, `.claude/rules/`, and the per-project auto-memory directory under `~/.claude/projects/<project>/memory/`.

The internal managed Codex memory store is not a project file and is not directly writable from this repo workflow. The practical durable surface for this project is therefore:

1. keep the current Claude memory copied in the repo,
2. keep compact digests for high-value transcripts,
3. point `AGENTS.md` at those files so future Codex sessions read them before work.

## Official-doc basis checked

- Claude Code docs: `CLAUDE.md` files and auto memory are loaded at session start; auto memory is stored per project at `~/.claude/projects/<project>/memory/` with `MEMORY.md` as the entrypoint.
- Claude Code docs: `.claude/rules/` is a structured instruction surface; `CLAUDE.md` remains the main memory/instruction file, not `AGENTS.md`.
- Claude Code settings docs: local transcripts are retained according to `cleanupPeriodDays`; this is why the JSONL conversation files under `~/.claude/projects/C--Code-YGN-SAGE/` were inspected.
- OpenAI Codex docs: Codex reads `AGENTS.md` files before work, layering global and project files; repo-local pointers are the right place for project memory without polluting global Codex state.

## Imported sources

### Claude Code auto memory

Source:
`C:\Users\yann.abadie\.claude\projects\C--Code-YGN-SAGE\memory`

Repo copy:
`docs/claude-memory`

Import result on 2026-05-01 10:01 Europe/Paris:
- 53 files seen in source.
- 11 new or newer files copied.
- `docs/claude-memory/MEMORY.md` is now the current Claude auto-memory entrypoint, not the older 2026-04-24 snapshot.

High-signal newest files copied:
- `project_april30_cycle8_closeout_architect_review.md`
- `project_april30_cycle8_r6_1c.md`
- `project_april29_cycle7_flip.md`
- `project_april29_r6_1a_cycle6.md`
- `project_april27_boot_loop_fix.md`
- `project_april26_cgpro_review_findings.md`
- `feedback_cgpro_mastery.md`
- `feedback_cgpro_source_of_truth.md`

### Last two Claude Code conversations

Raw source transcripts remain in Claude's local project directory:
- `C:\Users\yann.abadie\.claude\projects\C--Code-YGN-SAGE\dc83c9bb-b729-40fa-aa8c-ca8f426eebc5.jsonl`
- `C:\Users\yann.abadie\.claude\projects\C--Code-YGN-SAGE\b7b56b62-e6ea-4a71-965c-def15a6da3a2.jsonl`

Codex digest:
`docs/codex-memory/claude-last-two-conversations.md`

The raw JSONL files are large and include tool output, hook payloads, and transient details. Future agents should read the digest first, then query the raw JSONL with `rg` for exact evidence when needed.

### Memory bank

Source:
`memory-bank`

Repo copy:
`docs/codex-memory/memory-bank`

Assessment: the current `memory-bank` content is mostly generic scaffold from 2026-04-03 (`Goal 1`, placeholder product context, placeholder patterns). Treat it as low-trust background. The fresher Claude memory and current repo files supersede it.

## Read order for future Codex sessions

1. `CLAUDE.md`
2. `docs/claude-memory/MEMORY.md`
3. `docs/codex-memory/CLAUDE_PROJECT_IMPORT_2026-05-01.md`
4. `docs/codex-memory/claude-last-two-conversations.md`
5. Task-specific files under `docs/claude-memory/`, `.claude/rules/`, and current source/tests/docs.
6. `docs/codex-memory/memory-bank/` only when historical scaffold context is specifically useful.

## Trust rules

- Current repo behavior, tests, branch, and Git status beat any imported memory.
- Claude memory is directional context, not proof.
- Conversation digests are summaries; use raw JSONL paths for exact prompts, commands, or timestamps.
- Do not copy project-specific YGN-SAGE content into global Codex memory surfaces.
