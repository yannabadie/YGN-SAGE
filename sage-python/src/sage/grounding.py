"""Emitter grounding: file tree, localization, verbatim file blocks.

GROUNDING block (2026-06-11, cgpro DESIGN_LOCKED on conv
``cgpro_emission_fixes_design``): the post-fix paired re-test showed arm
D emitting 9/10 patches with 0/10 applying — context grounding measured
against the real files found 8/9 patches predominantly hallucinated
(0-23% real context lines, 10/21 INVENTED file paths), while arm A —
whose emitter receives ``### FILE:`` blocks with verbatim bytes —
applies 6/10 with the same models.

G1 primary: ONE light localizer call over a deterministically
prefiltered tree, then VERBATIM bytes from the on-disk checkout — never
from prior tool-call memory (prior reads are localizer HINTS only).
Activation is gated by the verified task profile
(``sage.patch_artifacts.artifact_profile_active``), never an
LLM-inferred label.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

LOCALIZE_PROMPT = """You are a senior engineer. Given a bug report and the \
repository file listing, name the files (max {max_files}) most likely \
needing changes to fix the bug.
Reply with ONE repo-relative path per line, nothing else.

## Bug report
{problem}

## Repository files
{tree}
"""

GROUNDING_INSTRUCTION = (
    "Patch only files shown below unless you first use repo tools to "
    "verify another existing path. Context lines in your diff MUST match "
    "these bytes exactly."
)

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def repo_file_tree(repo_dir: str, *, max_files: int = 400) -> str:
    """``git ls-files`` tree for the localization call, capped."""
    try:
        proc = subprocess.run(  # noqa: S603 - git trusted, paths ours
            ["git", "-C", repo_dir, "ls-files"],
            capture_output=True,
            timeout=60,
            check=False,
        )
        files = (proc.stdout or b"").decode(
            "utf-8", errors="replace"
        ).splitlines()
    except (subprocess.TimeoutExpired, OSError):
        return ""
    if len(files) > max_files:
        files = files[:max_files] + [
            f"... (+{len(files) - max_files} more files)"
        ]
    return "\n".join(files)


def parse_file_list(
    reply: str, repo_dir: str, *, max_files: int = 6
) -> list[str]:
    """Extract EXISTING repo-relative paths from a localization reply.

    Path-grounding is half the battle: 10/21 target paths in the
    ungrounded arm-D patches were invented — every candidate here is
    validated against the real worktree."""
    candidates: list[str] = []
    for raw in (reply or "").splitlines():
        line = raw.strip().strip("`*-• ").strip()
        if not line:
            continue
        if " " in line:
            found = re.findall(r"[\w./\\-]+\.[A-Za-z0-9_]+", line)
            line = found[0] if found else ""
        line = line.strip("`'\"")
        if not line:
            continue
        rel = line.replace("\\", "/").lstrip("./")
        if (Path(repo_dir) / rel).is_file() and rel not in candidates:
            candidates.append(rel)
        if len(candidates) >= max_files:
            break
    return candidates


def files_block(
    repo_dir: str, rel_paths: list[str], *, max_chars_total: int = 60000
) -> str:
    """Concatenate selected file contents verbatim with headers, capped.

    The emitting model MUST see the real bytes it will write context
    lines against — this block is the difference between arm A's 6/10
    apply rate and arm D's hallucinated 0/10."""
    blocks: list[str] = []
    budget = max_chars_total
    for rel in rel_paths:
        try:
            text = (Path(repo_dir) / rel).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            continue
        chunk = (
            f"### FILE: {rel}\n```\n{text[: max(0, budget - 40)]}\n```\n"
        )
        blocks.append(chunk)
        budget -= len(chunk)
        if budget <= 1000:
            break
    return "\n".join(blocks)


def prefilter_paths(
    all_paths: list[str], problem: str, *, cap: int = 400
) -> list[str]:
    """Deterministic lexical prefilter (cgpro GROUNDING amendment: NOT
    the first-N-alphabetical slice). Scores each path by problem-token
    hits in its segments; stable order = (score desc, path asc)."""
    tokens = {t.lower() for t in _WORD_RE.findall(problem or "")}
    scored: list[tuple[int, str]] = []
    for path in all_paths:
        norm = path.replace("\\", "/").lower()
        segments = set(re.split(r"[/._-]", norm))
        hits = len(tokens & segments)
        partial = sum(1 for t in tokens if len(t) >= 5 and t in norm)
        scored.append((-(hits * 10 + partial), path))
    scored.sort()
    return [p for _score, p in scored[:cap]]


async def build_grounding_block(
    repo_dir: str,
    problem: str,
    llm: object,
    *,
    prior_paths: list[str] | None = None,
    max_files: int = 6,
    max_chars_total: int = 60000,
    call_timeout_s: float = 120.0,
) -> tuple[str, dict]:
    """G1 primary (cgpro GROUNDING DESIGN_LOCKED): localizer call over
    the prefiltered tree, then verbatim disk bytes in a GroundingEnvelope
    with the VERIFIED REPOSITORY CONTEXT header.

    Returns ``(block, telemetry)``; block is "" (with
    ``skipped_reason``) outside the verified artifact profile or when no
    valid path is localized."""
    import asyncio as _asyncio

    from sage.patch_artifacts import artifact_profile_active

    telemetry: dict = {
        "localizer_valid_paths": [],
        "localizer_dropped_paths": [],
        "grounding_truncated_files": [],
        "file_count": 0,
        "total_bytes": 0,
        "skipped_reason": None,
    }
    if not artifact_profile_active():
        telemetry["skipped_reason"] = "artifact_profile_inactive"
        return "", telemetry

    try:
        proc = subprocess.run(  # noqa: S603 - git trusted
            ["git", "-C", repo_dir, "ls-files"],
            capture_output=True,
            timeout=60,
            check=False,
        )
        all_paths = (proc.stdout or b"").decode(
            "utf-8", errors="replace"
        ).splitlines()
    except (subprocess.TimeoutExpired, OSError):
        telemetry["skipped_reason"] = "ls_files_failed"
        return "", telemetry
    if not all_paths:
        telemetry["skipped_reason"] = "empty_tree"
        return "", telemetry

    tree = "\n".join(prefilter_paths(all_paths, problem, cap=400))
    hints = ""
    if prior_paths:
        hints = (
            "\n\n## Paths already touched this run (hints, unverified)\n"
            + "\n".join(str(p) for p in prior_paths[:10])
        )

    from sage.llm.base import Message, Role

    prompt = LOCALIZE_PROMPT.format(
        max_files=max_files, problem=(problem or "")[:6000], tree=tree
    ) + hints
    try:
        response = await _asyncio.wait_for(
            llm.generate(  # type: ignore[attr-defined]
                messages=[Message(role=Role.USER, content=prompt)]
            ),
            timeout=call_timeout_s,
        )
    except Exception:  # noqa: BLE001 - localizer failure => no grounding
        telemetry["skipped_reason"] = "localizer_call_failed"
        return "", telemetry

    reply = getattr(response, "content", None) or ""
    ls_set = set(all_paths)
    raw_candidates: list[str] = []
    for raw in reply.splitlines():
        line = raw.strip().strip("`*-• ").strip()
        if not line:
            continue
        if " " in line:
            found = re.findall(r"[\w./\\-]+\.[A-Za-z0-9_]+", line)
            line = found[0] if found else ""
        line = line.strip("`'\"")
        if line:
            raw_candidates.append(line.replace("\\", "/").lstrip("./"))

    valid: list[str] = []
    dropped: list[str] = telemetry["localizer_dropped_paths"]
    for rel in raw_candidates:
        if (
            rel in ls_set
            and (Path(repo_dir) / rel).is_file()
            and rel not in valid
        ):
            valid.append(rel)
        elif rel not in valid and rel not in dropped:
            dropped.append(rel)
        if len(valid) >= max_files:
            break
    telemetry["localizer_valid_paths"] = valid
    if not valid:
        telemetry["skipped_reason"] = "no_valid_localized_paths"
        return "", telemetry

    blocks: list[str] = []
    budget = max_chars_total
    total_bytes = 0
    for rel in valid:
        try:
            text = (Path(repo_dir) / rel).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            continue
        total_bytes += len(text)
        if len(text) > max(0, budget - 80):
            cut = text[: max(0, budget - 80)]
            blocks.append(
                f"### FILE: {rel}\n```\n{cut}\n```\n"
                "[TRUNCATED — file exceeds the grounding cap; use "
                "read_file to verify content beyond this point before "
                "patching it]\n"
            )
            telemetry["grounding_truncated_files"].append(rel)
            budget = 0
        else:
            chunk = f"### FILE: {rel}\n```\n{text}\n```\n"
            blocks.append(chunk)
            budget -= len(chunk)
        if budget <= 1000:
            break
    telemetry["file_count"] = len(blocks)
    telemetry["total_bytes"] = total_bytes

    base_commit = ""
    try:
        head = subprocess.run(  # noqa: S603
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            capture_output=True,
            timeout=30,
            check=False,
        )
        base_commit = (head.stdout or b"").decode().strip()
    except (subprocess.TimeoutExpired, OSError):
        pass

    header = (
        "### VERIFIED REPOSITORY CONTEXT\n"
        f"base_commit: {base_commit}\n"
        f"file_count: {len(blocks)}\n"
        f"total_bytes: {total_bytes}\n\n"
        + GROUNDING_INSTRUCTION
        + "\n\n"
    )
    return header + "\n".join(blocks), telemetry


GROUNDING_MARKER = "### VERIFIED REPOSITORY CONTEXT"

_EMITTER_ROLE_HINTS = ("cod", "implement", "emit", "patch", "fixer", "developer")
_NON_EMITTER_HINTS = (
    "plan", "verif", "synth", "format", "output", "aggregat",
    "research", "review", "judge",
)


def is_emitter_role(role: str) -> bool:
    """Emitter-class roles receive the GroundingEnvelope; planners,
    verifiers and synthesizers do NOT (cgpro GROUNDING trap: FILE blocks
    on non-emitters are forbidden)."""
    role_lower = (role or "").lower()
    if any(h in role_lower for h in _NON_EMITTER_HINTS):
        return False
    return any(h in role_lower for h in _EMITTER_ROLE_HINTS)


def compose_grounded_task(grounding_block: str, task: str) -> str:
    """Prepend the GroundingEnvelope to a task as its own section —
    idempotent (the marker guards against bypass/topology
    double-injection)."""
    if not grounding_block:
        return task
    if GROUNDING_MARKER in (task or ""):
        return task
    return grounding_block + "\n\n## Task:\n" + task
