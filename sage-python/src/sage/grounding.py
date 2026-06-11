"""Emitter grounding: file tree, localization parsing, verbatim file blocks.

GROUNDING block (2026-06-11, follow-up to the cgpro DESIGN_LOCKED
EMISSION_FIXES): the post-fix paired re-test showed arm D emitting 9/10
patches with 0/10 applying — context grounding measured against the real
files found 8/9 patches predominantly hallucinated (0-23% real context
lines, 10/21 INVENTED file paths), while arm A — whose emitter receives
``### FILE:`` blocks with verbatim bytes — applies 6/10 with the same
models. These helpers are the arm-A provisioning anatomy, promoted from
``scripts/run_mini_ab.py`` into the SDK so the pipeline's emitting nodes
(bypass loop and topology emitter roles) can be grounded the same way.

Pure helpers — no side effects; activation is gated elsewhere by the
verified task profile (``sage.patch_artifacts.artifact_profile_active``),
never an LLM-inferred label.
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
