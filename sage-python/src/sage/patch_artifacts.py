"""Unified-diff artifact detection, scoring and selection.

Shared by the TopologyRunner (F2 artifact pass-through), the agent loop
(F3) and the bench/canary layer (F1 rescue) — cgpro DESIGN_LOCKED
2026-06-11 (conv ``cgpro_emission_fixes_design``): detection is
UNIVERSAL and pure; any side-effect (final-output override, bench
rescue) is gated elsewhere by a *verified* task profile, never by an
LLM-inferred label.

Anti-false-positive contract (DESIGN trap #3): markdown horizontal
rules (``---``) and prose snippets must not register; a candidate needs
real unified-diff structure — file headers AND at least one ``@@`` hunk
with body lines.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

_FENCE_RE = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.DOTALL)
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", re.MULTILINE)
_HEADER_RE = re.compile(
    r"^(?:diff --git a/|--- (?:a/|/dev/null))", re.MULTILINE
)


@dataclass
class DiffArtifact:
    """One detected unified-diff artifact with its deterministic score."""

    payload: str
    sha256: str
    score: float
    parse_status: str  # "complete" | "partial"
    hunk_count: int
    file_count: int
    source_output_len: int
    node_idx: int | None = None
    role: str | None = None
    apply_status: str | None = None
    extra: dict = field(default_factory=dict)


def _looks_like_unified_diff(text: str) -> bool:
    return bool(_HEADER_RE.search(text)) and bool(_HUNK_RE.search(text))


def _hunk_bodies_complete(text: str) -> bool:
    """A 'complete' artifact has every hunk followed by at least one
    body line starting with ' ', '+', '-' or '\\' — truncation mid-hunk
    (the 'corrupt patch at line N' class) shows up as a header with no
    body or a body cut before any +/- line."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if _HUNK_RE.match(line):
            body = lines[i + 1 : i + 2]
            if not body or body[0][:1] not in (" ", "+", "-", "\\"):
                return False
    # The final hunk must not end ON the @@ header.
    return not (lines and _HUNK_RE.match(lines[-1]))


def extract_unified_diff_artifacts(
    text: str, *, node_idx: int | None = None, role: str | None = None
) -> list[DiffArtifact]:
    """Pure universal detector. Returns ALL plausible unified-diff
    artifacts in ``text`` (fenced blocks first, then the raw tail scan),
    each scored deterministically. No side effects."""
    if not text:
        return []
    candidates: list[str] = []
    for match in _FENCE_RE.finditer(text):
        block = match.group(1)
        if _looks_like_unified_diff(block):
            candidates.append(block)
    if not candidates and _looks_like_unified_diff(text):
        # Raw scan: slice from the first header line to the end.
        start = min(
            (m.start() for m in _HEADER_RE.finditer(text)), default=0
        )
        candidates.append(text[start:])

    artifacts: list[DiffArtifact] = []
    for cand in candidates:
        cand = cand.strip("\n") + "\n"
        hunks = len(_HUNK_RE.findall(cand))
        if hunks == 0:
            continue
        files = len(
            re.findall(r"^\+\+\+ (?:b/|/dev/null)", cand, re.MULTILINE)
        )
        complete = _hunk_bodies_complete(cand)
        # Deterministic score: completeness dominates, then structural
        # richness; length is a weak tiebreak (longer == more context).
        score = (
            (1000.0 if complete else 0.0)
            + 10.0 * hunks
            + 5.0 * files
            + min(len(cand), 20000) / 20000.0
        )
        artifacts.append(
            DiffArtifact(
                payload=cand,
                sha256=hashlib.sha256(cand.encode("utf-8")).hexdigest(),
                score=score,
                parse_status="complete" if complete else "partial",
                hunk_count=hunks,
                file_count=files,
                source_output_len=len(text),
                node_idx=node_idx,
                role=role,
            )
        )
    return artifacts


def select_best_artifact(
    artifacts: list[DiffArtifact],
) -> DiffArtifact | None:
    """Deterministic selection: highest score; on ties the LAST one wins
    (latest node's valid artifact — DESIGN test
    ``test_tiebreak_last_valid_deterministic``)."""
    best: DiffArtifact | None = None
    for art in artifacts:
        if best is None or art.score >= best.score:
            best = art
    return best


def git_apply_check(patch: str, repo_dir: str) -> tuple[bool, str]:
    """``git apply --check`` against a worktree. Shared by the canary
    rescue (F1) and the mini bench. Empty patch short-circuits."""
    if not patch.strip():
        return False, "empty_patch"
    with tempfile.NamedTemporaryFile(
        "w", suffix=".patch", delete=False, encoding="utf-8", newline="\n"
    ) as fh:
        fh.write(patch if patch.endswith("\n") else patch + "\n")
        patch_path = fh.name
    try:
        proc = subprocess.run(  # noqa: S603 - git trusted, paths ours
            ["git", "-C", repo_dir, "apply", "--check", "--verbose",
             patch_path],
            capture_output=True,
            timeout=60,
            check=False,
        )
        if proc.returncode == 0:
            return True, "applies"
        tail = (proc.stderr or b"").decode("utf-8", errors="replace")[-400:]
        return False, f"git_apply_exit={proc.returncode} {tail.strip()}"
    except subprocess.TimeoutExpired:
        return False, "git_apply_timeout"
    finally:
        try:
            Path(patch_path).unlink(missing_ok=True)
        except OSError:
            pass
