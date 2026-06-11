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
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ARTIFACT_PROFILE_ENV = "SAGE_TASK_ARTIFACT_PROFILE"


def artifact_profile_active() -> bool:
    """Verified task-profile gate (cgpro DESIGN_LOCKED 2026-06-11
    amendment #2): artifact-aware side-effects (final-output override,
    emitter budget promotion) only activate from this EXPLICIT
    operator/bench-set env var — never an LLM-inferred label."""
    return os.environ.get(ARTIFACT_PROFILE_ENV, "") == "unified_diff"


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


_FILE_HEADER_RE = re.compile(r"^--- (?:a/(?P<old>\S+)|/dev/null)$")


def positional_reground_exact(
    patch: str, repo_dir: str
) -> tuple[str, str]:
    """G2 strict mechanical reground (cgpro GROUNDING DESIGN_LOCKED
    2026-06-11): REPOSITION hunks whose old-side sequence (context ' '
    + removed '-' lines) matches the real file EXACTLY and UNIQUELY;
    rewrite @@ positions and recount from the body. Never invents:
    0 matches, >1 matches, missing path, truncated hunk or incoherent
    hunk order → ``reground_rejected:<reason>`` with the patch
    untouched. Built for the openlibrary class (100% grounded context,
    wrong positions) — useless by design against hallucinated context.

    Returns ``(patch_or_fixed, status)`` where status is
    ``reground_applied`` | ``reground_not_needed`` |
    ``reground_rejected:<reason>``.
    """
    if not patch.strip():
        return patch, "reground_rejected:empty_patch"
    lines = patch.splitlines()
    out: list[str] = []
    i = 0
    current_file: str | None = None
    file_lines: list[str] | None = None
    cumulative_delta = 0
    last_old_start = 0
    changed = False
    while i < len(lines):
        line = lines[i]
        header = _FILE_HEADER_RE.match(line)
        if line.startswith("diff --git") or header or line.startswith("+++ "):
            if header is not None:
                rel = header.group("old")
                if rel is None:
                    # /dev/null = new file: positions are fixed (@@ -0,0)
                    current_file, file_lines = None, None
                else:
                    target = Path(repo_dir) / rel
                    if not target.is_file():
                        return patch, "reground_rejected:missing_path"
                    current_file = rel
                    file_lines = target.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                    cumulative_delta = 0
                    last_old_start = 0
            out.append(line)
            i += 1
            continue
        if _HUNK_RE.match(line):
            # Collect the hunk body.
            body: list[str] = []
            j = i + 1
            while j < len(lines) and lines[j][:1] in (" ", "+", "-", "\\"):
                if lines[j].startswith(("--- ", "+++ ")):
                    break
                body.append(lines[j])
                j += 1
            old_side = [b[1:] for b in body if b[:1] in (" ", "-")]
            new_count = sum(1 for b in body if b[:1] in (" ", "+"))
            old_count = len(old_side)
            if not body or not old_side:
                return patch, "reground_rejected:truncated_hunk"
            if current_file is None or file_lines is None:
                # new-file hunk: pass through untouched
                out.append(line)
                out.extend(body)
                i = j
                continue
            # Exact-match the old-side sequence in the real file.
            matches = [
                idx
                for idx in range(len(file_lines) - old_count + 1)
                if file_lines[idx : idx + old_count] == old_side
            ]
            if len(matches) == 0:
                return patch, "reground_rejected:no_exact_match"
            if len(matches) > 1:
                return patch, "reground_rejected:ambiguous_match"
            old_start = matches[0] + 1  # 1-based
            if old_start <= last_old_start:
                return patch, "reground_rejected:incoherent_hunk_order"
            last_old_start = old_start
            new_start = old_start + cumulative_delta
            new_header = (
                f"@@ -{old_start},{old_count} +{new_start},{new_count} @@"
            )
            if new_header != line:
                changed = True
            out.append(new_header)
            out.extend(body)
            cumulative_delta += new_count - old_count
            i = j
            continue
        out.append(line)
        i += 1
    if not changed:
        return patch, "reground_not_needed"
    fixed = "\n".join(out)
    if not fixed.endswith("\n"):
        fixed += "\n"
    return fixed, "reground_applied"


ALLOW_NEW_FILES_ENV = "SAGE_ARTIFACT_ALLOW_NEW_FILES"


def patch_path_coverage(patch: str, repo_dir: str) -> dict:
    """Post-emission path-coverage guard (cgpro GROUNDING amendment #5:
    with 10/21 paths invented, this guard matters as much as the
    injection). A path referenced by the patch but absent from the repo
    is MISSING — except a true file creation (``--- /dev/null``), and
    only when the profile explicitly allows new files
    (``SAGE_ARTIFACT_ALLOW_NEW_FILES=1``; default blocked for SWE-style
    bugfixes).

    Returns ``{referenced, missing, new_files, coverage}`` where
    coverage = grounded/(referenced or 1).
    """
    referenced: list[str] = []
    missing: list[str] = []
    new_files: list[str] = []
    allow_new = os.environ.get(ALLOW_NEW_FILES_ENV, "") == "1"
    lines = (patch or "").splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^\+\+\+ b/(\S+)", line)
        if not m:
            continue
        rel = m.group(1)
        if rel in referenced or rel in new_files:
            continue
        is_new = i > 0 and lines[i - 1].startswith("--- /dev/null")
        if is_new:
            new_files.append(rel)
            if not allow_new:
                missing.append(rel)
            continue
        referenced.append(rel)
        if not (Path(repo_dir) / rel).is_file():
            missing.append(rel)
    total = len(referenced) + len(new_files)
    grounded = total - len(missing)
    return {
        "referenced": referenced,
        "missing": missing,
        "new_files": new_files,
        "coverage": (grounded / total) if total else 0.0,
    }


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
