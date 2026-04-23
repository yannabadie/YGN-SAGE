"""Pre-emission diff-context verifier for SWE-bench.

Spec:
    docs/superpowers/specs/2026-04-23-diff-context-verifier-design.md

Purpose
-------
Check that each hunk in a unified diff accurately describes the bytes
it claims to modify: the context lines (space-prefixed) and removed
lines (``-``-prefixed) must match the repo file at the hunk's old-side
start position. This catches the "context hallucination" class of
emission error (astropy-14182 Arm B, 2026-04-22 parity smoke) where
the agent emits a diff whose context lines disagree with file bytes it
has just read — the patch applies only because single-line additions
don't collide with the prefix-matching behaviour of ``git apply``.

Scope
-----
* UNIFIED DIFFS ONLY. SR blocks are validated separately by
  ``_scan_repo_for_unique_match``. Synthesised unified diffs from
  ``_blocks_to_unified_diff`` DO flow through here (per spec open
  question #3, confirmed via tests).
* Content verification only. Structural issues (header vs body line
  counts, malformed headers) are the scope of
  ``swebench_patch_repair``; this module defers to it by returning an
  empty list on malformed input.
* Pure function — no logging, no global state, no I/O beyond reading
  files whose paths come from the diff's own ``+++``/``---`` headers.

Match policy
------------
* Exact string equality first. If every context and removed line
  matches the file byte-for-byte: no mismatch.
* Otherwise, compare whitespace-stripped forms:
  * Stripped lines equal + ``difflib.SequenceMatcher.ratio()`` >= 0.95:
    accepted as clean. Whitespace drift within tolerance.
  * Stripped lines equal + ratio < 0.95:
    ``kind="fuzzy_below_threshold"``. Purely whitespace divergence
    heavy enough (e.g. tabs vs 8-space indent across many lines) to
    drop the raw ratio below the SR-parity threshold.
  * Stripped lines differ: ``kind="content_mismatch"``, regardless of
    ratio. Semantic divergence is never accepted — the 0.95 threshold
    gates whitespace tolerance only, NOT a generic "close enough"
    escape hatch.

The spec (2026-04-23-diff-context-verifier-design.md) literal policy
was "exact or ratio >= 0.95", but the verbatim astropy-14182 Arm B
emitted patch has ONE hallucinated line (``Table.__init__`` vs real
``FixedWidth.__init__``) in a 6-line body: SequenceMatcher ratio
0.956 > 0.95. That policy would have let the very hallucination the
feature exists to catch slip through. Spec open question #1 left room
for exactly this data-driven clarification — value 0.95 is unchanged,
its domain of applicability narrowed.

The module intentionally ships no external deps. ``difflib`` is stdlib;
the unified-diff parser here is ~40 LOC of regex + line accumulator,
matching the shape emitted by both ``git diff`` and our own
``_blocks_to_unified_diff``.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


_FUZZY_THRESHOLD = 0.95
"""Mirror of the SR extractor's threshold. Keep aligned — if this ever
forks, move both into ``swebench_emission_constants`` per spec guidance."""


_HUNK_HEADER_RE = re.compile(
    r"^@@\s+-(?P<old_start>\d+)(?:,(?P<old_count>\d+))?\s+"
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))?\s+@@"
)
"""Matches a unified-diff hunk header. Trailing function-context hint
(after the second ``@@``) is intentionally unconsumed — we don't
verify it (it's decorative; ``git apply`` ignores it for matching)."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HunkMismatch:
    """Structured mismatch record. Field semantics are NORMATIVE — the
    wiring in ``swebench_bench.py`` serialises a subset into
    ``_diff_verifier_mismatches`` for per-bucket analysis.

    * ``file`` — target file path as stated by the diff (``b/`` side,
      a/ prefix stripped). Relative to ``repo_dir``.
    * ``hunk_index`` — zero-indexed position across ALL hunks in the
      whole diff (not reset per-file).
    * ``old_start`` / ``old_count`` — old-side range from the hunk
      header (``@@ -old_start,old_count +...``). ``old_count`` defaults
      to 1 when the header omits it.
    * ``expected`` — the context + removed body lines the diff claims
      are at this position (one entry per line, trailing newline
      stripped).
    * ``actual`` — the repo file's bytes at ``[old_start,
      old_start+old_count)``, same shape.
    * ``kind`` — see ``Match policy`` in the module docstring.
    * ``match_ratio`` — SequenceMatcher ratio of joined bodies. 1.0 for
      an exact textual match (which doesn't land here — we only emit
      a mismatch when the bodies diverge). ``0.0`` for ``file_missing``
      (no actual content to compare against).
    """

    file: str
    hunk_index: int
    old_start: int
    old_count: int
    expected: list[str]
    actual: list[str]
    kind: Literal["file_missing", "content_mismatch", "fuzzy_below_threshold"]
    match_ratio: float


def verify_diff_context(
    diff: str,
    repo_dir: Path,
    fuzzy_threshold: float = _FUZZY_THRESHOLD,
) -> list[HunkMismatch]:
    """Return one ``HunkMismatch`` per problematic hunk; empty list = all ok.

    The function is total — it does not raise on malformed input. Any
    condition the verifier's scope does not cover (missing file
    headers, body-vs-header count drift, ``/dev/null`` sides for
    creation/deletion) collapses to "no opinion, return []". The caller
    (``generate_patches``) wraps in a defensive ``try/except`` as well,
    so even an unexpected crash here won't break the bench.
    """
    hunks = _parse_hunks(diff)
    if not hunks:
        return []

    mismatches: list[HunkMismatch] = []
    for idx, (file_path, old_start, old_count, expected_body) in enumerate(hunks):
        # File creations (--- /dev/null) and deletions (+++ /dev/null)
        # have nothing to verify on the old side. Skip cleanly.
        if file_path is None:
            continue

        abs_path = (repo_dir / file_path).resolve()
        if not abs_path.is_file():
            mismatches.append(
                HunkMismatch(
                    file=file_path,
                    hunk_index=idx,
                    old_start=old_start,
                    old_count=old_count,
                    expected=expected_body,
                    actual=[],
                    kind="file_missing",
                    match_ratio=0.0,
                )
            )
            continue

        # Body lines in the hunk must equal old_count (that's what the
        # header claims). When they don't, this is the structural-repair
        # case, not ours — defer.
        if len(expected_body) != old_count:
            continue

        try:
            file_text = abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            # Defensive — symlink loops, permissions, etc.
            continue
        file_lines = file_text.splitlines()

        # Old-side is 1-indexed. slice is [start-1 : start-1+count].
        end_idx = old_start - 1 + old_count
        actual_body = file_lines[old_start - 1 : end_idx]

        if actual_body == expected_body:
            continue

        # Exact diverged. Compute ratio (reported on every mismatch) and
        # classify: whitespace-only drift vs semantic divergence.
        expected_joined = "\n".join(expected_body)
        actual_joined = "\n".join(actual_body)
        ratio = difflib.SequenceMatcher(
            None, expected_joined, actual_joined
        ).ratio()

        stripped_expected = [ln.strip() for ln in expected_body]
        stripped_actual = [ln.strip() for ln in actual_body]
        stripped_equal = (
            len(stripped_expected) == len(stripped_actual)
            and stripped_expected == stripped_actual
        )

        kind: Literal[
            "file_missing", "content_mismatch", "fuzzy_below_threshold"
        ]
        if stripped_equal:
            if ratio >= fuzzy_threshold:
                # Whitespace drift within tolerance — accept as clean.
                continue
            kind = "fuzzy_below_threshold"
        else:
            # Semantic divergence — never accepted regardless of ratio.
            kind = "content_mismatch"

        mismatches.append(
            HunkMismatch(
                file=file_path,
                hunk_index=idx,
                old_start=old_start,
                old_count=old_count,
                expected=expected_body,
                actual=actual_body,
                kind=kind,
                match_ratio=ratio,
            )
        )

    return mismatches


# ---------------------------------------------------------------------------
# Parser — minimal stdlib, ~40 LOC.
# ---------------------------------------------------------------------------


def _parse_hunks(
    diff: str,
) -> list[tuple[str | None, int, int, list[str]]]:
    """Parse a unified diff into ``(file, old_start, old_count, body)``.

    Returns one tuple per hunk across all file sections. A hunk with no
    preceding ``--- ``/``+++ `` pair gets ``file=None`` and is skipped
    by the caller (nothing to verify).

    Note (2026-04-23 b9b25c0 bug fix): the earlier implementation returned
    ``[]`` when the diff had no ``diff --git`` header. That turned out
    to be a false-negative gate — real-world model emissions often use
    the shorter ``--- a/path`` / ``+++ b/path`` / ``@@ ... @@`` form
    that ``git apply`` accepts. Observability smoke on django-10914
    surfaced the bug (verifier passed a patch with clearly-mismatched
    context because the patch lacked ``diff --git``). Parsing now keys
    off ``---``/``+++``/``@@`` triples directly.

    ``file`` is ``None`` for old-side ``/dev/null`` sections (file
    creation) — those are skipped by the caller. Otherwise, ``file`` is
    the ``a/`` path with the prefix stripped.

    ``body`` contains one entry per context / removed line (leading
    ``' '`` or ``'-'`` character stripped). ``+``-prefixed lines are
    excluded — they are the new-side content, not the old-side the
    verifier checks against. ``\\ No newline at end of file`` markers
    are skipped (they don't count toward ``old_count``).
    """
    hunks: list[tuple[str | None, int, int, list[str]]] = []
    current_old_file: str | None = None
    have_file_header = False

    in_hunk = False
    hunk_old_start = 0
    hunk_old_count = 0
    hunk_body: list[str] = []

    def _flush() -> None:
        nonlocal in_hunk, hunk_body
        if in_hunk:
            hunks.append(
                (
                    current_old_file if have_file_header else None,
                    hunk_old_start,
                    hunk_old_count,
                    hunk_body,
                )
            )
            in_hunk = False
            hunk_body = []

    for raw_line in diff.splitlines():
        if raw_line.startswith("diff --git "):
            _flush()
            current_old_file = None
            have_file_header = False
            continue

        if raw_line.startswith("--- "):
            _flush()
            # Strip ``--- a/`` or ``--- `` prefix. ``/dev/null`` marks
            # a file creation — old-side nothing to verify.
            tail = raw_line[4:].strip()
            if tail == "/dev/null":
                current_old_file = None
            elif tail.startswith("a/"):
                current_old_file = tail[2:]
            else:
                current_old_file = tail
            continue

        if raw_line.startswith("+++ "):
            # The +++ line signals the file header is complete. We key
            # off the ``---`` side, but only treat the pair as a valid
            # header once both halves have been seen.
            have_file_header = current_old_file is not None
            continue

        if raw_line.startswith("@@"):
            _flush()
            m = _HUNK_HEADER_RE.match(raw_line)
            if not m:
                continue
            hunk_old_start = int(m.group("old_start"))
            # Unified diff: omitted count means 1.
            old_count_s = m.group("old_count")
            hunk_old_count = int(old_count_s) if old_count_s is not None else 1
            in_hunk = True
            hunk_body = []
            continue

        if not in_hunk:
            continue

        # Body line handling.
        if raw_line.startswith("\\"):
            # ``\ No newline at end of file`` — not a content line,
            # doesn't count toward old_count.
            continue
        if raw_line.startswith(" "):
            hunk_body.append(raw_line[1:])
        elif raw_line.startswith("-"):
            hunk_body.append(raw_line[1:])
        elif raw_line.startswith("+"):
            # New-side only; not part of old-side verification.
            continue
        else:
            # Empty line (completely blank) — in a hunk body this
            # usually means a whitespace-stripped context line. Treat
            # as a context line with empty content to keep counts
            # aligned with what ``git diff`` emitted.
            hunk_body.append("")

    _flush()
    return hunks
