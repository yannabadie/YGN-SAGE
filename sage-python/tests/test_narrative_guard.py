"""Narrative-coherence guard for capability-state docs.

Cycle-13 K Phase 0.6b (cgpro post-push 2026-05-06 EDIT_REQUIRED): the
README, AI-ARCHITECTURE.md, CLAUDE.md, and `.claude/rules/architecture.md`
are the four marketing-grade / agent-facing surfaces that external
readers (and the agent itself) consume to learn YGN-SAGE's capabilities.
ALIRE.md flagged narrative drift between these surfaces and the
machine-readable CLAIMS registry as the #1 truthfulness risk.

This test bans residual marketing-grade phrasings that would silently
re-introduce that drift:

  - Bare "92% GT" / "88% GT" / "34% GT" claims. Since 2026-05-07
    (commits da582a77 + e785753a) the `routing.knn_92pct` and
    `routing.system_router_88pct` claims are `delivered` against a
    strict-equal floor on the 60-task GT (50/60 and 52/60 respectively).
    The historical 92% / 88% figures were measured on an earlier
    50-task GT subset and are provenance only — not recertified by the
    floor. The same-line caveat tokens require any narrative mention
    of the bare figure to make this provenance explicit.
  - "Path 6: Learned policy" / "Path 6 (learned" / "OR Path 6" — these
    forms perpetuate the historical naming collision where "Path 6" was
    used for two different things (engine-path-6 = template fallback
    per Rust `TopologySource`, AND the learned-policy sibling-of-6).
    Phase 0.6 chose: engine path 6 = template fallback. The learned
    policy is "optional learned-policy path" / sibling-of-6.

Self-caveatted forms (a line that ALSO mentions `CLAIMS.yaml` or a
`routing.*` registry id, OR uses `historical` / `historically` /
`provenance only` / `not recertified` / `50-task` / `delivered` /
`floor` near the figure) are allowed.

If a future PR reintroduces a bare phrasing, this test fails and the
ALIRE-driven truthfulness contract is preserved as a hard gate.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]

_GUARDED_DOCS: tuple[str, ...] = (
    "README.md",
    "AI-ARCHITECTURE.md",
    "CLAUDE.md",
    ".claude/rules/architecture.md",
    # Phase 0.6c (cgpro post-Phase-0.6b EDIT_REQUIRED): extend to additional
    # agent-facing / user-facing surfaces flagged on conv 'Analyse approfondie
    # de repo' (id 69fb0d11).
    ".claude/rules/critical-directives.md",
    ".claude/rules/research-decisions.md",
    "ui/README.md",
    "ui/app.py",
    "sage-python/src/sage/routing/README.md",
    "sage-core/src/README.md",
    # Phase 0.6d (cgpro post-Phase-0.6c EDIT_REQUIRED): the Obsidian-vault
    # archive docs were left out of the guard in Phase 0.6c — banner-only
    # protection is decorative when guard patterns target this exact class
    # of phrasings. Add them so any regression is caught at PR-time.
    "YGN-SAGE/Architecture/Pillar-5-Strategy.md",
    "YGN-SAGE/Architecture/Pipeline.md",
    "YGN-SAGE/Architecture/00-Architecture-MOC.md",
    # Phase 0.6e (cgpro post-Phase-0.6d EDIT_REQUIRED): long-tail Papers /
    # Benchmarks / Decisions docs and one code-side docstring still cited
    # routing percentages as authoritative without registry anchors.
    "YGN-SAGE/Papers/kNN-Routing.md",
    "YGN-SAGE/Benchmarks/Routing-GT.md",
    "YGN-SAGE/Decisions/ADR-001-kNN-over-Heuristic.md",
    "YGN-SAGE/Decisions/00-Decisions-MOC.md",
    "YGN-SAGE/Papers/00-Papers-MOC.md",
    "YGN-SAGE/Benchmarks/00-Benchmarks-MOC.md",
    "docs/papers/paper1_knn_routing.md",
    "docs/papers/paper2_sage_system.md",
    "docs/benchmarks/results.md",
    "sage-python/src/sage/pipeline_v2/classify.py",
    # Phase 0.6e closeout: 2 additional READMEs caught by repo-wide grep.
    "sage-python/README.md",
    "sage-python/src/sage/bench/README.md",
    # Phase 0.6f (post-Phase-0.6e repo-wide grep): final 2 surfaces caught.
    "sage-python/src/sage/strategy/adaptive_router.py",
    "docs/heuristics-needing-ablation.md",
)

# Each pattern below is a (label, regex, allow_substrings) tuple.
# `allow_substrings` is a list of strings; if any of them appears on the
# SAME line as a regex match, that line is treated as self-caveatted
# (acceptable). The caveat must be in the same line so a reader scanning
# the doc cannot miss it. A future stronger version could check by
# paragraph rather than line, but line-level is unambiguous and easy to
# reason about.
#
# Phase 4 (cycle-13 K post-routing-evidence-pin, 2026-05-07): the
# routing.knn_92pct + routing.system_router_88pct claims flipped from
# `evidence_pending` to `delivered` (commits da582a77 + e785753a). The
# allow-token set tightens accordingly — `evidence_pending` is no longer a
# valid same-line caveat for headline routing figures since the registry
# entries have flipped to `delivered`. New caveat tokens are added so the
# rewritten narrative ("historical 92% on the earlier 50-task GT, provenance
# only, not recertified by the 60-task floor") passes. Registry anchors
# (`CLAIMS.yaml`, `routing.knn_92pct`, `routing.system_router_88pct`) and
# the existing `historic`/`historically`/`non-autoritative` tokens stay.
_ROUTING_KNN_ALLOW: tuple[str, ...] = (
    "CLAIMS.yaml",
    "routing.knn_92pct",
    "historical",
    "historically",
    "historic",
    "provenance only",
    "not recertified",
    "50-task",
    "delivered",
    "floor",
    "non-autoritative",
)
_ROUTING_SR_ALLOW: tuple[str, ...] = (
    "CLAIMS.yaml",
    "routing.system_router_88pct",
    "historical",
    "historically",
    "historic",
    "provenance only",
    "not recertified",
    "50-task",
    "delivered",
    "floor",
    "non-autoritative",
)
# Heuristic ComplexityRouter is `retired` per registry — historical-only is
# now the only acceptable framing.
_ROUTING_HEURISTIC_ALLOW: tuple[str, ...] = (
    "CLAIMS.yaml",
    "historical",
    "historically",
    "historic",
    "Priority-3",
    "emergency fallback",
    "retired",
    "non-autoritative",
)

_FORBIDDEN_PATTERNS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "bare-92-GT",
        r"\b92%\s*GT\b",
        _ROUTING_KNN_ALLOW,
    ),
    (
        "bare-88-GT",
        r"\b88%\s*GT\b",
        _ROUTING_SR_ALLOW,
    ),
    (
        "bare-34-GT",
        r"\b34%\s*GT\b",
        _ROUTING_HEURISTIC_ALLOW,
    ),
    (
        "bare-kNN-92",
        r"\bkNN\s+92%",
        _ROUTING_KNN_ALLOW,
    ),
    (
        "bare-SystemRouter-88",
        r"\bSystemRouter\s+88%",
        _ROUTING_SR_ALLOW,
    ),
    (
        "path6-learned-collision",
        r"Path\s*6\s*:\s*Learned",
        (),  # No exception — flip to "Optional learned-policy path".
    ),
    (
        "path6-learned-paren",
        r"Path\s*6\s*\(learned",
        (),
    ),
    (
        "or-path6",
        r"\bOR\s+Path\s*6\b",
        (),
    ),
    # Phase 0.6c additions
    (
        "bare-93p3-GT",
        r"\b93\.3%\s*GT\b",
        _ROUTING_KNN_ALLOW,
    ),
    (
        "dead-code-routing-framing",
        r"\bDEAD CODE\b",
        # AUDIT2 2026-04-24 corrected this framing to "Priority-3 emergency
        # fallback only". The original "DEAD CODE" framing was technically
        # contradicted by the live fallback path. Allow the phrase only when
        # the same line acknowledges the correction.
        ("Priority-3", "emergency fallback", "AUDIT2", "corrected", "non-autoritative", "NOT dead code"),
    ),
    # Phase 0.6d additions (cgpro round-5): the Phase 0.6c regex set covered
    # `kNN 92%` / `SystemRouter 88%` / `34% GT` / etc., but missed the
    # parenthesised-attribution form `Rust kNN (92%)` / `SystemRouter (88%)`
    # / `ComplexityRouter (34%)`. Phase 0.6d closes that class.
    (
        "bare-knn-paren-92",
        r"\bkNN\b[^\n]*?\(\s*92%",
        _ROUTING_KNN_ALLOW,
    ),
    (
        "bare-systemrouter-paren-88",
        r"\bSystemRouter\b[^\n]*?\(\s*88%",
        _ROUTING_SR_ALLOW,
    ),
    (
        "bare-complexityrouter-paren-34",
        r"\bComplexityRouter\b[^\n]*?\(\s*34%",
        _ROUTING_HEURISTIC_ALLOW,
    ),
    # Phase 0.6e additions (cgpro round-6): the paren-attribution patterns
    # of Phase 0.6d caught `kNN (92%)` but missed table-row / paper-prose
    # forms like `kNN (arctic-embed-m) | **92%** (46/50)` and `kNN routing
    # achieves 92% accuracy`. The broader "kNN within 160 chars of any
    # XX%" pattern closes the long-tail surface while keeping the same-
    # line caveat allowance.
    (
        "bare-knn-near-92",
        r"\bkNN\b[^\n]{0,160}\b92%",
        _ROUTING_KNN_ALLOW,
    ),
    (
        "bare-knn-near-93p3",
        r"\bkNN\b[^\n]{0,160}\b93\.3%",
        _ROUTING_KNN_ALLOW,
    ),
    (
        "bare-systemrouter-near-88",
        r"\bSystemRouter\b[^\n]{0,160}\b88%",
        _ROUTING_SR_ALLOW,
    ),
    (
        "bare-routing-sample-sizes",
        # Two routing-GT sample-size signatures: 56/60 (60-task GT) and
        # 46/50 (legacy 50-task subset). Banning the bare form forces
        # any future citation to come with a registry anchor.
        r"\b56/60\b|\b46/50\b",
        _ROUTING_KNN_ALLOW,
    ),
    (
        "bare-complexityrouter-near-45",
        r"\bComplexityRouter\b[^\n]{0,160}\b45%",
        _ROUTING_HEURISTIC_ALLOW,
    ),
    # Phase 4 (cgpro post-Phase-4 EDIT_REQUIRED 2026-05-07, Option C): the
    # bare-XX% patterns above only fire when the offending line carries a
    # numeric figure. They miss the inverse class — a line that says
    # `routing.knn_92pct evidence_pending` (claim ID + stale status, no
    # 92%) — which is exactly what survived in the README Strategy
    # bullet + AI-ARCHITECTURE Stage 0 bullets after the routing flip.
    #
    # cgpro Option C lock: archive/historic docs MAY say claims "were
    # evidence_pending at the March 2026 snapshot" (past tense + registry
    # pointer), but MUST NOT say "are evidence_pending" / "currently
    # evidence_pending" (present tense, since the registry now says
    # `delivered`). The `_is_archive_historic_status_line` predicate
    # below enforces the 3-token AND: archive context + past-tense
    # status + current-registry pointer. The any-token allowlist
    # alternative was rejected as too permissive.
    (
        "stale-routing-knn-status",
        r"(routing\.knn_92pct[^\n]{0,160}evidence_pending|evidence_pending[^\n]{0,160}routing\.knn_92pct)",
        (),
    ),
    (
        "stale-routing-systemrouter-status",
        r"(routing\.system_router_88pct[^\n]{0,160}evidence_pending|evidence_pending[^\n]{0,160}routing\.system_router_88pct)",
        (),
    ),
    # Generic catch for the README project-tree style "kNN ... accuracy
    # ... evidence_pending" wording (no claim ID, no figure, just the
    # status drift).
    (
        "stale-knn-accuracy-pending",
        r"\bkNN\b[^\n]{0,120}\baccuracy\b[^\n]{0,80}\bevidence_pending\b",
        (),
    ),
)


# Stale-status pattern labels — these use the archive-historic predicate
# instead of the any-token allowlist, per cgpro Option C.
_STALE_STATUS_PATTERN_LABELS = frozenset({
    "stale-routing-knn-status",
    "stale-routing-systemrouter-status",
    "stale-knn-accuracy-pending",
})

# Archive context tokens (line must contain one of these for the line
# to be considered archive/historic content).
_ARCHIVE_CONTEXT_TOKENS = (
    "archive snapshot",
    "historic snapshot",
    "historic figures",
    "historical snapshot",
    "non-autoritative",
    "historique",
)

# Past-tense status tokens (line must explicitly past-tense the
# evidence_pending status).
_PAST_STATUS_TOKENS = (
    "was `evidence_pending`",
    "were `evidence_pending`",
    "was evidence_pending",
    "were evidence_pending",
    "était `evidence_pending`",
    "étaient `evidence_pending`",
    "etait `evidence_pending`",
    "etaient `evidence_pending`",
    "was evidence_pending at",
    "were evidence_pending in",
    "was evidence_pending in",
)

# Current-registry pointer tokens (line must point readers at the
# authoritative registry).
_CURRENT_STATUS_POINTER_TOKENS = (
    "current status",
    "current authoritative status",
    "current authoritative status lives",
    "current status lives",
    "docs/CLAIMS.yaml",
)


def _is_archive_historic_status_line(line: str) -> bool:
    """cgpro Option C predicate for stale-status patterns.

    Archive/historic docs MAY say claims were evidence_pending in the
    past, but only when the line ALSO carries an archive-context token
    AND points readers at the current registry. This blocks the
    "non-autoritative + claims are evidence_pending" weasel pattern
    (which is the failure mode the unrestricted allowlist would
    permit).
    """
    lower = line.lower()
    has_archive_context = any(t in lower for t in _ARCHIVE_CONTEXT_TOKENS)
    has_past_status = any(t in lower for t in _PAST_STATUS_TOKENS)
    has_registry_pointer = any(
        t.lower() in lower for t in _CURRENT_STATUS_POINTER_TOKENS
    )
    return has_archive_context and has_past_status and has_registry_pointer


def _scan_file(path: Path) -> list[tuple[int, str, str]]:
    """Return [(line_no, label, line_text), ...] of forbidden hits in `path`.

    Empty list when clean. Self-caveatted lines (allow_substrings on same
    line) are excluded for normal patterns; stale-status patterns use
    the archive-historic predicate instead per cgpro Option C 2026-05-07.
    """
    text = path.read_text(encoding="utf-8")
    hits: list[tuple[int, str, str]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for label, regex, allow_substrings in _FORBIDDEN_PATTERNS:
            if re.search(regex, line):
                if label in _STALE_STATUS_PATTERN_LABELS:
                    if _is_archive_historic_status_line(line):
                        continue
                elif any(allow in line for allow in allow_substrings):
                    continue
                hits.append((line_no, label, line.strip()))
    return hits


@pytest.mark.parametrize("rel_path", _GUARDED_DOCS)
def test_doc_has_no_bare_marketing_phrasings(rel_path: str) -> None:
    target = _REPO_ROOT / rel_path
    if not target.is_file():
        pytest.skip(f"{target} not present in this checkout")
    hits = _scan_file(target)
    assert not hits, (
        f"Narrative drift in {rel_path} — Phase 0.6b banned phrases reappeared:\n"
        + "\n".join(f"  L{n} [{label}]: {snippet}" for n, label, snippet in hits)
        + "\nFix: cite the figure WITH `evidence_pending` / `CLAIMS.yaml` / "
        "registry id (`routing.knn_92pct` etc.) on the SAME line, OR "
        "rewrite to use 'optional learned-policy path' instead of 'Path 6'."
    )


def test_guarded_docs_all_exist() -> None:
    """Cheap sanity check: the guarded docs MUST exist; if one is
    moved/renamed, this test fails so the rename is caught at PR time."""
    missing = [d for d in _GUARDED_DOCS if not (_REPO_ROOT / d).is_file()]
    assert not missing, f"Guarded narrative docs missing: {missing}"
