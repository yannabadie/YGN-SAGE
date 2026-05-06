"""Narrative-coherence guard for capability-state docs.

Cycle-13 K Phase 0.6b (cgpro post-push 2026-05-06 EDIT_REQUIRED): the
README, AI-ARCHITECTURE.md, CLAUDE.md, and `.claude/rules/architecture.md`
are the four marketing-grade / agent-facing surfaces that external
readers (and the agent itself) consume to learn YGN-SAGE's capabilities.
ALIRE.md flagged narrative drift between these surfaces and the
machine-readable CLAIMS registry as the #1 truthfulness risk.

This test bans residual marketing-grade phrasings that would silently
re-introduce that drift:

  - Bare "92% GT" / "88% GT" / "34% GT" claims (the figures are
    `evidence_pending` in `docs/CLAIMS.yaml` until a CI-runnable test
    pins them — citing the bare number elsewhere undermines the whole
    Phase 0.4 contract).
  - "Path 6: Learned policy" / "Path 6 (learned" / "OR Path 6" — these
    forms perpetuate the historical naming collision where "Path 6" was
    used for two different things (engine-path-6 = template fallback
    per Rust `TopologySource`, AND the learned-policy sibling-of-6).
    Phase 0.6 chose: engine path 6 = template fallback. The learned
    policy is "optional learned-policy path" / sibling-of-6.

Self-caveatted forms (a line that ALSO mentions `evidence_pending` or
`CLAIMS.yaml` or a `routing.*` registry id near the figure) are
allowed — that's the explicit anchor pointing at the registry, which
is exactly what we want.

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
    # de repo' (id 69fb0d11). The guard now spans 10 narrative-grade docs.
    ".claude/rules/critical-directives.md",
    ".claude/rules/research-decisions.md",
    "ui/README.md",
    "ui/app.py",
    "sage-python/src/sage/routing/README.md",
    "sage-core/src/README.md",
)

# Each pattern below is a (label, regex, allow_substrings) tuple.
# `allow_substrings` is a list of strings; if any of them appears on the
# SAME line as a regex match, that line is treated as self-caveatted
# (acceptable). The caveat must be in the same line so a reader scanning
# the doc cannot miss it. A future stronger version could check by
# paragraph rather than line, but line-level is unambiguous and easy to
# reason about.
_FORBIDDEN_PATTERNS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "bare-92-GT",
        r"\b92%\s*GT\b",
        ("evidence_pending", "CLAIMS.yaml", "routing.knn_92pct", "historically"),
    ),
    (
        "bare-88-GT",
        r"\b88%\s*GT\b",
        ("evidence_pending", "CLAIMS.yaml", "routing.system_router_88pct", "historically"),
    ),
    (
        "bare-34-GT",
        r"\b34%\s*GT\b",
        ("evidence_pending", "CLAIMS.yaml", "historically"),
    ),
    (
        "bare-kNN-92",
        r"\bkNN\s+92%",
        ("evidence_pending", "CLAIMS.yaml", "routing.knn_92pct", "historically"),
    ),
    (
        "bare-SystemRouter-88",
        r"\bSystemRouter\s+88%",
        ("evidence_pending", "CLAIMS.yaml", "routing.system_router_88pct", "historically"),
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
        ("evidence_pending", "CLAIMS.yaml", "routing.knn_92pct", "historic", "non-autoritative"),
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
)


def _scan_file(path: Path) -> list[tuple[int, str, str]]:
    """Return [(line_no, label, line_text), ...] of forbidden hits in `path`.

    Empty list when clean. Self-caveatted lines (allow_substrings on same
    line) are excluded.
    """
    text = path.read_text(encoding="utf-8")
    hits: list[tuple[int, str, str]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for label, regex, allow_substrings in _FORBIDDEN_PATTERNS:
            if re.search(regex, line):
                if any(allow in line for allow in allow_substrings):
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
    """Cheap sanity check: the four guarded docs MUST exist; if one is
    moved/renamed, this test fails so the rename is caught at PR time."""
    missing = [d for d in _GUARDED_DOCS if not (_REPO_ROOT / d).is_file()]
    assert not missing, f"Guarded narrative docs missing: {missing}"
