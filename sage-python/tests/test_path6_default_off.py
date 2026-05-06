"""Optional-learned-policy default-off non-regression test (sometimes called "Path 6").

Cycle-13 K Phase 0.5 + 0.6 (cgpro `Analyse approfondie de repo`
2026-05-06 answer 4 + post-push EDIT_REQUIRED): the optional
learned-policy generation path (env var `SAGE_ENABLE_PATH6` for
backward compat) is shipped on `main` as inference-only behind
`SAGE_ENABLE_PATH6=1`. The runtime contract this test BINDS:

  1. Source default-off — no module under `sage-python/src/` may
     set `os.environ["SAGE_ENABLE_PATH6"]` to a truthy value in
     module-level or boot-path code (direct mutation OR setdefault
     to "1" / "true" / etc. is rejected).
  2. Sanctioned channel — the runtime feature-flag allowlist
     (`sage.runtime.run_frame.builder._RunFrameBuilder._ALLOWED_FEATURE_FLAGS`)
     still includes `SAGE_ENABLE_PATH6`, proving the env-var is the
     ONLY sanctioned enabling channel (no config file / hardcoded
     toggle / sys.argv parser etc.).
  3. Registry agreement — `docs/claims/topology.yaml` entry
     `topology.path6_learned` has status `opt-in` (not `delivered`,
     not `default-on`).

EXPLICIT NON-PROMISE (cgpro 2026-05-06 trap #3): this test does NOT
prove the runtime path itself short-circuits when `SAGE_ENABLE_PATH6`
is unset. That assertion requires a Python-side seam on the
learned-policy callsite (which currently lives in Rust) and is
deferred to Phase 2.1/2.2 of cycle-13 K, when the topology engine
refactor exposes such a seam. The contract this file ships is
narrower-but-binding: source-default-off + sanctioned-channel +
registry-agreement.

If a future PR ever flips any of (1)-(3), this test fails and the
ALIRE-driven default-off contract is preserved as a hard gate.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "sage-python" / "src" / "sage"


def test_path6_claim_status_is_opt_in() -> None:
    """The claim registry's `topology.path6_learned` MUST stay opt-in."""
    claims_yaml = _REPO_ROOT / "docs" / "claims" / "topology.yaml"
    if not claims_yaml.is_file():
        pytest.skip(f"{claims_yaml} not present in this checkout")
    payload = yaml.safe_load(claims_yaml.read_text(encoding="utf-8"))
    matches = [c for c in payload.get("claims", []) if c.get("id") == "topology.path6_learned"]
    assert len(matches) == 1, "topology.path6_learned claim not found"
    assert matches[0]["status"] == "opt-in", (
        f"Path 6 claim drifted to status={matches[0]['status']!r}; ALIRE contract requires `opt-in`."
    )


def test_runtime_feature_flag_allowlist_includes_path6() -> None:
    """Only sanctioned enabling channel: the env-var allowlisted by RunFrameBuilder.

    The class is internal (`_RunFrameBuilder`); accessed via attribute lookup so
    the test stays robust to a future visibility change.
    """
    from sage.runtime.run_frame import builder as builder_mod

    builder_cls = getattr(builder_mod, "_RunFrameBuilder", None) or getattr(
        builder_mod, "RunFrameBuilder", None
    )
    assert builder_cls is not None, "RunFrameBuilder class missing from runtime.run_frame.builder"
    allowed = getattr(builder_cls, "_ALLOWED_FEATURE_FLAGS", frozenset())
    assert "SAGE_ENABLE_PATH6" in allowed, (
        "SAGE_ENABLE_PATH6 dropped from the RunFrameBuilder allowlist — Path 6 "
        "now has no sanctioned enabling channel; investigate before merging."
    )


def test_no_source_file_implicitly_enables_path6() -> None:
    """No code under sage-python/src/ may set SAGE_ENABLE_PATH6 to truthy.

    Patterns rejected:
      - os.environ["SAGE_ENABLE_PATH6"] = "1" / "true" / etc.
      - os.environ.setdefault("SAGE_ENABLE_PATH6", "1")
      - Any single-line assignment whose RHS is a quoted truthy literal.

    Patterns allowed:
      - Reading the env-var (os.environ.get("SAGE_ENABLE_PATH6"))
      - Setting it to "0" / "false" / unset / empty (defensive turnoff)
      - Tests under sage-python/tests/ are out of scope.
    """
    # Truthy values: digit literals other than 0, common true/yes/on/enable* tokens.
    rhs_truthy = r"['\"](1|2|3|4|5|6|7|8|9|true|TRUE|True|yes|YES|on|ON|enable|enabled|ENABLE|ENABLED)['\"]"
    forbidden_patterns = [
        re.compile(
            rf"os\.environ\[\s*['\"]SAGE_ENABLE_PATH6['\"]\s*\]\s*=\s*{rhs_truthy}"
        ),
        re.compile(
            rf"os\.environ\.setdefault\(\s*['\"]SAGE_ENABLE_PATH6['\"]\s*,\s*{rhs_truthy}"
        ),
    ]

    offenders: list[tuple[Path, int, str]] = []
    for py_file in _SRC_DIR.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pattern in forbidden_patterns:
                if pattern.search(line):
                    offenders.append((py_file, line_no, line.strip()))

    assert not offenders, (
        "Source files implicitly enable Path 6 — ALIRE default-off contract violated:\n"
        + "\n".join(f"  {p}:{n}: {line}" for p, n, line in offenders)
    )
