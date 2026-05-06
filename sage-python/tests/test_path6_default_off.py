"""Path 6 default-off non-regression test.

Cycle-13 K Phase 0.5 (cgpro `Analyse approfondie de repo` 2026-05-06
answer 4): Path 6 (learned topology policy) is shipped on `main` as
inference-only behind `SAGE_ENABLE_PATH6=1`. Per the cgpro contract,
keeping it on `main` requires:

  1. Feature flag exists and is read-only-on-set (already true).
  2. Default-off (no auto-enable in any boot path).
  3. CLAIMS.yaml entry `topology.path6_learned` with status `opt-in`.
  4. THIS test — non-regression that asserts (a) the registry agrees
     on `opt-in`, (b) no source file implicitly sets
     `os.environ["SAGE_ENABLE_PATH6"] = "1"` (or equivalent) during
     boot or normal runtime paths.

If a future PR ever flips the default, this test fails and the
ALIRE-driven default-off contract is preserved as a hard gate.

Concrete contract checked here:
  - `topology.path6_learned` is in `docs/claims/topology.yaml` with
    status == "opt-in".
  - No source file under `sage-python/src/` mutates
    `os.environ["SAGE_ENABLE_PATH6"]` to truthy ("1", "true", ...)
    in module-level or boot-path code. CI tests may set the env var
    to exercise the opt-in path; that's allowed because tests live
    under `tests/` not `src/`.
  - The runtime feature-flag allowlist
    (`sage.runtime.run_frame.builder.RunFrameBuilder._ALLOWED_FEATURE_FLAGS`)
    still includes `SAGE_ENABLE_PATH6` — proving that the only
    sanctioned channel for enabling Path 6 is the env-var, NOT a
    config file or a hardcoded toggle.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

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
