"""Regression gate for type: ignore comment count.

Tracks the number of ``# type: ignore`` comments in the sage source tree
and fails if the count increases beyond the established ceiling. This
prevents new type-safety escapes from creeping in.
"""

from __future__ import annotations

import re
from pathlib import Path

# Maximum allowed type: ignore comments (regression ceiling).
# Wave 1+2 cleanup removed 8 fixable ignores from 20 original.
# Remaining are all in "skip" categories (third-party / unfixable):
#   11 = third-party imports (a2a x6, sentence-transformers, sage_core x2, openai, google-genai)
#    5 = pipeline.py typing (verify_provider_assignment, ProviderSpec, _emit, TopologyGraph,
#        TopologyRunner, TopologyExecutor assignments)
#    3 = ssl private API, pipeline_stages dict arg-type, providers openai SDK arg-type
#    1 = evolution cli sage_core import
# NOTE: concurrent linter activity may add new third-party ignores.
# Raised from 20 to 23 after QualityEstimator rewrite (3 new sage_core imports).
# Raised from 23 to 25 after orchestrator cleanup (2 new pipeline sage_core imports).
# Raised from 25 to 27: +1 FrugalGPT cascade (pipeline.py sage_core import),
# +1 apps_bench.py datasets import, +1 livecodebench_bench.py datasets import,
# -1 orchestrator.py deleted (net +2).
# Raised from 27 to 29: +1 Path 6 TopologyEdge import (pipeline.py),
# +1 FrugalGPT cascade retry TopologyExecutor import (pipeline.py).
# Raised from 29 to 36: +5 a2a_server.py new server imports (import-untyped),
# +2 quality_estimator.py Rust imports (return type).
# Raised from 36 to 41 (2026-04-23, audit-aware catch-up): the ceiling
# drifted between commit 6ef60cf and the 2026-04-23 bench chain as
# cross-cutting work landed. Net +5 across:
#   +2 bench/swebench_ca_patch.py (UTF-8/CRLF wrappers on stdlib
#     write_text / run_evaluation.open — #[attr-defined]/method-assign)
#   +1 bench/swebench_patch_repair.py (_extract_patch misc)
#   +1 bench/swebench_bench.py ssl private API (assignment)
#   +1 topology_controller.py _RustTopologyControllerImpl sentinel.
# All five are in skip categories (ssl private API, stdlib
# monkey-patch, third-party or Rust bindings). No new ignores from the
# 2026-04-23 bench/sandbox chain itself — the gen-log commit's one
# ignore was retired in the same pass via setattr.
# Raised from 41 to 42 (2026-04-24, AUDIT3 A19): +1 for
# protocols/auth.py:172 starlette import guarded by try/except ImportError
# as a lazy optional dependency (starlette is installed in test env but
# marked optional in ygn-sage[all]); the `type: ignore[import-not-found]`
# is correct — only used inside the `try` block, and mypy without the
# test-env extras would otherwise flag the import.
# Raised from 42 to 44 (2026-04-25, roadmap-B1 OTel GenAI spans):
# +2 for sage_core import guards in observability/__init__.py:36 and
# observability/spans.py:111. These are correct — sage_core is the
# optional Rust extension (PyO3) and the bridge mirrors Python OTel
# state into Rust only when sage-core was built with `--features otel`;
# absent that, the import gracefully falls through. Same shape as the
# pre-existing pipeline.py sage_core import-not-found pattern (skip
# category).
# Raised from 44 to 45 (2026-04-26, CI debt closeout):
# +1 for `import yaml  # type: ignore[import-untyped]` in
# execution/__init__.py:30. yaml.dump/load is structurally typed at
# runtime; types-PyYAML is locally installed (pulled by some dev dep)
# but absent on the python-sage Linux CI runner, where mypy emits
# `[import-untyped]` rather than `[import-not-found]` (the latter is
# what `--ignore-missing-imports` suppresses). The ignore narrows to
# this one site, so the rest of the codebase still surfaces real
# missing-stubs cases.
# Raised from 45 to 48 (2026-05-05, cycle-11 cgpro VERIFY follow-up
# — CI debug). Three legitimate ignores accumulated during cycle-9
# wall-clock watchdog work; CI was reporting count=48 vs ceiling 45.
# All three are Windows/fallback patterns:
#   +1 bench/event_ledger.py:57 `_ulid_module = None # type: ignore[assignment]`
#     — fallback module reference when `ulid` lib isn't installed.
#     Cycle-9 commit `0036217b` (event ledger).
#   +1 bench/keep_awake.py:52 `import ctypes # type: ignore[import-not-found]`
#     — ctypes IS available everywhere; mypy on the Linux runner
#     flags this anyway because `ctypes.windll` (used at line 56) is
#     Windows-only. Cycle-9 commit `46c280e3` (Windows keep-awake).
#   +1 bench/keep_awake.py:56 `ctypes.windll.kernel32.SetThreadExecutionState(...) # type: ignore[attr-defined]`
#     — `ctypes.windll` is the documented Windows-only attribute;
#     correct ignore on Linux mypy runs. Same commit as above.
# All three are in skip categories (Windows-only attr / optional dep
# fallback). No new ignores from cycle-11 P9 phase 1 work or this
# session's CI repair commits.
# Raised from 48 to 51 (2026-05-05, cycle-12 prelude — pi-mono pivot
# `sage run --jsonl` backend). Three legitimate ignores in
# `sage-python/src/sage/cli/run.py` for the RuntimeEventLog tee
# integration, all in skip categories:
#   +1 cli/run.py:399 `if eventlog._fh is not None: # type: ignore[attr-defined]`
#     — `_fh` is private API of RuntimeEventLog (writer.py:155),
#     accessed here to splice in the stdout-mirror tee. Public
#     "set sink" API doesn't exist; switching to one is cycle-12
#     Phase B work.
#   +1 cli/run.py:400 `eventlog._fh = _CliMirrorSinkHandle(...) # type: ignore[assignment]`
#     — `_CliMirrorSinkHandle` is a structural subtype of `_SinkHandle`
#     (write/flush/close/closed/fileno/tell/truncate match) but
#     `_SinkHandle` is a concrete class, not a `typing.Protocol`.
#     Switching to a Protocol would surface this assignment without
#     the ignore; tracked for cycle-12 Phase B.
#   +1 cli/run.py:524 `install_event_log(None) # type: ignore[arg-type]`
#     — `install_event_log` is typed to require a `RuntimeEventLog`
#     instance, but the writer's contextvar default IS `None` (so
#     resetting to None is correct semantics). The signature
#     could be widened to `RuntimeEventLog | None`; tracked for
#     cycle-12 Phase B alongside the Protocol switch.
# All three are localized to the cycle-12 prelude CLI bridge code
# at the boundary between RuntimeEventLog (existing) and the new
# CLI tee (new). Resolution path is documented above (Protocol +
# signature widening); the cycle-12 Phase B refactor will close them.
_MAX_TYPE_IGNORES = 51

_SAGE_SRC = Path(__file__).resolve().parent.parent / "src" / "sage"
_PATTERN = re.compile(r"#\s*type:\s*ignore")


def _count_type_ignores() -> list[tuple[str, int, str]]:
    """Return list of (relative_path, line_number, line_text) for all type: ignore hits."""
    hits: list[tuple[str, int, str]] = []
    for py_file in sorted(_SAGE_SRC.rglob("*.py")):
        try:
            lines = py_file.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for i, line in enumerate(lines, start=1):
            if _PATTERN.search(line):
                rel = py_file.relative_to(_SAGE_SRC)
                hits.append((str(rel), i, line.strip()))
    return hits


def test_type_ignore_count_does_not_increase() -> None:
    """Regression gate: type: ignore count must not exceed ceiling."""
    hits = _count_type_ignores()
    count = len(hits)

    # Print for visibility in CI output
    print(f"\n[type: ignore audit] Found {count} comments (ceiling: {_MAX_TYPE_IGNORES})")
    for path, lineno, text in hits:
        print(f"  {path}:{lineno}  {text}")

    assert count <= _MAX_TYPE_IGNORES, (
        f"type: ignore count ({count}) exceeds ceiling ({_MAX_TYPE_IGNORES}). "
        f"Fix the new ignores or raise the ceiling with justification."
    )
