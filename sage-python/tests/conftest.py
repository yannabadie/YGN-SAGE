"""Global test configuration for sage-python tests."""
import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

# Windows host guard: the default `%LOCALAPPDATA%\Temp\pytest-of-*` root can be
# ACL-denied on this machine. Keep pytest-created temp dirs inside the repo.
_PYTEST_TEMP_ROOT = (
    Path(__file__).resolve().parents[1] / ".tmp" / f"pytest-temp-root-{os.getpid()}"
)
_PYTEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
for _name in ("TMPDIR", "TEMP", "TMP"):
    os.environ.setdefault(_name, str(_PYTEST_TEMP_ROOT))
tempfile.tempdir = str(_PYTEST_TEMP_ROOT)

_PYTEST_HOME = _PYTEST_TEMP_ROOT / "home"
_PYTEST_HOME.mkdir(parents=True, exist_ok=True)
os.environ["HOME"] = str(_PYTEST_HOME)
os.environ["USERPROFILE"] = str(_PYTEST_HOME)
_PYTEST_LOCAL_APPDATA = _PYTEST_HOME / "AppData" / "Local"
_PYTEST_ROAMING_APPDATA = _PYTEST_HOME / "AppData" / "Roaming"
_PYTEST_LOCAL_APPDATA.mkdir(parents=True, exist_ok=True)
_PYTEST_ROAMING_APPDATA.mkdir(parents=True, exist_ok=True)
os.environ["LOCALAPPDATA"] = str(_PYTEST_LOCAL_APPDATA)
os.environ["APPDATA"] = str(_PYTEST_ROAMING_APPDATA)

# Allow SageTopologyEnv to be instantiated in tests without verl-agent.
# See Issue G audit fix: topology_env.py now guards against accidental use
# outside test context when verl-agent is not installed.
os.environ.setdefault("SAGE_TESTING", "1")


# ── Test pollution fix: Windows asyncio policy reset (2026-04-27) ──
#
# Pre-existing 18-failure cascade in `test_execution.py` /
# `test_sandbox*.py` / `test_sandbox_safety.py` was traced to
# `pytest.importorskip("swebench")` at the top of
# `test_swebench_ca_patch.py`. The `swebench` package's import calls
# `asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())`
# on Windows at package-init time — likely upstream defensive code to
# normalise some Linux/POSIX subprocess behaviour.
#
# Side effect: every subsequent `asyncio.create_subprocess_exec` call
# in the same process raises `NotImplementedError` because
# `WindowsSelectorEventLoop` has no subprocess support; only
# `WindowsProactorEventLoop` does. pytest's collection phase imports
# every `test_*.py` module before any test runs, so the swebench
# import (via `importorskip`) fired regardless of the `-k "not
# test_swebench"` filter, polluting the policy for the whole process.
#
# Fix is defence-in-depth on Windows only; macOS/Linux are unaffected:
#   1. `pytest_configure` resets the policy before collection starts
#      (in case anything imported at config-time also mutates it).
#   2. `pytest_collection_modifyitems` resets right after collection
#      finishes — covers the swebench `importorskip` case directly.
#   3. The `_restore_windows_proactor_policy` autouse fixture restores
#      the policy at the start of every test, in case some test sets
#      the Selector policy mid-suite. Belt-and-suspenders.


def pytest_configure(config: pytest.Config) -> None:  # noqa: ARG001
    """Force `WindowsProactorEventLoopPolicy` on Windows before collection."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


def pytest_collection_modifyitems(  # noqa: ARG001
    config: pytest.Config,
    items: list,
) -> None:
    """Restore `WindowsProactorEventLoopPolicy` after collection.

    Some imported modules (notably the `swebench` package, pulled in
    via `pytest.importorskip` at `test_swebench_ca_patch.py`) mutate
    `asyncio.set_event_loop_policy()` to the Selector variant, which
    has no Windows subprocess support. Resetting at the boundary
    between collection and execution means the very first async test
    inherits the correct policy.
    """
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


@pytest.fixture(autouse=True)
def _restore_windows_proactor_policy():
    """Snap-restore Proactor policy on Windows before each test."""
    if sys.platform != "win32":
        yield
        return
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    yield


@pytest.fixture(autouse=True)
def _grant_all_tool_capabilities_in_tests(monkeypatch):
    """Phase 1.5 cycle-13 K (cgpro DESIGN trap 8): isolate ToolPolicy for
    every test.

    Two test-only relaxations applied at session start:

      1. Set the effective `ToolPolicy` to grant ALL six ToolCapability
         tiers, so existing tests that exercise file IO / network mocks
         / subprocess sandboxes / agent delegation don't trip on the
         production default `{pure}`-only policy.

      2. Monkey-patch `sage.tools.registry.resolve_tool_capability` to
         fall back to `ToolCapability.PURE` when the strict resolver
         finds no manifest entry, no class default, and no explicit
         `capability=` kwarg. This lets tests construct ad-hoc tools
         like `Tool(spec=ToolDef(name="tool_a", ...), handler=...)`
         without manually classifying every fixture. Production is
         strict — only the test surface gets this fallback.

    Tests that specifically want to exercise ToolPolicy denial paths
    use their own scoped policies inside `set_current_tool_policy(...)`
    (the ContextVar mechanism is reentrant, so a test-local policy
    overrides this autouse grant for the duration of the with-block /
    `set/reset` pair). Tests that want to exercise registration-denial
    patch `sage.tools.registry.resolve_tool_capability` themselves to
    point at the strict version (`sage.policy.manifest.resolve_tool_capability`).

    Trap: this fixture must NOT mutate operator env-var state.
    `SAGE_TOOL_GRANTS` and `~/.sage/tool_policy.toml` are deliberately
    untouched — only the in-process ContextVar policy and the tools
    registry's resolver attribute are scoped to the test.
    """
    from sage.policy import (
        ToolCapability,
        ToolPolicy,
        set_current_tool_policy,
    )
    from sage.policy.errors import ToolPolicyDeclarationError
    from sage.policy.manifest import resolve_tool_capability as _strict_resolve

    permissive = ToolPolicy(grants=frozenset(ToolCapability))
    token = set_current_tool_policy(permissive)

    def _permissive_resolve(tool):
        try:
            return _strict_resolve(tool)
        except ToolPolicyDeclarationError:
            return ToolCapability.PURE

    # Monkey-patch the resolver alias inside `sage.tools.registry`.
    # `resolve_tool_capability` is imported lazily inside
    # `ToolRegistry.register`, so we must patch the module attribute
    # itself rather than the function reference.
    import sage.tools.registry as _registry_mod
    monkeypatch.setattr(
        _registry_mod, "_resolve_tool_capability_for_tests_only", _permissive_resolve, raising=False
    )

    # Also patch the resolver name as imported by `register`. The
    # `register` body does `from sage.policy.manifest import resolve_tool_capability`
    # — so we patch the source module's name to the permissive variant.
    import sage.policy.manifest as _manifest_mod
    monkeypatch.setattr(_manifest_mod, "resolve_tool_capability", _permissive_resolve)

    try:
        yield
    finally:
        try:
            from sage.policy.tool_policy import _CURRENT_POLICY

            _CURRENT_POLICY.reset(token)
        except (ValueError, LookupError):
            set_current_tool_policy(ToolPolicy.default())


@pytest.fixture(autouse=True)
def _isolate_sage_state_dir():
    """Wipe ``<HOME>/.sage/`` before each test to prevent A14 cross-test pollution.

    Cycle-11 follow-up to cgpro VERIFY round 2026-05-04 Q3 (commit
    147ce18e). Tests that boot a real SAGE Pipeline (via ``sage.boot``
    or directly via ``CognitiveOrchestrationPipeline``) trigger
    ``pipeline.py:3075`` periodic ``engine.save_state(Path.home() /
    ".sage")`` on every ``BANDIT_FLUSH_INTERVAL=10`` pipeline.run()
    call. Without isolation, state files leak into the per-pid
    ``_PYTEST_HOME`` (set above at conftest module load), so a
    downstream test's boot sees ``bandit_state.db / archive_state.db /
    engine_extras.json`` without a matching ``posterior_epoch.json``
    and fail-closes on the A14 epoch guard (CLAUDE.md directive #8,
    invariant 3 in ``docs/contracts/runtime-integrity-ledger.md``).

    The fixture is fixture-driven cleanup (per cgpro acceptance
    criteria), NOT an env-var bypass: ``SAGE_BOOT_BYPASS_EPOCH_GUARD``
    is and stays explicitly forensic-only per directive #8 — silently
    flipping it suite-wide would re-introduce the exact "declared
    label vs verified content" drift directive #9 was written for.

    Tests that legitimately need pre-populated state (e.g.
    ``test_engine_persistence``, ``test_a14_reset``,
    ``test_posterior_epoch``, ``test_boot_topology_epoch_guard``)
    already use their own ``tmp_path`` state dir and are unaffected
    by this wipe.
    """
    sage_dir = Path(os.environ["HOME"]) / ".sage"
    if sage_dir.exists():
        shutil.rmtree(sage_dir, ignore_errors=True)
    yield
    # Post-test cleanup: leave inspection-friendly state for the very
    # last test of the session (atexit handler may also write here),
    # but wipe any residual files BEFORE the next test starts via the
    # pre-yield branch above. No post-yield cleanup needed.


@pytest.fixture
def tmp_path() -> Path:
    """Repo-local tmp_path replacement for Windows ACL-restricted sandboxes."""
    path = _PYTEST_TEMP_ROOT / "cases" / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
