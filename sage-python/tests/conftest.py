"""Global test configuration for sage-python tests."""
import asyncio
import os
import sys

import pytest

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
