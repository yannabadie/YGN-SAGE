"""Tests for `sage.ops.sage_core_version` — Rust binary freshness check.

Cycle-13 B Q1 follow-up to Rust commit `b035973e` (cgpro post-push
2026-05-06 NEXT_BLOCK_ID=G). The Rust side exposes
`sage_core.__commit_sha__` etc.; this module consumes them. Tests
mock `sage_core` attributes + `subprocess.run` so the full matrix
of {binary known, binary unknown} × {source known, source unknown,
source matches, source differs} is exercised without a real git
repo or a real .pyd.

Per cgpro HARD_STOP 2026-05-06 (round 2): `get_source_head_sha`
guards against running from an unrelated git repo via
`git rev-parse --show-toplevel` + sentinel-file check. Tests cover
both the happy path (sentinels present -> SHA returned) and the
guard (sentinels absent -> "unknown").
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

import pytest

from sage.ops import sage_core_version


def _make_subprocess_result(stdout: str) -> MagicMock:
    """Helper: build a mock subprocess.run return value."""
    result = MagicMock()
    result.stdout = stdout
    return result


def _seed_ygn_sage_sentinels(root: Path) -> None:
    """Helper: create the YGN-SAGE sentinel files under `root`."""
    for sentinel in sage_core_version._YGN_SAGE_SENTINEL_FILES:
        target = root / sentinel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("sentinel", encoding="utf-8")


@pytest.fixture
def fake_sage_core(monkeypatch: pytest.MonkeyPatch):
    """Inject a fake `sage_core` module with controllable build attrs."""
    import sys

    def _make(commit_sha: str = "abc123", timestamp: str = "1778055279",
              profile: str = "release", version: str = "0.1.0") -> Any:
        mod = MagicMock()
        mod.__commit_sha__ = commit_sha
        mod.__build_timestamp__ = timestamp
        mod.__build_profile__ = profile
        mod.__version__ = version
        mod.__file__ = "/fake/path/sage_core.pyd"
        monkeypatch.setitem(sys.modules, "sage_core", mod)
        return mod

    return _make


# ── get_build_info() ─────────────────────────────────────────────────────────


def test_get_build_info_reads_all_4_attrs(fake_sage_core) -> None:
    fake_sage_core(
        commit_sha="abc123def456",
        timestamp="1778055279",
        profile="release",
        version="0.1.0",
    )

    info = sage_core_version.get_build_info()

    assert info["commit_sha"] == "abc123def456"
    assert info["build_timestamp"] == "1778055279"
    assert info["build_profile"] == "release"
    assert info["version"] == "0.1.0"
    assert info["module_path"] == "/fake/path/sage_core.pyd"


def test_get_build_info_falls_back_to_unknown_for_missing_attrs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-`b035973e` wheels won't have these attrs — fall back gracefully."""
    import sys

    bare_mod = MagicMock(spec=[])  # spec=[] -> no attributes
    monkeypatch.setitem(sys.modules, "sage_core", bare_mod)

    info = sage_core_version.get_build_info()

    assert info["commit_sha"] == "unknown"
    assert info["build_timestamp"] == "unknown"
    assert info["build_profile"] == "unknown"
    assert info["version"] == "unknown"


def test_get_build_info_returns_unknown_when_sage_core_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No sage_core installed at all — every field is unknown."""
    import sys

    # Force ImportError by removing sage_core + raising on re-import.
    monkeypatch.setitem(sys.modules, "sage_core", None)

    info = sage_core_version.get_build_info()

    for key in ("commit_sha", "build_timestamp", "build_profile", "version", "module_path"):
        assert info[key] == "unknown", f"{key} should be unknown when sage_core absent"


# ── get_source_head_sha() ────────────────────────────────────────────────────


def test_get_source_head_sha_returns_git_output(tmp_path: Path) -> None:
    """Happy path: git toplevel resolves AND has YGN-SAGE sentinels."""
    _seed_ygn_sage_sentinels(tmp_path)

    side_effects = [
        _make_subprocess_result(f"{tmp_path}\n"),     # --show-toplevel
        _make_subprocess_result("deadbeefcafe1234\n"),  # HEAD
    ]
    with patch("subprocess.run", side_effect=side_effects) as mock_run:
        sha = sage_core_version.get_source_head_sha()

    assert sha == "deadbeefcafe1234"
    assert mock_run.call_count == 2
    # Second call was rev-parse HEAD with cwd=toplevel (cgpro HARD_STOP fix).
    second_call_args = mock_run.call_args_list[1]
    assert second_call_args[0][0] == ["git", "rev-parse", "HEAD"]
    assert second_call_args[1]["cwd"] == str(tmp_path)


def test_get_source_head_sha_unknown_when_git_absent() -> None:
    with patch("subprocess.run", side_effect=FileNotFoundError("git")):
        assert sage_core_version.get_source_head_sha() == "unknown"


def test_get_source_head_sha_unknown_when_not_in_repo() -> None:
    with patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(128, ["git", "rev-parse", "--show-toplevel"]),
    ):
        assert sage_core_version.get_source_head_sha() == "unknown"


def test_get_source_head_sha_unknown_when_timeout() -> None:
    with patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(["git"], 5),
    ):
        assert sage_core_version.get_source_head_sha() == "unknown"


def test_get_source_head_sha_unknown_when_unrelated_git_repo(tmp_path: Path) -> None:
    """cgpro HARD_STOP 2026-05-06 (round 2): operator runs the helper
    from inside a DIFFERENT git repo. The helper MUST NOT compare
    YGN-SAGE wheel's commit_sha against the unrelated repo's HEAD —
    that would falsely flag the wheel as stale."""
    # tmp_path is a "git repo" (rev-parse succeeds) but has NO YGN-SAGE
    # sentinel files. The guard should reject it.
    side_effects = [
        _make_subprocess_result(f"{tmp_path}\n"),  # --show-toplevel
        # HEAD call MUST NOT be reached.
    ]
    with patch("subprocess.run", side_effect=side_effects) as mock_run:
        sha = sage_core_version.get_source_head_sha()

    assert sha == "unknown"
    assert mock_run.call_count == 1, (
        "second subprocess (rev-parse HEAD) must be skipped when "
        "sentinels are absent — otherwise the unrelated repo's SHA "
        "leaks into the freshness comparison"
    )


def test_get_source_head_sha_returns_unknown_when_only_one_sentinel(
    tmp_path: Path,
) -> None:
    """Defense-in-depth: ALL sentinels must exist. A directory that
    has `sage-core/Cargo.toml` but not `sage-python/src/sage/__init__.py`
    is not enough — the YGN-SAGE structure is intentional."""
    (tmp_path / "sage-core").mkdir()
    (tmp_path / "sage-core" / "Cargo.toml").write_text("partial", encoding="utf-8")
    # Intentionally do NOT create the python sentinel.

    side_effects = [_make_subprocess_result(f"{tmp_path}\n")]
    with patch("subprocess.run", side_effect=side_effects):
        assert sage_core_version.get_source_head_sha() == "unknown"


def test_get_source_head_sha_unknown_when_head_lookup_fails_post_validation(
    tmp_path: Path,
) -> None:
    """Edge case: toplevel passes sentinel check, but the HEAD lookup
    fails (e.g. corrupt ref, ENOENT race). Must still return unknown,
    not raise."""
    _seed_ygn_sage_sentinels(tmp_path)

    side_effects = [
        _make_subprocess_result(f"{tmp_path}\n"),  # --show-toplevel succeeds
        subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"]),  # HEAD fails
    ]
    with patch("subprocess.run", side_effect=side_effects):
        assert sage_core_version.get_source_head_sha() == "unknown"


# ── check_freshness() ────────────────────────────────────────────────────────


def test_check_freshness_matches_when_shas_equal(
    fake_sage_core, tmp_path: Path
) -> None:
    fake_sage_core(commit_sha="abc123")
    _seed_ygn_sage_sentinels(tmp_path)
    side_effects = [
        _make_subprocess_result(f"{tmp_path}\n"),  # --show-toplevel
        _make_subprocess_result("abc123\n"),       # HEAD
    ]
    with patch("subprocess.run", side_effect=side_effects):
        info = sage_core_version.check_freshness()

    assert info["matches"] is True
    assert info["commit_sha"] == "abc123"
    assert info["source_head_sha"] == "abc123"


def test_check_freshness_stale_when_shas_differ(
    fake_sage_core, tmp_path: Path
) -> None:
    """The exact case that motivated this feature: 2026-04-27 binary
    + 2026-04-30 source = stale."""
    fake_sage_core(commit_sha="0000000_old_binary_sha")
    _seed_ygn_sage_sentinels(tmp_path)
    side_effects = [
        _make_subprocess_result(f"{tmp_path}\n"),
        _make_subprocess_result("ffffffffffff_new_source_sha\n"),
    ]
    with patch("subprocess.run", side_effect=side_effects):
        info = sage_core_version.check_freshness()

    assert info["matches"] is False
    assert info["commit_sha"] == "0000000_old_binary_sha"
    assert info["source_head_sha"] == "ffffffffffff_new_source_sha"


def test_check_freshness_unknown_when_binary_sha_unknown(
    fake_sage_core, tmp_path: Path
) -> None:
    """Pre-b035973e wheel — no SHA exposed by binary."""
    fake_sage_core(commit_sha="unknown")
    _seed_ygn_sage_sentinels(tmp_path)
    side_effects = [
        _make_subprocess_result(f"{tmp_path}\n"),
        _make_subprocess_result("abc123\n"),
    ]
    with patch("subprocess.run", side_effect=side_effects):
        info = sage_core_version.check_freshness()

    assert info["matches"] is None


def test_check_freshness_unknown_when_source_sha_unknown(
    fake_sage_core,
) -> None:
    """Installed via PyPI; no .git in cwd."""
    fake_sage_core(commit_sha="abc123")
    with patch("subprocess.run", side_effect=FileNotFoundError("git")):
        info = sage_core_version.check_freshness()

    assert info["matches"] is None
    assert info["commit_sha"] == "abc123"
    assert info["source_head_sha"] == "unknown"


def test_check_freshness_unknown_when_in_unrelated_repo(
    fake_sage_core, tmp_path: Path
) -> None:
    """cgpro HARD_STOP guard: running from another git repo (no
    YGN-SAGE sentinels) MUST set source_head_sha to unknown rather
    than the unrelated repo's HEAD."""
    fake_sage_core(commit_sha="abc123")
    # tmp_path is a "git repo" but no sentinels.
    side_effects = [
        _make_subprocess_result(f"{tmp_path}\n"),  # --show-toplevel
        # No second call expected — guard skips HEAD lookup.
    ]
    with patch("subprocess.run", side_effect=side_effects):
        info = sage_core_version.check_freshness()

    assert info["source_head_sha"] == "unknown"
    assert info["matches"] is None
    # commit_sha from binary is preserved even when source can't be
    # determined.
    assert info["commit_sha"] == "abc123"


def test_check_freshness_humanizes_build_timestamp(fake_sage_core) -> None:
    """ISO-8601 UTC for human display alongside the raw UNIX seconds."""
    fake_sage_core(timestamp="1778055279")
    with patch("subprocess.run", side_effect=FileNotFoundError("git")):
        info = sage_core_version.check_freshness()

    # 1778055279 = 2026-05-06 something UTC
    iso = info["build_timestamp_iso"]
    assert iso.startswith("2026-05-")
    assert iso.endswith("+00:00")
    # Raw value still preserved.
    assert info["build_timestamp"] == "1778055279"


def test_check_freshness_humanizes_unknown_timestamp(fake_sage_core) -> None:
    fake_sage_core(timestamp="unknown")
    with patch("subprocess.run", side_effect=FileNotFoundError("git")):
        info = sage_core_version.check_freshness()

    assert info["build_timestamp_iso"] == "unknown"


def test_check_freshness_humanizes_malformed_timestamp(fake_sage_core) -> None:
    fake_sage_core(timestamp="not_a_number")
    with patch("subprocess.run", side_effect=FileNotFoundError("git")):
        info = sage_core_version.check_freshness()

    assert info["build_timestamp_iso"] == "unknown"
    assert info["build_timestamp"] == "not_a_number"  # raw preserved


def test_check_freshness_humanizes_overflow_timestamp(fake_sage_core) -> None:
    """cgpro deep VERIFY 2026-05-06 Q3: extreme timestamp MUST NOT
    crash the ops CLI. Year-9999+ values raise OverflowError on
    Windows; year-1 values may raise OSError on some platforms."""
    # 10**18 seconds > year 32 billion — OverflowError on most platforms.
    fake_sage_core(timestamp="1000000000000000000")
    with patch("subprocess.run", side_effect=FileNotFoundError("git")):
        info = sage_core_version.check_freshness()

    assert info["build_timestamp_iso"] == "unknown", (
        "OverflowError from datetime.fromtimestamp must be swallowed"
    )
    assert info["build_timestamp"] == "1000000000000000000"


# ── main() / CLI ─────────────────────────────────────────────────────────────


def test_main_exit_0_when_matches(
    fake_sage_core, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    fake_sage_core(commit_sha="abc123")
    _seed_ygn_sage_sentinels(tmp_path)
    side_effects = [
        _make_subprocess_result(f"{tmp_path}\n"),
        _make_subprocess_result("abc123\n"),
    ]
    with patch("subprocess.run", side_effect=side_effects):
        rc = sage_core_version.main([])

    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["matches"] is True


def test_main_exit_1_when_stale(
    fake_sage_core, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Operator-visible signal: stale binary, rebuild needed."""
    fake_sage_core(commit_sha="old_sha")
    _seed_ygn_sage_sentinels(tmp_path)
    side_effects = [
        _make_subprocess_result(f"{tmp_path}\n"),
        _make_subprocess_result("new_sha\n"),
    ]
    with patch("subprocess.run", side_effect=side_effects):
        rc = sage_core_version.main([])

    assert rc == 1, "stale binary MUST exit 1 — operators rely on this for shell guards"
    payload = json.loads(capsys.readouterr().out)
    assert payload["matches"] is False


def test_main_exit_0_when_unknown_default(
    fake_sage_core, capsys: pytest.CaptureFixture[str]
) -> None:
    """No git, can't compare — default mode does NOT block."""
    fake_sage_core(commit_sha="abc123")
    with patch("subprocess.run", side_effect=FileNotFoundError("git")):
        rc = sage_core_version.main([])

    assert rc == 0, "unknown comparison must default to exit 0 (informational)"
    payload = json.loads(capsys.readouterr().out)
    assert payload["matches"] is None


def test_main_exit_1_when_unknown_strict(
    fake_sage_core, capsys: pytest.CaptureFixture[str]
) -> None:
    """--strict flips unknown to a failure — useful in CI release pipelines
    that want a published wheel to ALWAYS know its provenance."""
    fake_sage_core(commit_sha="abc123")
    with patch("subprocess.run", side_effect=FileNotFoundError("git")):
        rc = sage_core_version.main(["--strict"])

    assert rc == 1


def test_main_quiet_suppresses_json_output(
    fake_sage_core, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    fake_sage_core(commit_sha="abc123")
    _seed_ygn_sage_sentinels(tmp_path)
    side_effects = [
        _make_subprocess_result(f"{tmp_path}\n"),
        _make_subprocess_result("abc123\n"),
    ]
    with patch("subprocess.run", side_effect=side_effects):
        rc = sage_core_version.main(["--quiet"])

    assert rc == 0
    assert capsys.readouterr().out == ""
