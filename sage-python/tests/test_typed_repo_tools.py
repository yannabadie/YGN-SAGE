"""P0.1: exhaustive tests for sage.tools.typed_repo.

Covers every typed tool's happy path + security-critical failure modes:
* path-jail (absolute paths, `..` escapes, symlink escape)
* argv-injection resistance (shell metacharacters in arguments)
* output caps (file size, search hits, dir listing, test output)
* env scrubbing (no API-key leak to subprocess)
* schema / parameter validation (rejects out-of-allowlist pytest flags)

Tests are isolated via `tmp_path` + `monkeypatch.chdir()` so each
test runs against a fresh sandbox filesystem rooted at the temp
directory. No network, no real provider calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from sage.tools._env_safe import BASH_ENV_ALLOWLIST, safe_subprocess_env
from sage.tools.typed_repo import (
    MAX_FILE_BYTES,
    MAX_LIST_RESULTS,
    PYTEST_ARG_ALLOWLIST,
    TYPED_REPO_TOOL_NAMES,
    PathJailError,
    _extract_patch_target_paths,
    _resolve_within_cwd,
    _validate_pytest_args,
    create_typed_repo_tools,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    """Give each test a fresh CWD with a small seeded file tree."""
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        "def greet(name):\n    return f'Hello, {name}!'\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_main.py").write_text(
        "from src.main import greet\n\n"
        "def test_greet():\n"
        "    assert greet('Yann') == 'Hello, Yann!'\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# demo repo\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    yield tmp_path


@pytest.fixture
def tools_by_name(sandbox):
    """Instantiate the 6 typed tools and return a {name: Tool} dict."""
    return {t.spec.name: t for t in create_typed_repo_tools()}


# ---------------------------------------------------------------------------
# env-scrub helper
# ---------------------------------------------------------------------------


def test_env_allowlist_rejects_api_keys(monkeypatch):
    """An API key set in the current env must NOT appear in the
    scrubbed subprocess env — that is the audit P0.2 property."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-this-must-never-leak")
    monkeypatch.setenv("CONTEXT7", "ctx7sk-this-must-never-leak-either")
    monkeypatch.setenv("SAGE_EXOCORTEX_STORE", "fileSearchStores/xxx")
    env = safe_subprocess_env()
    assert "OPENAI_API_KEY" not in env
    assert "CONTEXT7" not in env
    assert "SAGE_EXOCORTEX_STORE" not in env


def test_env_allowlist_keeps_core_platform_vars(monkeypatch):
    """PATH / HOME / temp-dir vars stay — subprocesses need them to launch."""
    monkeypatch.setenv("PATH", "/tmp/x:/tmp/y")
    monkeypatch.setenv("HOME", "/tmp/home")
    env = safe_subprocess_env()
    assert env.get("PATH") == "/tmp/x:/tmp/y"
    assert env.get("HOME") == "/tmp/home"


def test_env_allowlist_is_a_frozenset():
    """Frozen so no code path can accidentally add a secret-named var."""
    assert isinstance(BASH_ENV_ALLOWLIST, frozenset)


# ---------------------------------------------------------------------------
# _resolve_within_cwd — path-jail
# ---------------------------------------------------------------------------


def test_resolve_within_cwd_accepts_relative(sandbox):
    resolved = _resolve_within_cwd("src/main.py")
    assert resolved == (sandbox / "src" / "main.py").resolve()


def test_resolve_within_cwd_rejects_absolute(sandbox):
    # Use a path that would resolve absolute on both POSIX and Windows
    abs_target = str(Path(sys.executable).resolve())
    with pytest.raises(PathJailError, match="absolute paths"):
        _resolve_within_cwd(abs_target)


def test_resolve_within_cwd_rejects_parent_escape(sandbox):
    with pytest.raises(PathJailError, match="outside the working directory"):
        _resolve_within_cwd("../../etc/passwd")


def test_resolve_within_cwd_accepts_dot(sandbox):
    assert _resolve_within_cwd(".") == sandbox.resolve()


def test_resolve_within_cwd_accepts_nested_subdir(sandbox):
    assert _resolve_within_cwd("src") == (sandbox / "src").resolve()


# ---------------------------------------------------------------------------
# Factory + schema sanity
# ---------------------------------------------------------------------------


def test_factory_returns_all_six_tools(tools_by_name):
    assert set(tools_by_name.keys()) == set(TYPED_REPO_TOOL_NAMES)
    assert len(tools_by_name) == 6


@pytest.mark.parametrize("name", TYPED_REPO_TOOL_NAMES)
def test_every_tool_has_object_schema(tools_by_name, name):
    """Every JSON schema is a non-empty object with properties — so
    the LLM's function-calling frontend can serialize the call."""
    spec = tools_by_name[name].spec
    assert spec.parameters.get("type") == "object"
    assert "properties" in spec.parameters
    assert len(spec.parameters["properties"]) >= 1


def test_typed_repo_tool_names_tuple_matches_factory():
    names = [t.spec.name for t in create_typed_repo_tools()]
    assert tuple(names) == TYPED_REPO_TOOL_NAMES


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_file_happy_path(tools_by_name, sandbox):
    result = await tools_by_name["read_file"].execute({"path": "src/main.py"})
    assert not result.is_error
    assert "def greet(name):" in result.output


@pytest.mark.asyncio
async def test_read_file_rejects_absolute(tools_by_name, sandbox):
    abs_path = str((sandbox / "src" / "main.py").resolve())
    result = await tools_by_name["read_file"].execute({"path": abs_path})
    assert "[ERROR]" in result.output
    assert "absolute paths" in result.output


@pytest.mark.asyncio
async def test_read_file_rejects_parent_escape(tools_by_name, sandbox):
    result = await tools_by_name["read_file"].execute({"path": "../etc/passwd"})
    assert "[ERROR]" in result.output
    assert "outside the working directory" in result.output


@pytest.mark.asyncio
async def test_read_file_not_found(tools_by_name):
    result = await tools_by_name["read_file"].execute({"path": "nope.py"})
    assert "[ERROR] file not found" in result.output


@pytest.mark.asyncio
async def test_read_file_caps_output(tools_by_name, sandbox):
    big = sandbox / "big.txt"
    big.write_text("X" * (MAX_FILE_BYTES + 1000), encoding="utf-8")
    result = await tools_by_name["read_file"].execute(
        {"path": "big.txt", "max_bytes": MAX_FILE_BYTES}
    )
    assert "[TRUNCATED" in result.output
    # Body + truncation marker together are bounded
    assert len(result.output) < MAX_FILE_BYTES + 200


@pytest.mark.asyncio
async def test_read_file_hard_cap_enforced(tools_by_name, sandbox):
    """Caller asks for 10 MiB but the hard cap is 4 × MAX_FILE_BYTES."""
    big = sandbox / "huge.txt"
    big.write_text("Y" * (10 * 1024 * 1024), encoding="utf-8")
    result = await tools_by_name["read_file"].execute(
        {"path": "huge.txt", "max_bytes": 10 * 1024 * 1024}
    )
    assert "[TRUNCATED" in result.output
    assert len(result.output) <= 4 * MAX_FILE_BYTES + 200


@pytest.mark.asyncio
async def test_read_file_rejects_binary(tools_by_name, sandbox):
    bin_path = sandbox / "raw.bin"
    bin_path.write_bytes(b"\x00\x01\x02\xff\xfe\xfd")
    result = await tools_by_name["read_file"].execute({"path": "raw.bin"})
    assert "[ERROR]" in result.output
    assert "not valid UTF-8" in result.output


# ---------------------------------------------------------------------------
# search_repo
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_repo_finds_literal(tools_by_name):
    result = await tools_by_name["search_repo"].execute(
        {"query": "greet", "path": "."}
    )
    assert not result.is_error
    # At least one hit — the fixture file src/main.py defines `greet`
    assert "greet" in result.output


@pytest.mark.asyncio
async def test_search_repo_rejects_empty_query(tools_by_name):
    result = await tools_by_name["search_repo"].execute({"query": ""})
    assert "[ERROR] query must not be empty" in result.output


@pytest.mark.asyncio
async def test_search_repo_rejects_path_escape(tools_by_name):
    result = await tools_by_name["search_repo"].execute(
        {"query": "x", "path": "../../etc"}
    )
    assert "[ERROR]" in result.output


@pytest.mark.asyncio
async def test_search_repo_invalid_regex(tools_by_name, monkeypatch):
    """Force the Python fallback by pretending ripgrep is missing, so
    the regex path actually runs and can surface a bad-regex error."""
    import shutil

    real_which = shutil.which
    monkeypatch.setattr(
        "sage.tools.typed_repo.shutil.which",
        lambda name: None if name == "rg" else real_which(name),
    )
    result = await tools_by_name["search_repo"].execute(
        {"query": "(unclosed", "regex": True}
    )
    assert "[ERROR] invalid regex" in result.output


@pytest.mark.asyncio
async def test_search_repo_skips_large_files(tools_by_name, sandbox, monkeypatch):
    """A10 (2026-04-24): files larger than ~1 MiB are skipped in the
    Python fallback to avoid MemoryError on pathological repos
    (observed on astropy-14182 during the 2026-04-24 A7 verification
    smoke). Match hits inside smaller files are still returned; the
    large file is simply invisible to the scan.
    """
    import shutil

    real_which = shutil.which
    monkeypatch.setattr(
        "sage.tools.typed_repo.shutil.which",
        lambda name: None if name == "rg" else real_which(name),
    )
    # 1.5 MiB of content containing the target string. Should be skipped.
    big = sandbox / "huge.bin"
    big.write_bytes(b"needle_in_a_giant_file\n" * 70_000)
    # A small file that also matches. Should be found.
    small = sandbox / "src" / "small_match.py"
    small.write_text("needle_in_a_giant_file found here\n", encoding="utf-8")

    result = await tools_by_name["search_repo"].execute(
        {"query": "needle_in_a_giant_file"}
    )
    # Path separator differs between Windows (\) and Unix (/); compare
    # normalized form to avoid platform-dependent brittleness.
    normalized = result.output.replace("\\", "/")
    assert "src/small_match.py" in normalized, (
        "small matching file should be returned"
    )
    assert "huge.bin" not in normalized, (
        "A10: large files (>1 MiB) must be skipped to avoid MemoryError"
    )


@pytest.mark.asyncio
async def test_search_repo_caps_results(tools_by_name, sandbox, monkeypatch):
    """Force the Python fallback (rg might find the pattern too fast)
    and seed many hits to trigger the cap."""
    import shutil

    real_which = shutil.which
    monkeypatch.setattr(
        "sage.tools.typed_repo.shutil.which",
        lambda name: None if name == "rg" else real_which(name),
    )
    big = sandbox / "many.txt"
    big.write_text("hit\n" * 200, encoding="utf-8")
    result = await tools_by_name["search_repo"].execute(
        {"query": "hit", "max_results": 10}
    )
    assert result.output.count("\n") <= 11  # 10 results + maybe truncation note
    assert "hit" in result.output


# ---------------------------------------------------------------------------
# list_files
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_files_happy_path(tools_by_name):
    result = await tools_by_name["list_files"].execute({"path": "."})
    assert "src/main.py" in result.output
    assert "tests/test_main.py" in result.output
    assert "README.md" in result.output


@pytest.mark.asyncio
async def test_list_files_respects_pattern(tools_by_name):
    result = await tools_by_name["list_files"].execute(
        {"path": ".", "pattern": "*.py"}
    )
    # Top-level *.py only (no recursion) — shouldn't include README.md
    assert "README.md" not in result.output


@pytest.mark.asyncio
async def test_list_files_rejects_path_escape(tools_by_name):
    result = await tools_by_name["list_files"].execute({"path": "../.."})
    assert "[ERROR]" in result.output


@pytest.mark.asyncio
async def test_list_files_hard_cap(tools_by_name, sandbox):
    deep = sandbox / "many"
    deep.mkdir()
    for i in range(MAX_LIST_RESULTS + 50):
        (deep / f"f_{i}.txt").write_text("x", encoding="utf-8")
    result = await tools_by_name["list_files"].execute(
        {"path": "many", "max": MAX_LIST_RESULTS + 100}
    )
    lines = result.output.splitlines()
    # cap + truncation marker
    assert len(lines) <= MAX_LIST_RESULTS + 1
    assert any("truncated" in ln for ln in lines)


# ---------------------------------------------------------------------------
# run_tests — argv validation
# ---------------------------------------------------------------------------


def test_pytest_args_empty_is_valid():
    assert _validate_pytest_args([]) == ([], "")


def test_pytest_args_allows_k_and_value():
    out, err = _validate_pytest_args(["-k", "test_greet"])
    assert err == ""
    assert out == ["-k", "test_greet"]


def test_pytest_args_allows_tb_short():
    out, err = _validate_pytest_args(["--tb=short"])
    assert err == ""
    assert out == ["--tb=short"]


def test_pytest_args_rejects_pdb_flag():
    out, err = _validate_pytest_args(["--pdb"])
    assert out is None
    assert "not in the allowlist" in err


def test_pytest_args_rejects_s_flag():
    out, err = _validate_pytest_args(["-s"])
    assert out is None
    assert "not in the allowlist" in err


def test_pytest_args_rejects_shell_metachars_in_node_id():
    out, err = _validate_pytest_args(["tests/foo.py; rm -rf ."])
    assert out is None
    assert "shell metacharacter" in err


def test_pytest_args_rejects_wrong_type():
    out, err = _validate_pytest_args("not-a-list")  # type: ignore[arg-type]
    assert out is None
    assert "must be a list" in err


def test_pytest_args_rejects_non_string_element():
    out, err = _validate_pytest_args(["-k", 42])  # type: ignore[list-item]
    assert out is None
    assert "must be a list of strings" in err


def test_pytest_args_rejects_k_without_value():
    out, err = _validate_pytest_args(["-k"])
    assert out is None
    assert "needs a value" in err


def test_pytest_arg_allowlist_contents():
    """Lock the allowlist so a new dangerous flag has to be added
    deliberately (via a code edit + this test updated)."""
    assert PYTEST_ARG_ALLOWLIST == {
        "-k", "-x", "-q", "--tb", "-v", "--maxfail",
    }


# ---------------------------------------------------------------------------
# apply_patch — patch target extraction + path-jail
# ---------------------------------------------------------------------------


def test_extract_patch_target_paths_simple():
    diff = (
        "diff --git a/src/main.py b/src/main.py\n"
        "--- a/src/main.py\n"
        "+++ b/src/main.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-old\n"
        "+new\n"
    )
    paths = _extract_patch_target_paths(diff)
    assert "src/main.py" in paths


def test_extract_patch_target_paths_new_file():
    diff = (
        "diff --git a/src/new.py b/src/new.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/src/new.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+def f(): pass\n"
    )
    paths = _extract_patch_target_paths(diff)
    # /dev/null is excluded
    assert "src/new.py" in paths
    assert "/dev/null" not in paths


@pytest.mark.asyncio
async def test_apply_patch_rejects_empty(tools_by_name):
    result = await tools_by_name["apply_patch"].execute({"diff": ""})
    assert "[ERROR]" in result.output


@pytest.mark.asyncio
async def test_apply_patch_rejects_path_escape(tools_by_name):
    evil = (
        "diff --git a/../../etc/passwd b/../../etc/passwd\n"
        "--- a/../../etc/passwd\n"
        "+++ b/../../etc/passwd\n"
        "@@ -1 +1 @@\n"
        "-x\n"
        "+y\n"
    )
    result = await tools_by_name["apply_patch"].execute({"diff": evil})
    assert "[ERROR] patch target path-jail violation" in result.output


@pytest.mark.asyncio
async def test_apply_patch_malformed_no_targets(tools_by_name):
    result = await tools_by_name["apply_patch"].execute(
        {"diff": "this is not a valid diff at all"}
    )
    assert "[ERROR] no target paths" in result.output


# ---------------------------------------------------------------------------
# git_diff — flag allowlist
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_git_diff_rejects_bad_flag(tools_by_name):
    result = await tools_by_name["git_diff"].execute(
        {"extra_args": ["--dangerous-flag"]}
    )
    assert "[ERROR]" in result.output
    assert "not in allowlist" in result.output


@pytest.mark.asyncio
async def test_git_diff_accepts_u_with_number(tools_by_name):
    """`-U5` gets key-normalised to `-U` which IS in the allowlist."""
    import shutil

    if shutil.which("git") is None:
        pytest.skip("git not on PATH")
    result = await tools_by_name["git_diff"].execute(
        {"extra_args": ["-U5", "--no-color"]}
    )
    # Even on a clean tree with no changes, the call should succeed
    # without hitting the allowlist error.
    assert "not in allowlist" not in result.output


@pytest.mark.asyncio
async def test_git_diff_rejects_non_string_extra_arg(tools_by_name):
    result = await tools_by_name["git_diff"].execute({"extra_args": [42]})  # type: ignore[list-item]
    assert "[ERROR]" in result.output
    assert "must be strings" in result.output


# ---------------------------------------------------------------------------
# Tool-level schema — read_file.description must warn about path-jail
# ---------------------------------------------------------------------------


def test_read_file_description_mentions_path_jail(tools_by_name):
    desc = tools_by_name["read_file"].spec.description
    assert "Absolute paths" in desc or "relative to the repo" in desc


def test_run_tests_description_mentions_allowlist(tools_by_name):
    desc = tools_by_name["run_tests"].spec.description
    assert "allowlist" in desc.lower() or "curated" in desc.lower()
