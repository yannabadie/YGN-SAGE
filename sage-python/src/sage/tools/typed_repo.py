"""Typed repository tools — P0.1 of the 2026-04-22 audit remediation.

Replaces the single `execute_bash` tool with six narrow, type-checked
primitives covering the 95% of bench use cases we know about
(SWE-bench repo exploration, pytest runs, patch application, git
diff). Every tool:

* Takes a typed argument dict and validates it before doing anything.
* Passes argv as a ``list`` to ``asyncio.create_subprocess_exec`` —
  never ``shell=True``. No command injection surface.
* Runs under the scrubbed env from :mod:`sage.tools._env_safe` — no
  API keys leak to LLM-generated arguments.
* Caps output (in bytes, count, or both) so a pathological invocation
  cannot flood the model's context.
* Path-jails every ``path`` / ``diff`` input: resolved paths must
  live under the process's current working directory (the checked-out
  repo, for benches). Absolute paths and ``..`` escapes are rejected.

See :file:`docs/superpowers/specs/2026-04-22-safe-sandbox-redesign-spec.md`
for the design decisions behind the tool inventory and the threat
model this replaces.
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
from pathlib import Path

from sage.tools._env_safe import safe_subprocess_env
from sage.tools.base import Tool

# ---------------------------------------------------------------------------
# Global caps — every tool obeys these unless explicitly overridden per-call
# ---------------------------------------------------------------------------

MAX_FILE_BYTES = 32 * 1024          # read_file default + hard cap
MAX_SEARCH_RESULTS = 80             # search_repo hard cap (default 40)
MAX_LIST_RESULTS = 400              # list_files hard cap (default 200)
MAX_TEST_OUTPUT_BYTES = 20 * 1024   # run_tests stdout+stderr cap
MAX_DIFF_OUTPUT_BYTES = 50 * 1024   # git_diff output cap
DEFAULT_TIMEOUT_SECS = 30

# pytest: only these CLI flags are accepted. Everything else is rejected
# so the LLM cannot invoke `--pdb`, `-s` (capture off), plugin hooks,
# doctest collection, etc. Extending this allowlist requires a code edit
# and a regression test.
PYTEST_ARG_ALLOWLIST: set[str] = {
    "-k",           # test name filter
    "-x",           # stop on first failure
    "-q",           # quiet
    "--tb",         # traceback format (short/long/line/no)
    "-v",           # verbose
    "--maxfail",    # max failures
}

# git diff: flag allowlist (diff is read-only, but still only curated flags).
GIT_DIFF_ARG_ALLOWLIST: set[str] = {
    "--stat",
    "--name-only",
    "--name-status",
    "-U",           # context lines count
    "--no-color",
}


# ---------------------------------------------------------------------------
# Path-jail — every tool runs inputs through this
# ---------------------------------------------------------------------------


class PathJailError(ValueError):
    """Raised when an input path would escape the working directory."""


def _resolve_within_cwd(user_path: str | os.PathLike[str]) -> Path:
    """Resolve ``user_path`` relative to CWD and reject escapes.

    Rules:
      * ``None`` is rejected with a clear error.
      * Paths containing embedded null bytes are rejected (cheap
        defense against C-string truncation tricks; Python would
        raise ValueError later anyway, but we surface it as
        PathJailError for a consistent error class).
      * Absolute paths are rejected. Even if the caller means well,
        we never let a tool touch ``/etc/passwd`` or
        ``C:\\Windows``. On Windows, ``/etc/passwd``-style paths
        that start with ``/`` or ``\\`` are ALSO rejected even
        though ``Path.is_absolute()`` returns False on them —
        leading-slash signals intent to use an absolute path, so we
        don't silently rebind it to ``cwd/etc/passwd``.
      * Paths are resolved (symlinks expanded) then checked against
        the current working directory. Anything not underneath CWD
        is rejected (this is what catches symlink escapes).
      * The special tokens ``.`` / empty string resolve to CWD.
      * Normalisation uses :meth:`Path.resolve(strict=False)` so we
        can reason about not-yet-created paths (apply_patch may
        create new files).
    """
    if user_path is None:
        raise PathJailError("path must not be None")
    path_str = str(user_path)
    if "\x00" in path_str:
        raise PathJailError(
            f"path contains an embedded null byte (got {user_path!r})"
        )
    # Defense-in-depth: normalize backslash separators to forward slashes
    # before any further check. On POSIX, `..\..\etc\passwd` is a single
    # filename containing literal backslashes (no traversal), so the
    # cross-platform attack surface only sees Windows-flavored escapes
    # if we don't normalize. After this rewrite, `..\..\etc\passwd`
    # parses as `../../etc/passwd` and gets caught by the relative_to
    # check below.
    path_str = path_str.replace("\\", "/")
    user_path = path_str
    # Leading `/` signals the caller intends an absolute path. (After
    # the backslash-normalize above, `\\` -> `//` also lands here.)
    # On POSIX this is caught by Path.is_absolute(); on Windows it is
    # NOT (Windows needs a drive letter), which would otherwise leave
    # `/etc/passwd` silently rebound to `cwd\etc\passwd`. Reject both.
    if path_str.startswith(("/", "\\")):
        raise PathJailError(
            f"absolute paths are not allowed (got {user_path!r}); "
            "pass a path relative to the working directory"
        )
    p = Path(user_path)
    if p.is_absolute():
        raise PathJailError(
            f"absolute paths are not allowed (got {user_path!r}); "
            "pass a path relative to the working directory"
        )
    cwd = Path(os.getcwd()).resolve()
    resolved = (cwd / p).resolve(strict=False)
    try:
        resolved.relative_to(cwd)
    except ValueError as exc:
        raise PathJailError(
            f"path {user_path!r} resolves to {resolved} which is "
            f"outside the working directory {cwd}"
        ) from exc
    return resolved


# ---------------------------------------------------------------------------
# Subprocess helper — shared argv-list + timeout + output-cap + env-scrub
# ---------------------------------------------------------------------------


async def _run_argv(
    argv: list[str],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECS,
    stdin_data: bytes | None = None,
    max_bytes: int = MAX_TEST_OUTPUT_BYTES,
) -> tuple[int, str]:
    """Run ``argv`` as a subprocess with scrubbed env and bounded output.

    Returns ``(exit_code, combined_output)``. Never raises — timeouts
    and OSErrors map to a non-zero exit code + diagnostic message.
    """
    env = safe_subprocess_env()
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=subprocess.PIPE if stdin_data is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
    except (FileNotFoundError, OSError) as exc:
        return (127, f"[ERROR] failed to launch {argv[0]!r}: {exc}")
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=stdin_data),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        try:
            await asyncio.wait_for(proc.communicate(), timeout=2)
        except asyncio.TimeoutError:
            pass
        return (124, f"[TIMEOUT after {timeout}s]")
    out_text = (stdout or b"").decode("utf-8", errors="replace")
    err_text = (stderr or b"").decode("utf-8", errors="replace")
    combined = out_text
    if err_text:
        combined = combined + ("\n[STDERR]\n" if combined else "") + err_text
    if len(combined) > max_bytes:
        combined = combined[:max_bytes] + f"\n[OUTPUT CAPPED AT {max_bytes} BYTES]"
    return (proc.returncode if proc.returncode is not None else -1, combined)


# ---------------------------------------------------------------------------
# Tool 1 — read_file
# ---------------------------------------------------------------------------


def _build_read_file_tool() -> Tool:
    @Tool.define(
        name="read_file",
        description=(
            "Read a text file from the current working directory. "
            "Path is interpreted relative to the repo / project root. "
            "Absolute paths and `..` escapes are rejected. Output is "
            "capped at 32 KiB by default (override via max_bytes, hard "
            "cap 128 KiB). Binary files return a short error string."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file (e.g. 'src/main.py').",
                },
                "max_bytes": {
                    "type": "integer",
                    "description": (
                        f"Max bytes to return (default {MAX_FILE_BYTES}, "
                        f"hard cap {4 * MAX_FILE_BYTES})."
                    ),
                    "default": MAX_FILE_BYTES,
                },
            },
            "required": ["path"],
        },
    )
    async def read_file(path: str, max_bytes: int = MAX_FILE_BYTES) -> str:
        try:
            resolved = _resolve_within_cwd(path)
        except PathJailError as e:
            return f"[ERROR] {e}"
        if not resolved.exists():
            return f"[ERROR] file not found: {path}"
        if not resolved.is_file():
            return f"[ERROR] path is not a regular file: {path}"
        # Clamp: non-positive / missing values → default; hard cap
        # always applied. Non-positive values indicate a caller bug
        # (or a red-team probe); prefer the safer default to a
        # truncation marker on an empty read.
        try:
            requested = int(max_bytes) if max_bytes else MAX_FILE_BYTES
        except (TypeError, ValueError):
            requested = MAX_FILE_BYTES
        if requested <= 0:
            requested = MAX_FILE_BYTES
        cap = min(requested, 4 * MAX_FILE_BYTES)
        try:
            with resolved.open("rb") as f:
                raw = f.read(cap + 1)
        except OSError as e:
            return f"[ERROR] cannot read {path}: {e}"
        truncated = len(raw) > cap
        raw = raw[:cap]
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return f"[ERROR] file {path} is not valid UTF-8 (likely binary)"
        if truncated:
            text += f"\n[TRUNCATED AT {cap} BYTES]"
        return text

    return read_file


# ---------------------------------------------------------------------------
# Tool 2 — search_repo
# ---------------------------------------------------------------------------


def _build_search_repo_tool() -> Tool:
    @Tool.define(
        name="search_repo",
        description=(
            "Search the repository for a text or regex pattern. Prefers "
            "`ripgrep` (rg) when available; falls back to a Python regex "
            "scan across text files. Results are file:line:match lines, "
            "capped at max_results (default 40, hard cap 80). Query is "
            "passed as a literal argument — no shell interpretation."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Text or regex to search for. When regex=False "
                        "(default), the query is passed to ripgrep's "
                        "fixed-string mode."
                    ),
                },
                "path": {
                    "type": "string",
                    "description": "Relative root to search in (default: '.').",
                    "default": ".",
                },
                "max_results": {
                    "type": "integer",
                    "description": f"Max hits to return (default 40, hard cap {MAX_SEARCH_RESULTS}).",
                    "default": 40,
                },
                "regex": {
                    "type": "boolean",
                    "description": "Treat query as a regex (default False → fixed-string).",
                    "default": False,
                },
            },
            "required": ["query"],
        },
    )
    async def search_repo(
        query: str,
        path: str = ".",
        max_results: int = 40,
        regex: bool = False,
    ) -> str:
        if not query:
            return "[ERROR] query must not be empty"
        try:
            resolved = _resolve_within_cwd(path)
        except PathJailError as e:
            return f"[ERROR] {e}"
        if not resolved.exists():
            return f"[ERROR] search path not found: {path}"

        cap = min(max(1, int(max_results or 40)), MAX_SEARCH_RESULTS)
        rg = shutil.which("rg")
        if rg:
            argv = [rg, "--no-heading", "--line-number", "--max-count", "20"]
            if not regex:
                argv.append("--fixed-strings")
            argv.append("--")
            argv.append(query)
            argv.append(str(resolved))
            code, output = await _run_argv(
                argv, timeout=DEFAULT_TIMEOUT_SECS, max_bytes=MAX_TEST_OUTPUT_BYTES
            )
            if code == 1 and not output.strip():
                return "[0 results]"
            lines = output.splitlines()[:cap]
            if len(lines) < len(output.splitlines()):
                lines.append(f"[... truncated to {cap} results]")
            return "\n".join(lines) if lines else "[0 results]"

        # Fallback: Python scan. Acceptable for small repos; correctness
        # over speed. Mirrors the happy-path output shape.
        try:
            pattern = re.compile(query if regex else re.escape(query))
        except re.error as exc:
            return f"[ERROR] invalid regex: {exc}"
        # A10 (2026-04-24): skip files >1 MiB and catch MemoryError so
        # one pathological file doesn't abort the whole scan. Observed
        # on astropy-14182 in the 2026-04-24 A7 verification smoke:
        # `p.read_text(...)` exhausted memory on a big data file and
        # raised MemoryError, which propagated up as a tool error →
        # agent retried → D8 stall → gen-timeout. Size cap avoids the
        # allocation entirely; per-file exception guard handles any
        # remaining pathology. Source files for modern code are
        # comfortably under 1 MiB; anything larger is almost certainly
        # not what `search_repo` wants to grep through.
        _MAX_FILE_BYTES = 1_048_576  # 1 MiB
        hits: list[str] = []
        for p in resolved.rglob("*"):
            if len(hits) >= cap:
                break
            if not p.is_file():
                continue
            try:
                if p.stat().st_size > _MAX_FILE_BYTES:
                    continue
                text = p.read_text(encoding="utf-8", errors="ignore")
            except (OSError, MemoryError):
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if len(hits) >= cap:
                    break
                if pattern.search(line):
                    rel = p.relative_to(Path(os.getcwd()).resolve())
                    hits.append(f"{rel}:{lineno}:{line.strip()[:200]}")
        if not hits:
            return "[0 results]"
        return "\n".join(hits)

    return search_repo


# ---------------------------------------------------------------------------
# Tool 3 — list_files
# ---------------------------------------------------------------------------


def _build_list_files_tool() -> Tool:
    @Tool.define(
        name="list_files",
        description=(
            "List files matching a glob pattern under a relative root. "
            "Uses Python pathlib's rglob — no shell interpolation. "
            "Capped at max (default 200, hard cap 400)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative root (default '.').",
                    "default": ".",
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (default '**/*').",
                    "default": "**/*",
                },
                "max": {
                    "type": "integer",
                    "description": f"Max paths to return (default 200, hard cap {MAX_LIST_RESULTS}).",
                    "default": 200,
                },
            },
            "required": [],
        },
    )
    async def list_files(
        path: str = ".",
        pattern: str = "**/*",
        max: int = 200,  # noqa: A002 — JSON-schema arg name kept for LLM compat
    ) -> str:
        # Alias immediately so we don't shadow the builtin inside this fn.
        _requested_max = max
        try:
            resolved = _resolve_within_cwd(path)
        except PathJailError as e:
            return f"[ERROR] {e}"
        if not resolved.exists():
            return f"[ERROR] path not found: {path}"
        if not resolved.is_dir():
            return f"[ERROR] not a directory: {path}"
        try:
            requested = int(_requested_max) if _requested_max else 200
        except (TypeError, ValueError):
            requested = 200
        cap = MAX_LIST_RESULTS if requested > MAX_LIST_RESULTS else (1 if requested < 1 else requested)
        pattern = pattern or "**/*"
        try:
            all_matches = list(resolved.rglob(pattern))
        except ValueError as exc:
            # Invalid glob pattern (e.g. "**" not at start)
            return f"[ERROR] invalid pattern {pattern!r}: {exc}"
        if not all_matches:
            return "[0 files]"
        truncated = len(all_matches) > cap
        paths = all_matches[:cap]
        cwd = Path(os.getcwd()).resolve()
        rendered = []
        for p in paths:
            try:
                rel = p.relative_to(cwd)
            except ValueError:
                continue
            # Always render with forward-slashes so LLMs see POSIX-style
            # paths (consistent with diff / git output regardless of
            # Windows vs POSIX host).
            rel_str = str(rel).replace(os.sep, "/")
            if p.is_dir():
                rendered.append(f"{rel_str}/")
            else:
                rendered.append(rel_str)
        if truncated:
            rendered.append(f"[... truncated at {cap} paths]")
        return "\n".join(rendered)

    return list_files


# ---------------------------------------------------------------------------
# Tool 4 — run_tests
# ---------------------------------------------------------------------------


_SHELL_METACHARS: tuple[str, ...] = (";", "|", "&", "`", "$", ">", "<", "\n", "\r")


def _contains_shell_metachar(s: str) -> str | None:
    """Return the first shell metachar found in ``s`` or None."""
    for c in _SHELL_METACHARS:
        if c in s:
            return c
    return None


def _validate_pytest_args(args: list[str]) -> tuple[list[str] | None, str]:
    """Return (validated_args, "") on success or (None, "error message").

    Reject classes (every token runs through the metachar scan, not
    just positionals — see red-team test_pytest_args_rejects_injection
    for the motivating cases):
      * non-list input
      * non-string element
      * flag not in PYTEST_ARG_ALLOWLIST
      * flag value starting with `-` (looks like a smuggled flag)
      * any token containing ``; | & ` $ > < \\n \\r``
      * flag like ``-k=value`` where value contains a shell metachar
      * ``--foo=bar`` where ``--foo`` is not in the allowlist
    """
    if args is None:
        return [], ""
    if not isinstance(args, list):
        return None, f"pytest_args must be a list, got {type(args).__name__}"
    # Early element-type check — fail clearly on non-strings before any
    # flag-allowlist logic runs (otherwise a non-string value attached
    # to a valid flag surfaces as "requires a non-flag value", which
    # muddles the error taxonomy).
    for idx, el in enumerate(args):
        if not isinstance(el, str):
            return None, (
                f"pytest_args must be a list of strings (element {idx} "
                f"is {type(el).__name__})"
            )
    cleaned: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if not isinstance(arg, str):
            return None, f"pytest_args must be a list of strings (element {i} is {type(arg).__name__})"
        if arg.startswith("-"):
            # Long form split: "--tb=short" → key "--tb", val "short"
            if "=" in arg:
                key, _, inline_val = arg.partition("=")
                meta = _contains_shell_metachar(inline_val)
                if meta is not None:
                    return None, (
                        f"pytest flag value in {arg!r} contains a "
                        f"shell metacharacter {meta!r}"
                    )
            else:
                key = arg
            if key not in PYTEST_ARG_ALLOWLIST:
                return None, (
                    f"pytest flag {arg!r} is not in the allowlist "
                    f"{sorted(PYTEST_ARG_ALLOWLIST)}. Edit "
                    "PYTEST_ARG_ALLOWLIST if you need it."
                )
            cleaned.append(arg)
            # Some flags take an arg: `-k pattern`, `--maxfail 3`
            if key in {"-k", "--maxfail", "--tb"} and "=" not in arg:
                if i + 1 >= len(args):
                    return None, f"flag {arg!r} needs a value"
                value = args[i + 1]
                if not isinstance(value, str) or value.startswith("-"):
                    return None, f"flag {arg!r} requires a non-flag value (got {value!r})"
                meta = _contains_shell_metachar(value)
                if meta is not None:
                    return None, (
                        f"pytest flag value {value!r} contains a "
                        f"shell metacharacter {meta!r}"
                    )
                cleaned.append(value)
                i += 2
                continue
        else:
            # Positional arg: a test node id like "tests/foo.py::test_bar".
            # Reject anything that looks like a shell metachar.
            meta = _contains_shell_metachar(arg)
            if meta is not None:
                return None, (
                    f"pytest_args element {arg!r} contains a shell "
                    f"metacharacter {meta!r}"
                )
            cleaned.append(arg)
        i += 1
    return cleaned, ""


def _build_run_tests_tool() -> Tool:
    @Tool.define(
        name="run_tests",
        description=(
            "Run pytest on a relative path. Only a curated set of pytest "
            "flags is accepted: -k, -x, -q, -v, --tb, --maxfail. Other "
            "flags (including --pdb, -s, --doctest-modules, any plugin "
            "option) are rejected before the subprocess runs. Output is "
            "capped at 20 KiB."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the test file or directory (e.g. 'tests/test_foo.py', 'tests/').",
                },
                "pytest_args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of extra pytest arguments, each a string. "
                        "Only flags in PYTEST_ARG_ALLOWLIST plus positional test "
                        "node ids (without shell metacharacters) are accepted."
                    ),
                    "default": [],
                },
            },
            "required": ["path"],
        },
    )
    async def run_tests(path: str, pytest_args: list[str] | None = None) -> str:
        try:
            resolved = _resolve_within_cwd(path)
        except PathJailError as e:
            return f"[ERROR] {e}"
        if not resolved.exists():
            return f"[ERROR] test path not found: {path}"
        validated, err = _validate_pytest_args(pytest_args or [])
        if validated is None:
            return f"[ERROR] {err}"
        python = shutil.which("python") or shutil.which("python3")
        if python is None:
            return "[ERROR] python executable not found on PATH"
        argv = [python, "-m", "pytest", str(resolved)] + list(validated)
        code, output = await _run_argv(
            argv,
            timeout=DEFAULT_TIMEOUT_SECS * 4,   # tests can legitimately take longer
            max_bytes=MAX_TEST_OUTPUT_BYTES,
        )
        prefix = "PASS" if code == 0 else f"FAIL (exit_code={code})"
        return f"[{prefix}]\n{output}"

    return run_tests


# ---------------------------------------------------------------------------
# Tool 5 — apply_patch
# ---------------------------------------------------------------------------


def _extract_patch_target_paths(diff: str) -> list[str]:
    """Pull out target file paths from a unified diff so we can path-jail them."""
    paths: list[str] = []
    for line in diff.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            part = line[4:].strip()
            if part.startswith("a/") or part.startswith("b/"):
                part = part[2:]
            if part and part != "/dev/null":
                paths.append(part)
        elif line.startswith("diff --git "):
            # diff --git a/X b/Y
            rest = line[len("diff --git "):].strip()
            toks = rest.split()
            for tok in toks:
                if tok.startswith("a/") or tok.startswith("b/"):
                    paths.append(tok[2:])
    return paths


def _build_apply_patch_tool() -> Tool:
    @Tool.define(
        name="apply_patch",
        description=(
            "Apply a unified diff to the working directory. Prefers "
            "`git apply --check` / `git apply` when git is available; "
            "falls back to `patch --fuzz=5`. The diff's target paths are "
            "path-jailed — any target that resolves outside CWD aborts "
            "the whole apply. Pass check_only=True to dry-run."
        ),
        parameters={
            "type": "object",
            "properties": {
                "diff": {
                    "type": "string",
                    "description": "Unified diff content (starts with `diff --git` or `--- a/...`).",
                },
                "check_only": {
                    "type": "boolean",
                    "description": "If true, only validate the patch without applying (default False).",
                    "default": False,
                },
            },
            "required": ["diff"],
        },
    )
    async def apply_patch(diff: str, check_only: bool = False) -> str:
        if not isinstance(diff, str) or not diff.strip():
            return "[ERROR] diff must be a non-empty string"
        targets = _extract_patch_target_paths(diff)
        if not targets:
            return "[ERROR] no target paths found in diff (malformed?)"
        for t in targets:
            try:
                _resolve_within_cwd(t)
            except PathJailError as e:
                return f"[ERROR] patch target path-jail violation: {e}"

        git = shutil.which("git")
        patch_bin = shutil.which("patch")
        if git is not None:
            check_args = [git, "apply", "--check", "-"]
            code_chk, out_chk = await _run_argv(
                check_args,
                timeout=DEFAULT_TIMEOUT_SECS,
                stdin_data=diff.encode("utf-8"),
                max_bytes=MAX_DIFF_OUTPUT_BYTES,
            )
            if code_chk != 0:
                return f"[INVALID PATCH]\n{out_chk}"
            if check_only:
                return "[PATCH OK — check_only=True, not applied]"
            apply_args = [git, "apply", "-"]
            code_app, out_app = await _run_argv(
                apply_args,
                timeout=DEFAULT_TIMEOUT_SECS,
                stdin_data=diff.encode("utf-8"),
                max_bytes=MAX_DIFF_OUTPUT_BYTES,
            )
            return f"[APPLIED]\n{out_app}" if code_app == 0 else f"[APPLY FAILED]\n{out_app}"

        if patch_bin is not None:
            # --fuzz=5 tolerates modest line-number drift; --dry-run if check_only
            args = [patch_bin, "-p1", "--fuzz=5"]
            if check_only:
                args.append("--dry-run")
            code, output = await _run_argv(
                args,
                timeout=DEFAULT_TIMEOUT_SECS,
                stdin_data=diff.encode("utf-8"),
                max_bytes=MAX_DIFF_OUTPUT_BYTES,
            )
            if code == 0:
                return "[APPLIED via patch]\n" + output if not check_only else "[PATCH OK via patch --dry-run]"
            return f"[PATCH FAILED]\n{output}"

        return "[ERROR] neither `git` nor `patch` is available on PATH"

    return apply_patch


# ---------------------------------------------------------------------------
# Tool 6 — git_diff
# ---------------------------------------------------------------------------


def _build_git_diff_tool() -> Tool:
    @Tool.define(
        name="git_diff",
        description=(
            "Show the current working-tree diff. Optionally restrict to a "
            "path (path-jailed) and/or the staged changes. Only a curated "
            "set of git diff flags is allowed via extra_args "
            "(--stat, --name-only, --name-status, -U, --no-color). Output "
            "capped at 50 KiB."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Optional relative path to restrict the diff to.",
                    "default": "",
                },
                "staged": {
                    "type": "boolean",
                    "description": "If true, diff the index vs HEAD (default False = working tree vs index).",
                    "default": False,
                },
                "extra_args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional extra git diff flags; only flags in GIT_DIFF_ARG_ALLOWLIST "
                        "are accepted (--stat, --name-only, --name-status, -U<n>, --no-color)."
                    ),
                    "default": [],
                },
            },
            "required": [],
        },
    )
    async def git_diff(
        path: str = "",
        staged: bool = False,
        extra_args: list[str] | None = None,
    ) -> str:
        git = shutil.which("git")
        if git is None:
            return "[ERROR] git not available on PATH"

        argv: list[str] = [git, "diff"]
        if staged:
            argv.append("--cached")

        for arg in (extra_args or []):
            if not isinstance(arg, str):
                return f"[ERROR] extra_args must be strings (got {type(arg).__name__})"
            key = arg.split("=", 1)[0]
            # `-U<n>` collapses key to `-U` (with a number suffix); normalise.
            if arg.startswith("-U") and len(arg) > 2 and arg[2:].isdigit():
                key = "-U"
            if key not in GIT_DIFF_ARG_ALLOWLIST:
                return (
                    f"[ERROR] git diff flag {arg!r} not in allowlist "
                    f"{sorted(GIT_DIFF_ARG_ALLOWLIST)}"
                )
            argv.append(arg)

        if path:
            try:
                resolved = _resolve_within_cwd(path)
            except PathJailError as e:
                return f"[ERROR] {e}"
            argv.append("--")
            argv.append(str(resolved))

        code, output = await _run_argv(
            argv,
            timeout=DEFAULT_TIMEOUT_SECS,
            max_bytes=MAX_DIFF_OUTPUT_BYTES,
        )
        if not output.strip():
            return "[no changes]"
        if code != 0:
            return f"[git diff failed exit_code={code}]\n{output}"
        return output

    return git_diff


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


TYPED_REPO_TOOL_NAMES: tuple[str, ...] = (
    "read_file",
    "search_repo",
    "list_files",
    "run_tests",
    "apply_patch",
    "git_diff",
)


def create_typed_repo_tools() -> list[Tool]:
    """Build and return the 6 typed repository tools.

    Ordered so the LLM sees read_file first (most common call) and
    apply_patch / git_diff last (more specialised). Registration
    order is stable so prompt-affordance listing is deterministic.
    """
    return [
        _build_read_file_tool(),
        _build_search_repo_tool(),
        _build_list_files_tool(),
        _build_run_tests_tool(),
        _build_apply_patch_tool(),
        _build_git_diff_tool(),
    ]
