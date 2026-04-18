"""
Minimal wrapper around `codex exec` for programmatic usage with logging.

Calls Codex CLI via subprocess, parses stream-json output, tracks
agent_messages and command_executions, returns a SessionResult that is
API-shape-compatible with upstream claude_wrapper.SessionResult so the
outer loop (meta_harness.py) stays agnostic.

Design choices:
- Use Codex's own OAuth (no ANTHROPIC_API_KEY / no OPENAI_API_KEY env var
  coupling) → one `codex login` done once, wrapper just invokes.
- Sandbox = "workspace-write" by default so the proposer CAN write new
  agent files to `agents/<id>.py`. Override via `sandbox="read-only"` for
  diagnosis-only runs.
- Tool allowlist is not a flag in Codex (differs from Claude CLI) — the
  sandbox level is the restriction surface. Filesystem sandbox is
  scoped to `cwd` passed via `-C`.
- Model default: `gpt-5.4` with `model_reasoning_effort=high` matching
  the "xhigh" requirement the user specified.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _resolve_codex_exe() -> str:
    """Find the codex executable, preferring .cmd on Windows.

    Python's subprocess on Windows can't execute shell-script `codex`
    (without extension) directly from `CreateProcess`. We resolve to the
    `.cmd` shim if present. Falls back to whatever `shutil.which` finds.
    """
    env_override = os.environ.get("CODEX_BIN")
    if env_override and Path(env_override).exists():
        return env_override
    if os.name == "nt":
        for candidate in (shutil.which("codex.cmd"), shutil.which("codex.exe")):
            if candidate:
                return candidate
    found = shutil.which("codex")
    if found:
        return found
    raise RuntimeError("codex CLI not found on PATH; set CODEX_BIN env var")


# ── Data classes ──────────────────────────────────────────────────

@dataclass
class ToolCall:
    """One tool invocation observed in the Codex event stream.

    Codex reports tool calls as `command_execution` items (shell) and
    other item types (e.g. future Read/Write primitives). We normalise
    a subset into ToolCall for the outer loop.
    """

    name: str
    tool_id: str
    input: dict
    output: str = ""
    is_error: bool = False


@dataclass
class SessionResult:
    """Everything the outer loop needs to know about a Codex session.

    Shape mirrors upstream claude_wrapper.SessionResult so meta_harness.py
    can stay wrapper-agnostic when switching proposers.
    """

    prompt: str
    text: str
    tool_calls: list[ToolCall]
    files_read: dict[str, dict[str, int]]
    files_written: dict[str, dict[str, int]]
    token_usage: dict[str, int]
    duration_seconds: float
    model: str
    session_id: str
    exit_code: int
    cost_usd: float
    raw_events: list[dict]
    command: list[str] | None = None
    cwd: str | None = None
    stderr: str = ""
    skill: dict | None = None
    name: str | None = None
    log_dir: str | None = None

    def show(self) -> None:
        """Print a compact one-line-per-event summary for interactive use."""
        print(f"[codex] model={self.model} dur={self.duration_seconds:.1f}s")
        print(f"        text={len(self.text)} chars, tools={len(self.tool_calls)}")
        print(f"        files_read={list(self.files_read)}")
        print(f"        files_written={list(self.files_written)}")


# ── Helpers ───────────────────────────────────────────────────────

def _slugify(text: str, max_words: int = 4) -> str:
    """Create a short directory-safe slug."""
    words = re.sub(r"[^a-z0-9\s]", "", text.lower()).split()
    return "-".join(words[:max_words]) or "run"


def _make_relative(path: str, cwd: str | Path | None) -> str:
    """Best-effort relative path for logs."""
    if cwd is None:
        return path
    try:
        return os.path.relpath(path, str(cwd))
    except ValueError:
        return path


# ── Core API ──────────────────────────────────────────────────────

def run(
    prompt: str,
    *,
    cwd: Path | str,
    model: str = "gpt-5.4",
    reasoning_effort: str = "high",
    sandbox: str = "workspace-write",
    timeout: float = 600.0,
    skip_git_repo_check: bool = True,
    ephemeral: bool = True,
    log_dir: Path | str | None = None,
    name: str | None = None,
) -> SessionResult:
    """Invoke `codex exec <prompt>` and return a SessionResult.

    Parameters
    ----------
    prompt : str
        The proposer prompt.
    cwd : Path | str
        Working directory for Codex. Sandbox writes are scoped here.
    model : str
        Codex model id. Defaults to `gpt-5.4`.
    reasoning_effort : str
        Codex `-c model_reasoning_effort=...` value. Defaults to `high`
        (the "xhigh" behaviour per user requirement).
    sandbox : str
        `-s` flag value: `read-only` | `workspace-write` | `danger-full-access`.
        Proposer runs that write candidate files need `workspace-write`.
    timeout : float
        Subprocess wall-clock timeout in seconds.
    skip_git_repo_check : bool
        Pass `--skip-git-repo-check` to allow running outside a git repo.
    ephemeral : bool
        Pass `--ephemeral` so Codex doesn't persist session state to disk.
    log_dir : Path | str, optional
        If provided, dump the raw JSONL stream + prompt + result to this dir.
    name : str, optional
        Label used in log filenames.

    Returns
    -------
    SessionResult
    """
    cwd_p = Path(cwd).resolve()
    cwd_p.mkdir(parents=True, exist_ok=True)

    codex_exe = _resolve_codex_exe()
    cmd = [
        codex_exe, "exec",
        prompt,
        "--json",
        "-m", model,
        "-c", f"model_reasoning_effort={reasoning_effort}",
        "-s", sandbox,
        "-C", str(cwd_p),
    ]
    if ephemeral:
        cmd.append("--ephemeral")
    if skip_git_repo_check:
        cmd.append("--skip-git-repo-check")

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        exit_code = proc.returncode
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout or b"").decode("utf-8", errors="replace") if e.stdout else ""
        stderr = (e.stderr or b"").decode("utf-8", errors="replace") if e.stderr else ""
        exit_code = 124  # GNU timeout convention

    duration = time.time() - t0

    result = _parse_stream_events(
        stdout=stdout,
        prompt=prompt,
        model=model,
        duration=duration,
        exit_code=exit_code,
        cwd=str(cwd_p),
    )
    result.command = cmd
    result.cwd = str(cwd_p)
    result.stderr = stderr
    result.name = name

    # Optional persistent log
    if log_dir is not None:
        _persist_log(result, Path(log_dir))

    return result


# ── Stream parser ─────────────────────────────────────────────────

def _parse_stream_events(
    stdout: str,
    prompt: str,
    model: str,
    duration: float,
    exit_code: int,
    cwd: str | None = None,
) -> SessionResult:
    """Parse Codex's JSONL event stream.

    Known Codex event shapes (0.120.0):
      {"type":"thread.started","thread_id":"..."}
      {"type":"turn.started"}
      {"type":"item.started","item":{"id":"item_N","type":"<kind>", ...}}
      {"type":"item.completed","item":{"id":"item_N","type":"<kind>", ...}}

    `<kind>` values we care about:
      - "agent_message"       → assistant text chunk
      - "command_execution"   → bash tool (command, aggregated_output, exit_code, status)
      - Other tool kinds when Codex adds them; we surface them as ToolCall too.
    """
    events: list[dict] = []
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    tool_call_map: dict[str, ToolCall] = {}
    session_id = ""
    cost_usd = 0.0
    token_usage = {"input_tokens": 0, "output_tokens": 0}

    for line in stdout.strip().split("\n") if stdout.strip() else []:
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        events.append(event)
        etype = event.get("type", "")

        if etype == "thread.started":
            session_id = event.get("thread_id", "")
            continue

        if etype == "item.completed":
            item = event.get("item", {}) or {}
            itype = item.get("type", "")
            iid = item.get("id", "")

            if itype == "agent_message":
                text_parts.append(item.get("text", ""))

            elif itype == "command_execution":
                # Shell tool. Map to a ToolCall named "Bash" for API compat
                # with claude_wrapper (outer loop already branches on that).
                tc = tool_call_map.get(iid) or ToolCall(
                    name="Bash",
                    tool_id=iid,
                    input={"command": item.get("command", "")},
                )
                tc.output = item.get("aggregated_output", "") or ""
                tc.is_error = (item.get("exit_code") or 0) != 0
                if iid not in tool_call_map:
                    tool_calls.append(tc)
                    tool_call_map[iid] = tc

            else:
                # Future-proof: any other completed tool kind
                tc = tool_call_map.get(iid) or ToolCall(
                    name=itype or "Tool",
                    tool_id=iid,
                    input=item,
                )
                if iid not in tool_call_map:
                    tool_calls.append(tc)
                    tool_call_map[iid] = tc

        elif etype == "item.started":
            item = event.get("item", {}) or {}
            itype = item.get("type", "")
            iid = item.get("id", "")
            if itype == "command_execution" and iid and iid not in tool_call_map:
                tc = ToolCall(
                    name="Bash",
                    tool_id=iid,
                    input={"command": item.get("command", "")},
                )
                tool_calls.append(tc)
                tool_call_map[iid] = tc

        elif etype in ("turn.completed", "usage"):
            # Codex may emit usage totals here (depends on version). Collect if present.
            usage = event.get("usage") or event.get("token_usage") or {}
            for k in ("input_tokens", "output_tokens"):
                if k in usage:
                    token_usage[k] = token_usage.get(k, 0) + int(usage[k])
            if "total_cost_usd" in event:
                cost_usd = float(event.get("total_cost_usd", 0.0) or 0.0)

    # Derive files_read / files_written from bash commands (best-effort heuristic).
    # Codex's shell tool doesn't split Read/Write/Edit like Claude Code; we parse
    # common bash verbs. Over-count is fine — downstream only uses it for audit.
    files_read: dict[str, dict[str, int]] = {}
    files_written: dict[str, dict[str, int]] = {}
    _cat_re = re.compile(r"\b(cat|less|head|tail)\s+([^\s|;&]+)")
    _grep_re = re.compile(r"\bgrep\s+(?:-[a-zA-Z0-9]+\s+)*[^\s]+\s+([^\s|;&]+)")
    _write_re = re.compile(r"\b(?:tee|echo\s+[^>]+>{1,2})\s*([^\s|;&]+)")
    for tc in tool_calls:
        cmd_str = tc.input.get("command", "") if isinstance(tc.input, dict) else ""
        if not isinstance(cmd_str, str):
            continue
        for m in _cat_re.finditer(cmd_str):
            path = _make_relative(m.group(2), cwd)
            files_read[path] = {"reads": files_read.get(path, {}).get("reads", 0) + 1,
                                "lines": 0}
        for m in _grep_re.finditer(cmd_str):
            path = _make_relative(m.group(1), cwd)
            files_read[path] = {"reads": files_read.get(path, {}).get("reads", 0) + 1,
                                "lines": 0}
        for m in _write_re.finditer(cmd_str):
            path = _make_relative(m.group(1), cwd)
            files_written[path] = {"lines_written": 0}

    return SessionResult(
        prompt=prompt,
        text="\n".join(text_parts),
        tool_calls=tool_calls,
        files_read=files_read,
        files_written=files_written,
        token_usage=token_usage,
        duration_seconds=duration,
        model=model,
        session_id=session_id,
        exit_code=exit_code,
        cost_usd=cost_usd,
        raw_events=events,
    )


# ── Persistent log (optional) ────────────────────────────────────

def _persist_log(result: SessionResult, log_dir: Path) -> None:
    """Dump raw JSONL + summary to disk for post-hoc inspection."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = _slugify(result.name or result.prompt[:80])
    base = log_dir / f"{ts}_{slug}"

    (base.with_suffix(".prompt.txt")).write_text(result.prompt, encoding="utf-8")
    (base.with_suffix(".stdout.jsonl")).write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in result.raw_events),
        encoding="utf-8",
    )
    summary = {
        "name": result.name,
        "model": result.model,
        "session_id": result.session_id,
        "duration_s": result.duration_seconds,
        "exit_code": result.exit_code,
        "cost_usd": result.cost_usd,
        "text_len": len(result.text),
        "tool_calls_n": len(result.tool_calls),
        "files_read": list(result.files_read),
        "files_written": list(result.files_written),
        "cmd": result.command,
        "cwd": result.cwd,
        "stderr_tail": (result.stderr or "")[-500:],
    }
    (base.with_suffix(".summary.json")).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    result.log_dir = str(base)
