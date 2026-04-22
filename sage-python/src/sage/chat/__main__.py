"""`python -m sage.chat` — interactive REPL entry point.

Usage:

    python -m sage.chat                  # new session
    python -m sage.chat --resume <id>    # continue an existing session

Per-turn loop:
    1. Read a line from stdin. Strip it; skip empty lines.
    2. `/shell` → toggle SAGE_CHAT_ALLOW_BASH for the session.
    3. `/replay` → print every event from the session's jsonl.
    4. `/exit`, `/quit`, EOF → clean shutdown, final system event logged.
    5. Anything else → `normalize_chat(line)` → `AgentSystem.run(task_input)`
       → print the assistant response, persist user + assistant events.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

from sage.chat.session import ChatSession, SESSIONS_DIR
from sage.input import normalize_chat

_PROMPT = "you: "
_ASSISTANT_PREFIX = "sage: "


def _print_banner(session: ChatSession, resumed: bool) -> None:
    mode = "resumed" if resumed else "new"
    print(f"sage.chat {mode} session {session.session_id}")
    print(f"  transcript: {session.path}")
    print(f"  bash tools: {'ALLOWED' if session.bash_allowed else 'OFF (chat defaults)'}")
    print("  commands: /shell  /replay  /exit")
    print("-" * 60)


def _looks_like_bash_opt_in() -> bool:
    val = (os.environ.get("SAGE_CHAT_ALLOW_BASH") or "").strip().lower()
    return val in {"1", "true", "yes", "on"}


async def _run_one_turn(agent_system, session: ChatSession, user_input: str) -> None:
    """Send one user turn through the agent and record both sides."""
    session.log_user(user_input)
    task_input = normalize_chat(user_input)
    t0 = time.perf_counter()
    try:
        response = await agent_system.run(task_input)
    except KeyboardInterrupt:
        session.log_system("turn_interrupted")
        print("^C — turn cancelled, session preserved", file=sys.stderr)
        return
    except Exception as exc:  # noqa: BLE001 — REPL should not die on a bad turn
        session.log_system("turn_error", error=f"{type(exc).__name__}: {exc}")
        print(f"{_ASSISTANT_PREFIX}[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        return
    latency_ms = int((time.perf_counter() - t0) * 1000)
    session.log_assistant(response, latency_ms=latency_ms)
    print(f"{_ASSISTANT_PREFIX}{response}")


async def _main(args: argparse.Namespace) -> int:
    # Boot is expensive; do it once, AFTER argparse so `--resume` + bad id
    # doesn't burn a boot.
    if args.resume:
        try:
            session = ChatSession.resume(args.resume)
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            print(f"  sessions live under: {SESSIONS_DIR}", file=sys.stderr)
            return 2
        resumed = True
    else:
        session = ChatSession.start(bash_allowed=_looks_like_bash_opt_in())
        resumed = False

    _print_banner(session, resumed)

    # Import booter lazily so --resume with a bad id can fail without a
    # 10-second provider-discovery phase.
    from sage.boot import boot_agent_system

    print("booting sage...", file=sys.stderr)
    agent_system = boot_agent_system()
    print("ready.", file=sys.stderr)

    try:
        while True:
            try:
                line = input(_PROMPT).strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\n(use /exit to end session cleanly)", file=sys.stderr)
                continue

            if not line:
                continue

            if line in {"/exit", "/quit"}:
                break

            if line == "/shell":
                new_state = session.toggle_bash()
                print(
                    f"bash tools: {'ALLOWED' if new_state else 'OFF (chat defaults)'}",
                    file=sys.stderr,
                )
                continue

            if line == "/replay":
                for event in session.replay():
                    kind = event.get("kind", "?")
                    ts = event.get("ts", "?")
                    text = event.get("text", event.get("event", ""))
                    preview = text if len(str(text)) < 200 else str(text)[:200] + "..."
                    print(f"  [{ts}] {kind}: {preview}", file=sys.stderr)
                continue

            await _run_one_turn(agent_system, session, line)
    finally:
        session.log_system("session_end")
        print(
            f"\nsession {session.session_id} saved to {session.path}",
            file=sys.stderr,
        )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sage.chat",
        description="Interactive chat REPL for YGN-SAGE.",
    )
    parser.add_argument(
        "--resume",
        metavar="SESSION_ID",
        help="Resume an existing session (ULID from ~/.sage/chat_sessions/).",
    )
    args = parser.parse_args()
    return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())
