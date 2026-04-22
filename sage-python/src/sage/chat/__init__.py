"""`sage.chat` — interactive REPL for YGN-SAGE.

Universal input adapter C5 (spec 2026-04-21). Thin read-eval loop that
converts each user turn into a `TaskInput` via `normalize_chat`, runs
it through `AgentSystem.run()`, and streams the turn to an
append-only jsonl on disk (`~/.sage/chat_sessions/<ulid>.jsonl`) so
sessions survive across restarts and can be replayed via `--resume`.

Per spec Q2 (jsonl append-only from day 1) and Q1 (bash opt-in via
`SAGE_CHAT_ALLOW_BASH` or the `/shell` REPL command).

Run it with:

    python -m sage.chat
    python -m sage.chat --resume 01K2BQWTDN...   # continue a prior session

Not wired to `sage_core` for anything — just a frontend around the
existing AgentSystem.
"""
from sage.chat.session import ChatSession, SESSIONS_DIR

__all__ = ["ChatSession", "SESSIONS_DIR"]
