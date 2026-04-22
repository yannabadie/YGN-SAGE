"""Unit tests for sage.chat.session — jsonl transcript persistence.

C5 (2026-04-22) of the universal input adapter spec. These tests do
NOT exercise the REPL loop (that requires a live AgentSystem). They
cover the append-only jsonl persistence + resume semantics + /shell
toggle that the REPL depends on.
"""
from __future__ import annotations

import json
import os

import pytest

from sage.chat.session import ChatSession, _iter_events


@pytest.fixture(autouse=True)
def _isolate_sessions_dir(monkeypatch, tmp_path):
    """Redirect SESSIONS_DIR into tmp so tests don't pollute ~/.sage."""
    import sage.chat.session as session_mod

    sandbox = tmp_path / "chat_sessions"
    monkeypatch.setattr(session_mod, "SESSIONS_DIR", sandbox)
    monkeypatch.delenv("SAGE_CHAT_ALLOW_BASH", raising=False)
    yield sandbox


# ---------------------------------------------------------------------------
# Construction + session_start event
# ---------------------------------------------------------------------------


def test_start_creates_sessions_dir_and_file(_isolate_sessions_dir):
    s = ChatSession.start()
    assert _isolate_sessions_dir.exists()
    assert s.path.exists()
    assert s.path.name == f"{s.session_id}.jsonl"


def test_start_writes_session_start_event():
    s = ChatSession.start()
    events = s.replay()
    assert len(events) == 1
    assert events[0]["kind"] == "session_start"
    assert events[0]["session_id"] == s.session_id
    assert "ts" in events[0]
    assert events[0]["bash_allowed"] is False


def test_start_captures_bash_allowed_flag():
    s = ChatSession.start(bash_allowed=True)
    assert s.bash_allowed is True
    events = s.replay()
    assert events[0]["bash_allowed"] is True


def test_session_id_is_a_ulid_string():
    """ULIDs are 26 chars, uppercase Base32. If the session_id factory
    swaps to something else silently (e.g. uuid4) the format check
    catches it."""
    s = ChatSession.start()
    assert isinstance(s.session_id, str)
    assert len(s.session_id) == 26
    # ULID character set is Crockford Base32: 0-9 A-Z minus I L O U
    assert all(c in "0123456789ABCDEFGHJKMNPQRSTVWXYZ" for c in s.session_id)


# ---------------------------------------------------------------------------
# Per-turn logging
# ---------------------------------------------------------------------------


def test_log_user_appends_event():
    s = ChatSession.start()
    s.log_user("hello world")
    events = s.replay()
    user_events = [e for e in events if e["kind"] == "user"]
    assert len(user_events) == 1
    assert user_events[0]["text"] == "hello world"


def test_log_assistant_records_latency():
    s = ChatSession.start()
    s.log_assistant("here is the answer", latency_ms=1234)
    events = [e for e in s.replay() if e["kind"] == "assistant"]
    assert events[0]["text"] == "here is the answer"
    assert events[0]["latency_ms"] == 1234


def test_log_assistant_without_latency_omits_field():
    s = ChatSession.start()
    s.log_assistant("quick")
    events = [e for e in s.replay() if e["kind"] == "assistant"]
    assert "latency_ms" not in events[0]


def test_log_system_carries_arbitrary_fields():
    s = ChatSession.start()
    s.log_system("custom_event", foo="bar", count=42)
    events = [e for e in s.replay() if e["kind"] == "system"]
    # [0] is the session_start (kind=session_start, not system)
    # but session_start and system are different kinds. Filter explicitly.
    custom = [e for e in events if e.get("event") == "custom_event"]
    assert len(custom) == 1
    assert custom[0]["foo"] == "bar"
    assert custom[0]["count"] == 42


def test_unicode_text_roundtrips_verbatim():
    """Jsonl serialization must preserve unicode (requirements audit
    flagged this as a regression risk when the default json.dumps
    ensure_ascii=True was observed elsewhere in the codebase)."""
    s = ChatSession.start()
    weird = "你好 — naïve — 🦀 rust"
    s.log_user(weird)
    events = [e for e in s.replay() if e["kind"] == "user"]
    assert events[0]["text"] == weird


# ---------------------------------------------------------------------------
# Append-only semantics
# ---------------------------------------------------------------------------


def test_each_turn_appends_one_jsonl_line():
    s = ChatSession.start()
    s.log_user("q1")
    s.log_assistant("a1")
    s.log_user("q2")
    s.log_assistant("a2")
    raw = s.path.read_text(encoding="utf-8")
    # session_start + 4 turns = 5 lines
    lines = [ln for ln in raw.split("\n") if ln]
    assert len(lines) == 5
    # Each line is valid JSON
    for ln in lines:
        json.loads(ln)


def test_interrupted_partial_file_still_replays_complete_records():
    """A crashed REPL may leave a half-written final line. replay()
    skips malformed lines rather than crashing."""
    s = ChatSession.start()
    s.log_user("fine")
    # Simulate partial write: append an incomplete JSON line
    with s.path.open("a", encoding="utf-8") as f:
        f.write('{"kind": "assistant", "ts"')  # no newline, no close
    events = s.replay()
    # session_start + fine-user-event survive; the half line is skipped
    assert any(e.get("kind") == "user" and e.get("text") == "fine" for e in events)
    assert len(events) >= 2


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def test_resume_reopens_session_without_rewriting_start():
    original = ChatSession.start()
    original.log_user("hello")
    original.log_assistant("hi back")

    resumed = ChatSession.resume(original.session_id)

    assert resumed.session_id == original.session_id
    assert resumed.path == original.path

    events = resumed.replay()
    kinds = [e["kind"] for e in events]
    # Exactly one session_start despite the resume
    assert kinds.count("session_start") == 1
    # The resume itself appended a "system" event
    assert any(e.get("kind") == "system" and e.get("event") == "resume" for e in events)


def test_resume_preserves_bash_allowed_state():
    """If a prior session toggled /shell ON, resuming must re-arm the
    env var + the flag — otherwise users lose consent context on a
    session restart."""
    original = ChatSession.start(bash_allowed=False)
    original.toggle_bash()  # flip ON
    assert original.bash_allowed is True

    resumed = ChatSession.resume(original.session_id)
    assert resumed.bash_allowed is True


def test_resume_unknown_id_raises():
    with pytest.raises(FileNotFoundError):
        ChatSession.resume("01ZZZZZZZZZZZZZZZZZZZZZZZZ")


# ---------------------------------------------------------------------------
# /shell toggle — mirrors env var + flag + event log
# ---------------------------------------------------------------------------


def test_toggle_bash_flips_flag_and_env_var():
    s = ChatSession.start(bash_allowed=False)
    assert os.environ.get("SAGE_CHAT_ALLOW_BASH") != "1"

    new_state = s.toggle_bash()
    assert new_state is True
    assert s.bash_allowed is True
    assert os.environ.get("SAGE_CHAT_ALLOW_BASH") == "1"

    new_state = s.toggle_bash()
    assert new_state is False
    assert s.bash_allowed is False
    assert os.environ.get("SAGE_CHAT_ALLOW_BASH") == "0"


def test_toggle_bash_logs_system_event():
    s = ChatSession.start(bash_allowed=False)
    s.toggle_bash()
    events = [e for e in s.replay() if e.get("event") == "shell_toggled"]
    assert len(events) == 1
    assert events[0]["bash_allowed"] is True


def test_toggle_bash_affects_normalize_chat():
    """Integration: /shell ON → normalize_chat returns tools_filter=None
    (all tools). This is the whole point of the toggle — users who opt
    into bash should see it on the very next turn's TaskInput."""
    from sage.input import normalize_chat

    s = ChatSession.start(bash_allowed=False)
    ti_before = normalize_chat("ls")
    assert ti_before.tools_filter is not None  # filtered

    s.toggle_bash()
    ti_after = normalize_chat("ls")
    assert ti_after.tools_filter is None  # all tools


# ---------------------------------------------------------------------------
# _iter_events — raw helper
# ---------------------------------------------------------------------------


def test_iter_events_skips_blank_lines(tmp_path):
    p = tmp_path / "fake.jsonl"
    p.write_text(
        '{"kind": "user", "text": "a"}\n'
        "\n"
        "   \n"
        '{"kind": "assistant", "text": "b"}\n',
        encoding="utf-8",
    )
    events = list(_iter_events(p))
    assert [e["kind"] for e in events] == ["user", "assistant"]


# ---------------------------------------------------------------------------
# Defensive: transcript file path is absolute + under SESSIONS_DIR
# ---------------------------------------------------------------------------


def test_session_path_is_absolute_under_sessions_dir(_isolate_sessions_dir):
    s = ChatSession.start()
    assert s.path.is_absolute()
    assert _isolate_sessions_dir in s.path.parents
