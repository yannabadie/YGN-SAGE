from __future__ import annotations

import sys

import sage.bench.__main__ as bench_main
import sage.cli as cli
import sage.protocols.serve as serve_main


def test_root_cli_dispatches_serve(monkeypatch):
    calls: dict[str, object] = {}

    def _fake_main():
        calls["argv"] = sys.argv[:]

    monkeypatch.setattr(serve_main, "main", _fake_main)

    rc = cli.main(["serve", "--mcp", "--mcp-port", "8001"])

    assert rc == 0
    assert calls == {"argv": ["sage serve", "--mcp", "--mcp-port", "8001"]}


def test_root_cli_supports_legacy_protocol_flags(monkeypatch):
    calls: dict[str, object] = {}

    def _fake_main():
        calls["argv"] = sys.argv[:]

    monkeypatch.setattr(serve_main, "main", _fake_main)

    rc = cli.main(["--mcp", "--a2a"])

    assert rc == 0
    assert calls == {"argv": ["sage serve", "--mcp", "--a2a"]}


def test_root_cli_dispatches_bench(monkeypatch):
    calls: dict[str, object] = {}

    def _fake_main():
        calls["argv"] = sys.argv[:]

    monkeypatch.setattr(bench_main, "main", _fake_main)

    rc = cli.main(["bench", "--type", "routing_gt"])

    assert rc == 0
    assert calls == {"argv": ["sage bench", "--type", "routing_gt"]}


def test_root_cli_restores_argv_after_dispatch(monkeypatch):
    original_argv = sys.argv[:]

    def _fake_main():
        assert sys.argv == ["sage bench", "--type", "routing_gt"]

    monkeypatch.setattr(bench_main, "main", _fake_main)

    rc = cli.main(["bench", "--type", "routing_gt"])

    assert rc == 0
    assert sys.argv == original_argv


def test_root_cli_reserves_chat(capsys):
    rc = cli.main(["chat"])
    captured = capsys.readouterr()

    assert rc == 2
    assert "pi-mono-derived chat interface" in captured.err


def test_root_cli_help(capsys):
    rc = cli.main(["--help"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "sage serve" in captured.out
    assert "sage bench" in captured.out
    assert "sage chat" in captured.out
