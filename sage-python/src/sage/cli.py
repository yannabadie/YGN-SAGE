"""Root CLI for YGN-SAGE."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence


_HELP = """YGN-SAGE root CLI

Usage:
  sage serve [serve-options]
  sage bench [bench-options]
  sage chat

Notes:
  - `sage serve` runs MCP and/or A2A protocol surfaces.
  - `sage bench` runs benchmark and evaluation flows.
  - `sage chat` is reserved for the future pi-mono-derived chat interface.
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the installed ``sage`` command to concrete subcommands."""
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in {"-h", "--help", "help"}:
        print(_HELP)
        return 0

    if args[0] == "serve":
        from sage.protocols import serve as serve_mod

        _dispatch_with_argv("sage serve", args[1:], serve_mod.main)
        return 0

    if args[0] == "bench":
        from sage.bench import __main__ as bench_main

        _dispatch_with_argv("sage bench", args[1:], bench_main.main)
        return 0

    if args[0] == "chat":
        print(
            "`sage chat` is reserved for the future pi-mono-derived chat interface.",
            file=sys.stderr,
        )
        return 2

    # Backward compatibility for the older `sage --mcp ...` console usage.
    if args[0].startswith("-"):
        from sage.protocols import serve as serve_mod

        _dispatch_with_argv("sage serve", args, serve_mod.main)
        return 0

    print(f"Unknown subcommand: {args[0]}", file=sys.stderr)
    print("Run `sage --help` for usage.", file=sys.stderr)
    return 2


def _dispatch_with_argv(prog: str, args: Sequence[str], target: Callable[[], None]) -> None:
    old_argv = sys.argv
    try:
        sys.argv = [prog, *args]
        target()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
