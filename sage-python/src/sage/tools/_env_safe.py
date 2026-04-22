"""Shared subprocess-environment scrubbing.

P0.2 (2026-04-22 audit remediation) introduced a bash-specific env
allowlist inside ``sage.boot``. P0.1 needs the same discipline for
every typed-tool subprocess (ripgrep, pytest, git diff, git apply).
Extracted here so both modules share one source of truth — any new
typed tool gets the allowlist for free.

Contract: only allow process-wide env vars that are safely needed
for subprocesses to launch (PATH, HOME, temp dirs, locale, Windows
system paths). Everything else — especially `*_API_KEY`,
`CONTEXT7`, `SAGE_EXOCORTEX_STORE`, any provider secret — is
dropped. If you need to pass a variable to a subprocess on purpose,
extend the allowlist AND add a regression test.
"""
from __future__ import annotations

import os

# Keep this sorted + commented; every entry should have a reason.
BASH_ENV_ALLOWLIST: frozenset[str] = frozenset({
    # Core POSIX
    "PATH",
    "HOME",
    "PWD",
    "USER",
    "USERNAME",
    "SHELL",
    # Temp directories
    "TEMP",
    "TMP",
    "TMPDIR",
    # Locale (affects `rg`, `pytest`, `git` output encoding)
    "LANG",
    "LC_ALL",
    # Windows / git-bash subprocess launch
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "PROGRAMDATA",
})


def safe_subprocess_env() -> dict[str, str]:
    """Return a scrubbed copy of ``os.environ``.

    Only keys in :data:`BASH_ENV_ALLOWLIST` pass through. API keys,
    Context7 token, ExoCortex store ids, SAGE internal flags —
    stripped.
    """
    return {
        key: val
        for key, val in os.environ.items()
        if key in BASH_ENV_ALLOWLIST
    }
