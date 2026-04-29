"""Centralized SAGE_ORACLE environment predicate.

cgpro 2026-04-29 cycle-7 default-on flip approval: rather than hand-edit
eight scattered ``os.environ.get("SAGE_ORACLE") == "1"`` predicates, route
every call through this single helper. Avoids the classic partial-flip
bug where one site gets inverted, another stays on the old semantics, and
the system ends up with OracleStack on but evidence producers off (or
vice-versa).

Default semantics (post cycle-7 flip):

- ``SAGE_ORACLE`` unset            ⇒ ON  (oracle path)
- ``SAGE_ORACLE=1``                ⇒ ON  (explicit)
- ``SAGE_ORACLE=true|on|yes``      ⇒ ON  (case-insensitive)
- ``SAGE_ORACLE=0``                ⇒ OFF (kill-switch)
- ``SAGE_ORACLE=false|off|no``     ⇒ OFF (case-insensitive)
- any other value                  ⇒ ON  (treat as enabled by default)

The kill-switch (``=0`` and friends) is the operator escape hatch — same
discipline as ``SAGE_UNSAFE_RAW_EXEC=1`` (gates an unsafe path) but
inverted: here we gate the SAFE/post-flip default OFF.

Pre-flip code used the predicate ``os.environ.get("SAGE_ORACLE") == "1"``;
that predicate's intent is preserved when callers move to ``oracle_enabled()``.
"""
from __future__ import annotations

import os

_FALSE_VALUES: frozenset[str] = frozenset({"0", "false", "off", "no"})


def oracle_enabled() -> bool:
    """True iff the OracleStack training-gate path should be active.

    Reads ``SAGE_ORACLE`` env var; default-on semantic (unset = enabled).
    Returns False only when the env var is explicitly set to a recognized
    false value (kill-switch). Whitespace is stripped, comparison is
    case-insensitive.
    """
    raw = os.environ.get("SAGE_ORACLE")
    if raw is None:
        return True
    return raw.strip().lower() not in _FALSE_VALUES
