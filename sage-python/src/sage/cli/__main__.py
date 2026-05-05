"""Entry point for ``python -m sage.cli ...``.

Restored when ``sage.cli`` was promoted from a single module
(``sage/cli.py``) to a package (``sage/cli/__init__.py``) in cycle-12
prelude (2026-05-05). Without this file, ``python -m sage.cli`` raises
``No module named sage.cli.__main__``.
"""
from sage.cli import main

raise SystemExit(main())
