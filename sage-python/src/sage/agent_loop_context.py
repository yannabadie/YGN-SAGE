"""LoopContext: shared state passed between perceive/think/act/learn phases.

Re-exports LoopContext from sage.phases for convenience. The canonical
definition lives in sage/phases/__init__.py.
"""
from __future__ import annotations

from sage.phases import LoopContext

__all__ = ["LoopContext"]
