"""Meta-Harness: End-to-end optimization of SAGE's harness code.

Applies Meta-Harness (Lee et al., arXiv 2603.28052) to YGN-SAGE:
searches over harness code via filesystem-based diagnostic history.

Usage with Claude Code:
    python -m sage.meta_harness init
    python -m sage.meta_harness propose
    python -m sage.meta_harness evaluate <id>
    python -m sage.meta_harness status
    python -m sage.meta_harness apply
"""

from sage.meta_harness.config import HarnessConfig
from sage.meta_harness.search_loop import MetaHarnessLoop

__all__ = ["HarnessConfig", "MetaHarnessLoop"]
