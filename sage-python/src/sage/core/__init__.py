"""Stable Python wrappers for the Rust sage_core boundary (Phase 8).

Each sub-module exposes a thin Python façade over the corresponding
Rust PyO3 class.  Direct ``import sage_core`` calls outside of this
package and ``sage.ops.*`` are discouraged per AUDITRUST.md Phase 8.
"""

from sage.core.routing import RustSystemRouter
from sage.core.topology import RustTopologyEngine
from sage.core.sandbox import RustToolExecutor
from sage.core.memory import RustMultiViewMMU

__all__ = [
    "RustSystemRouter",
    "RustTopologyEngine",
    "RustToolExecutor",
    "RustMultiViewMMU",
]
