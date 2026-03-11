"""SMT Firewall for evolved code validation.

Phase 1: Safety Gate — validates invariants before execution.
Phase 2: PRM Backend — provides formal proof scoring for reasoning steps.

Prefers Rust OxiZ backend (sage_core.SmtVerifier) for zero-dependency, sub-ms
verification. Falls back to Python z3-solver if sage_core[smt] is not built.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

# Try Rust OxiZ backend first (pure Rust, no C++ deps)
try:
    from sage_core import SmtVerifier as _RustSmtVerifier
    from sage_core import SmtVerificationResult as _RustSmtResult
    _RUST_SMT_AVAILABLE = True
except ImportError:
    _RUST_SMT_AVAILABLE = False

# Fallback: Python z3-solver
try:
    import z3
    _Z3_AVAILABLE = True
except ImportError:
    _Z3_AVAILABLE = False


@dataclass
class ValidationResult:
    """Result of a Z3 validation check."""
    safe: bool
    violations: list[str] = field(default_factory=list)
    proof_time_ms: float = 0.0


class Z3Validator:
    """SMT solver for formal verification of evolved code.

    Provides memory safety proofs, loop bound checking, and
    constraint-based mutation validation.

    Backend priority: Rust OxiZ (sage_core) > Python z3-solver.
    """

    def __init__(self) -> None:
        self._rust: _RustSmtVerifier | None = None
        if _RUST_SMT_AVAILABLE:
            self._rust = _RustSmtVerifier()
        elif not _Z3_AVAILABLE:
            raise ImportError(
                "No SMT backend: install sage_core[smt] or z3-solver"
            )

    def prove_memory_safety(self, addr_expr: int, limit: int) -> bool:
        """Prove that a given memory access is within bounds.

        Returns True if 0 <= addr < limit is guaranteed.
        """
        if self._rust:
            return self._rust.prove_memory_safety(addr_expr, limit)
        s = z3.Solver()
        addr = z3.IntVal(addr_expr)
        max_mem = z3.IntVal(limit)
        min_mem = z3.IntVal(0)

        out_of_bounds = z3.Or(addr < min_mem, addr >= max_mem)
        s.add(out_of_bounds)
        return s.check() == z3.unsat

    def check_loop_bound(self, var_name: str, hard_cap: int) -> bool:
        """Check if a loop variable is provably bounded below hard_cap.

        For an *unconstrained* symbolic variable, correctly returns False.
        """
        if self._rust:
            return self._rust.check_loop_bound(var_name, hard_cap)
        s = z3.Solver()
        iters = z3.Int(var_name)
        cap = z3.IntVal(hard_cap)

        s.add(iters > cap)
        return s.check() == z3.unsat

    def verify_array_bounds(
        self, accesses: list[tuple[int, int]]
    ) -> ValidationResult:
        """Verify that all array accesses are within bounds."""
        if self._rust:
            r = self._rust.verify_array_bounds(accesses)
            return ValidationResult(safe=r.safe, violations=r.violations, proof_time_ms=r.proof_time_ms)
        start = time.perf_counter()
        violations: list[str] = []

        for i, (index, length) in enumerate(accesses):
            if not self.prove_memory_safety(index, length):
                violations.append(
                    f"Access #{i}: index {index} may be out of bounds [0, {length})"
                )

        return ValidationResult(
            safe=len(violations) == 0,
            violations=violations,
            proof_time_ms=(time.perf_counter() - start) * 1000.0,
        )

    def validate_mutation(
        self, constraints: list[str]
    ) -> ValidationResult:
        """Unified entry point for mutation validation.

        Parses constraint strings like "bounds(5, 100)" or "loop(n, 1000000)".
        """
        if self._rust:
            r = self._rust.validate_mutation(constraints)
            return ValidationResult(safe=r.safe, violations=r.violations, proof_time_ms=r.proof_time_ms)
        start = time.perf_counter()
        violations: list[str] = []

        for constraint in constraints:
            trimmed = constraint.strip()

            if trimmed.startswith("bounds("):
                inner = trimmed[len("bounds("):]
                if inner.endswith(")"):
                    inner = inner[:-1]
                    parts = inner.split(",")
                    if len(parts) == 2:
                        try:
                            addr = int(parts[0].strip())
                            limit = int(parts[1].strip())
                            if not self.prove_memory_safety(addr, limit):
                                violations.append(
                                    f"Memory violation: {addr} out of [0, {limit})"
                                )
                            continue
                        except ValueError:
                            pass
                violations.append(f"Unparseable bounds constraint: {trimmed}")

            elif trimmed.startswith("loop("):
                inner = trimmed[len("loop("):]
                if inner.endswith(")"):
                    inner = inner[:-1]
                    parts = inner.split(",")
                    if len(parts) == 2:
                        var_name = parts[0].strip()
                        try:
                            cap = int(parts[1].strip())
                            if not self.check_loop_bound(var_name, cap):
                                violations.append(
                                    f"Loop '{var_name}' may exceed cap {cap}"
                                )
                            continue
                        except ValueError:
                            pass
                violations.append(f"Unparseable loop constraint: {trimmed}")

            else:
                violations.append(f"Unknown constraint type: {trimmed}")

        return ValidationResult(
            safe=len(violations) == 0,
            violations=violations,
            proof_time_ms=(time.perf_counter() - start) * 1000.0,
        )
