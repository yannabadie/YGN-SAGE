"""Tests for Z3 formal verification safety gate."""
import sys
import types
import pytest

# Mock sage_core (Rust extension) so we can import sage.sandbox without building it
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

z3_solver = pytest.importorskip("z3", reason="z3-solver not installed")

from sage.sandbox.z3_validator import Z3Validator, ValidationResult


@pytest.fixture
def validator():
    return Z3Validator()


class TestMemorySafety:
    def test_valid_access(self, validator):
        assert validator.prove_memory_safety(5, 100) is True

    def test_at_boundary(self, validator):
        # addr=99 is within [0, 100)
        assert validator.prove_memory_safety(99, 100) is True

    def test_out_of_bounds(self, validator):
        assert validator.prove_memory_safety(100, 100) is False

    def test_negative_address(self, validator):
        assert validator.prove_memory_safety(-1, 100) is False

    def test_zero_address(self, validator):
        assert validator.prove_memory_safety(0, 100) is True


class TestLoopBound:
    def test_symbolic_unbounded(self, validator):
        # A free symbolic variable can exceed any cap
        assert validator.check_loop_bound("n", 1_000_000) is False

    def test_different_variable_names(self, validator):
        assert validator.check_loop_bound("iter_count", 100) is False


class TestVerifyArrayBounds:
    def test_all_safe(self, validator):
        result = validator.verify_array_bounds([(0, 10), (5, 10), (9, 10)])
        assert result.safe is True
        assert result.violations == []
        assert result.proof_time_ms >= 0.0

    def test_single_violation(self, validator):
        result = validator.verify_array_bounds([(0, 10), (10, 10), (5, 10)])
        assert result.safe is False
        assert len(result.violations) == 1
        assert "index 10" in result.violations[0]

    def test_empty_accesses(self, validator):
        result = validator.verify_array_bounds([])
        assert result.safe is True


class TestValidateMutation:
    def test_bounds_safe(self, validator):
        result = validator.validate_mutation(["bounds(0, 100)", "bounds(50, 100)"])
        assert result.safe is True
        assert result.violations == []

    def test_bounds_violation(self, validator):
        result = validator.validate_mutation(["bounds(5, 100)", "bounds(200, 100)"])
        assert result.safe is False
        assert len(result.violations) == 1
        assert "200" in result.violations[0]

    def test_loop_constraint(self, validator):
        result = validator.validate_mutation(["loop(n, 1000000)"])
        assert result.safe is False
        assert len(result.violations) == 1

    def test_unknown_constraint(self, validator):
        result = validator.validate_mutation(["invalid_constraint"])
        assert result.safe is False
        assert "Unknown constraint type" in result.violations[0]

    def test_mixed_constraints(self, validator):
        result = validator.validate_mutation([
            "bounds(5, 100)",
            "bounds(200, 100)",
            "loop(n, 1000000)",
        ])
        assert result.safe is False
        assert len(result.violations) == 2

    def test_unparseable_bounds(self, validator):
        result = validator.validate_mutation(["bounds(abc, 100)"])
        assert result.safe is False
        assert "Unparseable" in result.violations[0]
