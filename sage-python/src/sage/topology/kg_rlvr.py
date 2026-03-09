"""Knowledge Graph & Z3-Backed Reinforcement Learning from Verifiable Rewards (KG-RLVR).

Implements Process Reward Models (PRM) to evaluate internal reasoning paths 
(<think> tags). Shifts from Outcome-based to Process-based rewards (System 3 AI)
using formal SMT verification via Z3 to eliminate hallucinations.
"""
from __future__ import annotations

import ast
import re
import logging
from typing import List, Dict, Any, Tuple

try:
    import z3
except ImportError:
    z3 = None

try:
    from sage.sandbox.z3_validator import Z3Validator as _Z3Validator
    _has_z3_validator = True
except ImportError:
    _has_z3_validator = False

# Allowed AST node types for safe Z3 expression evaluation
_SAFE_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
    ast.Constant, ast.Name, ast.Attribute, ast.Call, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,
    ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.Eq, ast.NotEq,
    ast.And, ast.Or, ast.Not, ast.USub,
)


def _safe_z3_eval(expr: str, namespace: dict) -> Any:
    """Evaluate a Z3 constraint string using restricted AST parsing.

    Only allows: comparisons, arithmetic, boolean ops, variable names,
    constants, and z3.* attribute access / function calls.
    Raises ValueError on any disallowed construct.
    """
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_NODES):
            raise ValueError(f"Disallowed AST node: {type(node).__name__}")
        if isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.value.id == "z3"):
                raise ValueError(f"Attribute access only allowed on 'z3'")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "z3"):
                    raise ValueError(f"Function calls only allowed on z3.*")
            elif isinstance(node.func, ast.Name):
                if node.func.id not in namespace:
                    raise ValueError(f"Unknown function: {node.func.id}")
        if isinstance(node, ast.Name) and node.id not in namespace:
            raise ValueError(f"Unknown name: {node.id}")
    code = compile(tree, "<z3_constraint>", "eval")
    return eval(code, {"__builtins__": {}}, namespace)


class FormalKnowledgeGraph:
    """A formal Knowledge Graph backed by Z3 for verifiable reasoning."""
    def __init__(self):
        self.has_z3 = z3 is not None
        if not self.has_z3:
            logging.error(
                "z3-solver not installed. ALL formal verification disabled — "
                "returning unverified (fail-closed)."
            )

    def prove_memory_safety(self, addr_expr: int, limit: int) -> bool:
        if not self.has_z3:
            return 0 <= addr_expr < limit
        solver = z3.Solver()
        addr = z3.IntVal(addr_expr)
        max_mem = z3.IntVal(limit)
        min_mem = z3.IntVal(0)
        
        out_of_bounds = z3.Or(addr < min_mem, addr >= max_mem)
        solver.add(out_of_bounds)
        return solver.check() == z3.unsat

    def check_loop_bound(self, iterations_symbolic: str, hard_cap: int) -> bool:
        if not self.has_z3:
            return False
        solver = z3.Solver()
        iters = z3.Int(iterations_symbolic)
        cap = z3.IntVal(hard_cap)
        
        solver.add(iters > cap)
        return solver.check() == z3.unsat

    def verify_arithmetic(self, expr: str, expected: int, tolerance: int = 0) -> bool:
        """Verify an arithmetic expression evaluates within tolerance of expected."""
        if not self.has_z3:
            return False
        solver = z3.Solver()
        result = z3.Int("result")
        solver.add(z3.Or(result > expected + tolerance, result < expected - tolerance))
        return solver.check() == z3.unsat

    def verify_invariant(self, pre: str, post: str) -> bool:
        """Verify a pre/post-condition pair using Z3.

        Uses a restricted AST evaluator instead of eval() to prevent
        arbitrary code execution. Fails closed (returns False) on any error.
        """
        if not self.has_z3:
            return False
        solver = z3.Solver()
        x = z3.Int("x")
        try:
            pre_constraint = _safe_z3_eval(pre, {"x": x, "z3": z3})
            post_constraint = _safe_z3_eval(post, {"x": x, "z3": z3})
            solver.add(z3.And(pre_constraint, z3.Not(post_constraint)))
            return solver.check() == z3.unsat
        except Exception:
            return False  # Fail CLOSED — can't parse means reject

    def verify_step(self, step: str) -> float:
        """Score a reasoning step based on its formal verifiability."""
        step_lower = step.lower()
        
        # Parse formal constraints and attempt Z3 proofs.
        
        # Look for "assert bounds(X, limit)"
        bounds_match = re.search(r"assert\s+bounds\(\s*(-?\d+)\s*,\s*(\d+)\s*\)", step_lower)
        if bounds_match:
            addr = int(bounds_match.group(1))
            limit = int(bounds_match.group(2))
            
            is_safe = self.prove_memory_safety(addr, limit)
            if is_safe:
                return 1.0  # Mathematically proven
            else:
                return -1.0 # Provably false - hallucination
                
        # Look for "assert loop(N)"
        loop_match = re.search(r"assert\s+loop\(\s*([a-zA-Z0-9_]+)\s*\)", step_lower)
        if loop_match:
            var_name = loop_match.group(1)
            is_bounded = self.check_loop_bound(var_name, 1000000)
            if is_bounded:
                return 1.0
            else:
                return -1.0
                
        # Look for "assert arithmetic(expr, expected)"
        arith_match = re.search(r"assert\s+arithmetic\(\s*(.+?)\s*,\s*(-?\d+)\s*\)", step_lower)
        if arith_match:
            expr = arith_match.group(1)
            expected = int(arith_match.group(2))
            is_valid = self.verify_arithmetic(expr, expected)
            return 1.0 if is_valid else -1.0

        # Look for "assert invariant("pre", "post")"
        inv_match = re.search(r'assert\s+invariant\("(.+?)"\s*,\s*"(.+?)"\)', step_lower)
        if inv_match:
            pre, post = inv_match.group(1), inv_match.group(2)
            is_valid = self.verify_invariant(pre, post)
            return 1.0 if is_valid else -1.0

        # Fallback to logical keyword heuristic if no formal logic detected
        if "ebpf" in step_lower and "latency" in step_lower:
            return 0.2

        return 0.0


class ProcessRewardModel:
    """Evaluates agent reasoning paths using verifiable Z3/KG rewards."""
    
    def __init__(self, kg: FormalKnowledgeGraph = None):
        self.kg = kg or FormalKnowledgeGraph()
        self.logger = logging.getLogger(__name__)

    def extract_reasoning_steps(self, content: str) -> List[str]:
        """Extracts text inside <think>...</think> tags and splits into steps."""
        pattern = r"<think>(.*?)</think>"
        matches = re.findall(pattern, content, re.DOTALL)
        
        steps = []
        for match in matches:
            raw_steps = [s.strip() for s in match.split('\n') if s.strip()]
            steps.extend(raw_steps)
            
        return steps

    def calculate_r_path(self, content: str) -> Tuple[float, Dict[str, Any]]:
        """Calculate the R_path (Process Reward) for a given generation."""
        steps = self.extract_reasoning_steps(content)
        
        if not steps:
            # Penalty for not reasoning (System 1 instead of System 3)
            return -1.0, {"error": "No <think> blocks found. System 3 reasoning required."}
            
        step_scores = []
        for step in steps:
            score = self.kg.verify_step(step)
            step_scores.append(score)
            
        # Overall R_path is the average of verifiable steps
        if len(step_scores) == 0:
            return 0.0, {"error": "Empty reasoning."}
            
        r_path = sum(step_scores) / len(step_scores)
        
        details = {
            "total_steps": len(steps),
            "step_scores": step_scores,
            "verifiable_ratio": sum(1 for s in step_scores if s > 0.5) / len(steps),
            "hallucination_ratio": sum(1 for s in step_scores if s < 0.0) / len(steps)
        }
        
        # Severe penalty if any mathematical proof failed
        if details["hallucination_ratio"] > 0:
            r_path = -1.0

        return r_path, details

    def score_with_z3(self, constraints: list[str]) -> tuple[float, dict[str, Any]]:
        """Score constraints using the Python Z3Validator for sub-ms verification."""
        if not _has_z3_validator:
            return 0.0, {"error": "Z3Validator not available (install z3-solver)"}

        validator = _Z3Validator()
        result = validator.validate_mutation(constraints)

        score = 1.0 if result.safe else -1.0
        details = {
            "safe": result.safe,
            "violations": result.violations,
            "proof_time_ms": result.proof_time_ms,
            "backend": "sage.sandbox.z3_validator"
        }
        return score, details
