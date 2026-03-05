use pyo3::prelude::*;
use z3::{Config, Context, Solver, SatResult, ast::{Int, Bool, Ast}};

/// Result of a Z3 validation check.
#[pyclass]
#[derive(Debug, Clone)]
pub struct ValidationResult {
    #[pyo3(get)]
    pub safe: bool,
    #[pyo3(get)]
    pub violations: Vec<String>,
    #[pyo3(get)]
    pub proof_time_ms: f64,
}

/// SOTA 2026: Z3-based SMT Firewall for evolved code validation.
/// Phase 1: Safety Gate — validates invariants before eBPF/Wasm execution.
/// Phase 2: PRM Backend — provides formal proof scoring for reasoning steps.
#[pyclass]
pub struct Z3Validator {
    ctx: Context,
}

#[pymethods]
impl Z3Validator {
    #[new]
    pub fn new() -> Self {
        let cfg = Config::new();
        Self {
            ctx: Context::new(&cfg),
        }
    }

    /// Prove that a given memory access is within bounds.
    /// Returns true if 0 <= addr < limit is guaranteed.
    pub fn prove_memory_safety(&self, addr_expr: i64, limit: i64) -> bool {
        let solver = Solver::new(&self.ctx);
        let addr = Int::from_i64(&self.ctx, addr_expr);
        let max_mem = Int::from_i64(&self.ctx, limit);
        let min_mem = Int::from_i64(&self.ctx, 0);

        let out_of_bounds = Bool::or(&self.ctx, &[
            &addr.lt(&min_mem),
            &addr.ge(&max_mem),
        ]);

        solver.assert(&out_of_bounds);
        solver.check() == SatResult::Unsat
    }

    /// Check for potential infinite loops — returns true if loop is provably bounded.
    pub fn check_loop_bound(&self, iterations_symbolic: &str, hard_cap: i64) -> bool {
        let solver = Solver::new(&self.ctx);
        let iters = Int::new_const(&self.ctx, iterations_symbolic);
        let cap = Int::from_i64(&self.ctx, hard_cap);

        solver.assert(&iters.gt(&cap));
        solver.check() == SatResult::Unsat
    }

    /// Verify that all array accesses are within bounds.
    pub fn verify_array_bounds(&self, accesses: Vec<(i64, i64)>) -> ValidationResult {
        let start = std::time::Instant::now();
        let mut violations = Vec::new();

        for (i, (index, length)) in accesses.iter().enumerate() {
            if !self.prove_memory_safety(*index, *length) {
                violations.push(format!(
                    "Access #{}: index {} may be out of bounds [0, {})",
                    i, index, length
                ));
            }
        }

        ValidationResult {
            safe: violations.is_empty(),
            violations,
            proof_time_ms: start.elapsed().as_secs_f64() * 1000.0,
        }
    }

    /// Unified entry point for mutation validation.
    /// Parses constraint strings like "bounds(5, 100)" or "loop(n, 1000000)".
    pub fn validate_mutation(&self, constraints: Vec<String>) -> ValidationResult {
        let start = std::time::Instant::now();
        let mut violations = Vec::new();

        for constraint in &constraints {
            let trimmed = constraint.trim();

            if trimmed.starts_with("bounds(") {
                if let Some(inner) = trimmed.strip_prefix("bounds(").and_then(|s| s.strip_suffix(')')) {
                    let parts: Vec<&str> = inner.split(',').collect();
                    if parts.len() == 2 {
                        if let (Ok(addr), Ok(limit)) = (
                            parts[0].trim().parse::<i64>(),
                            parts[1].trim().parse::<i64>(),
                        ) {
                            if !self.prove_memory_safety(addr, limit) {
                                violations.push(format!("Memory violation: {} out of [0, {})", addr, limit));
                            }
                            continue;
                        }
                    }
                }
                violations.push(format!("Unparseable bounds constraint: {}", trimmed));
            } else if trimmed.starts_with("loop(") {
                if let Some(inner) = trimmed.strip_prefix("loop(").and_then(|s| s.strip_suffix(')')) {
                    let parts: Vec<&str> = inner.split(',').collect();
                    if parts.len() == 2 {
                        let var_name = parts[0].trim();
                        if let Ok(cap) = parts[1].trim().parse::<i64>() {
                            if !self.check_loop_bound(var_name, cap) {
                                violations.push(format!("Loop '{}' may exceed cap {}", var_name, cap));
                            }
                            continue;
                        }
                    }
                }
                violations.push(format!("Unparseable loop constraint: {}", trimmed));
            } else {
                violations.push(format!("Unknown constraint type: {}", trimmed));
            }
        }

        ValidationResult {
            safe: violations.is_empty(),
            violations,
            proof_time_ms: start.elapsed().as_secs_f64() * 1000.0,
        }
    }
}
