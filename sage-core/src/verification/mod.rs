//! OxiZ-backed SMT verification for contracts and evolved code.
//!
//! Pure Rust replacement for Python `z3-solver` dependency.
//! Behind `smt` feature flag. Covers:
//! - Memory safety proofs (bounds checking)
//! - Loop bound verification
//! - Arithmetic verification
//! - Provider assignment (SAT with exactly-one constraint)
//!
//! All operations are QF_LIA (quantifier-free linear integer arithmetic).

use oxiz::{Solver, SolverResult, TermId, TermManager};
use pyo3::prelude::*;
use std::time::Instant;

/// Helper: create solver+tm, assert formula, check UNSAT.
fn is_unsat(build: impl FnOnce(&mut TermManager) -> TermId) -> bool {
    let mut solver = Solver::new();
    let mut tm = TermManager::new();
    let formula = build(&mut tm);
    solver.assert(formula, &mut tm);
    solver.check(&mut tm) == SolverResult::Unsat
}

/// Result of a formal verification check.
#[pyclass]
#[derive(Clone, Debug)]
pub struct SmtVerificationResult {
    #[pyo3(get)]
    pub safe: bool,
    #[pyo3(get)]
    pub violations: Vec<String>,
    #[pyo3(get)]
    pub proof_time_ms: f64,
}

#[pymethods]
impl SmtVerificationResult {
    fn __repr__(&self) -> String {
        format!(
            "SmtVerificationResult(safe={}, violations={}, proof_time_ms={:.3})",
            self.safe,
            self.violations.len(),
            self.proof_time_ms,
        )
    }
}

/// Pure-Rust SMT verifier powered by OxiZ.
///
/// Replaces Python z3-solver for formal verification:
/// - Memory bounds proofs
/// - Loop bound checking
/// - Arithmetic verification
/// - Provider assignment (constraint satisfaction)
#[pyclass]
pub struct SmtVerifier;

#[pymethods]
impl SmtVerifier {
    #[new]
    pub fn new() -> Self {
        Self
    }

    /// Prove that 0 <= addr < limit (memory bounds check).
    ///
    /// Returns True if addr is provably within bounds.
    pub fn prove_memory_safety(&self, addr: i64, limit: i64) -> bool {
        is_unsat(|tm| {
            let a = tm.mk_int(addr);
            let l = tm.mk_int(limit);
            let zero = tm.mk_int(0);
            let lt_zero = tm.mk_lt(a, zero);
            let ge_limit = tm.mk_ge(a, l);
            tm.mk_or(vec![lt_zero, ge_limit])
        })
    }

    /// Check if a symbolic loop variable is provably bounded below hard_cap.
    ///
    /// For an unconstrained variable, correctly returns False.
    pub fn check_loop_bound(&self, _var_name: &str, hard_cap: i64) -> bool {
        is_unsat(|tm| {
            let iters = tm.mk_var("iters", tm.sorts.int_sort);
            let cap = tm.mk_int(hard_cap);
            tm.mk_gt(iters, cap)
        })
    }

    /// Verify that actual value is within [expected-tol, expected+tol].
    pub fn verify_arithmetic(&self, actual: i64, expected: i64, tolerance: i64) -> bool {
        is_unsat(|tm| {
            let a = tm.mk_int(actual);
            let lo = tm.mk_int(expected - tolerance);
            let hi = tm.mk_int(expected + tolerance);
            let too_low = tm.mk_lt(a, lo);
            let too_high = tm.mk_gt(a, hi);
            tm.mk_or(vec![too_low, too_high])
        })
    }

    /// Verify array access bounds for a batch of (index, length) pairs.
    pub fn verify_array_bounds(&self, accesses: Vec<(i64, i64)>) -> SmtVerificationResult {
        let start = Instant::now();
        let mut violations = Vec::new();

        for (i, (index, length)) in accesses.iter().enumerate() {
            if !self.prove_memory_safety(*index, *length) {
                violations.push(format!(
                    "Access #{}: index {} may be out of bounds [0, {})",
                    i, index, length
                ));
            }
        }

        SmtVerificationResult {
            safe: violations.is_empty(),
            violations,
            proof_time_ms: start.elapsed().as_secs_f64() * 1000.0,
        }
    }

    /// Validate a batch of constraint strings.
    ///
    /// Supports: "bounds(addr, limit)", "loop(var, cap)"
    pub fn validate_mutation(&self, constraints: Vec<String>) -> SmtVerificationResult {
        let start = Instant::now();
        let mut violations = Vec::new();

        for constraint in &constraints {
            let trimmed = constraint.trim();

            if let Some(inner) = trimmed.strip_prefix("bounds(").and_then(|s| s.strip_suffix(')'))
            {
                let parts: Vec<&str> = inner.split(',').collect();
                if parts.len() == 2 {
                    if let (Ok(addr), Ok(limit)) = (
                        parts[0].trim().parse::<i64>(),
                        parts[1].trim().parse::<i64>(),
                    ) {
                        if !self.prove_memory_safety(addr, limit) {
                            violations.push(format!(
                                "Memory violation: {} out of [0, {})",
                                addr, limit
                            ));
                        }
                        continue;
                    }
                }
                violations.push(format!("Unparseable bounds constraint: {}", trimmed));
            } else if let Some(inner) =
                trimmed.strip_prefix("loop(").and_then(|s| s.strip_suffix(')'))
            {
                let parts: Vec<&str> = inner.split(',').collect();
                if parts.len() == 2 {
                    let var_name = parts[0].trim();
                    if let Ok(cap) = parts[1].trim().parse::<i64>() {
                        if !self.check_loop_bound(var_name, cap) {
                            violations
                                .push(format!("Loop '{}' may exceed cap {}", var_name, cap));
                        }
                        continue;
                    }
                }
                violations.push(format!("Unparseable loop constraint: {}", trimmed));
            } else {
                violations.push(format!("Unknown constraint type: {}", trimmed));
            }
        }

        SmtVerificationResult {
            safe: violations.is_empty(),
            violations,
            proof_time_ms: start.elapsed().as_secs_f64() * 1000.0,
        }
    }

    /// Verify provider assignment (SAT with exactly-one constraint).
    ///
    /// nodes: list of (node_id, required_capabilities)
    /// providers: list of (provider_name, provided_capabilities, exclusion_pairs)
    #[pyo3(signature = (nodes, providers))]
    pub fn verify_provider_assignment(
        &self,
        nodes: Vec<(String, Vec<String>)>,
        providers: Vec<(String, Vec<String>, Vec<(String, String)>)>,
    ) -> (bool, Vec<String>) {
        if providers.is_empty() {
            for (nid, caps) in &nodes {
                if !caps.is_empty() {
                    return (
                        false,
                        vec![format!(
                            "Node '{}' requires {:?} but no providers available",
                            nid, caps
                        )],
                    );
                }
            }
            return (true, vec![]);
        }

        let mut solver = Solver::new();
        let mut tm = TermManager::new();
        let mut all_pvars: Vec<(String, Vec<(String, TermId)>)> = Vec::new();

        for (nid, required) in &nodes {
            if required.is_empty() {
                continue;
            }
            let req_set: std::collections::HashSet<&str> =
                required.iter().map(|s| s.as_str()).collect();
            let mut pvars = Vec::new();

            for (pname, pcaps, exclusions) in &providers {
                let var = tm.mk_var(
                    &format!("a_{}_{}", nid, pname),
                    tm.sorts.bool_sort,
                );
                let pcap_set: std::collections::HashSet<&str> =
                    pcaps.iter().map(|s| s.as_str()).collect();
                let has_all = req_set.is_subset(&pcap_set);
                let has_conflict = exclusions
                    .iter()
                    .any(|(a, b)| req_set.contains(a.as_str()) && req_set.contains(b.as_str()));

                if !has_all || has_conflict {
                    let f = tm.mk_false();
                    let eq = tm.mk_eq(var, f);
                    solver.assert(eq, &mut tm);
                }
                pvars.push((pname.clone(), var));
            }

            if !pvars.is_empty() {
                // At-least-one
                let vars: Vec<TermId> = pvars.iter().map(|(_, v)| *v).collect();
                let alo = tm.mk_or(vars);
                solver.assert(alo, &mut tm);

                // At-most-one (pairwise negation)
                for i in 0..pvars.len() {
                    for j in (i + 1)..pvars.len() {
                        let both = tm.mk_and(vec![pvars[i].1, pvars[j].1]);
                        let nb = tm.mk_not(both);
                        solver.assert(nb, &mut tm);
                    }
                }
                all_pvars.push((nid.clone(), pvars));
            }
        }

        if solver.check(&mut tm) == SolverResult::Sat {
            return (true, vec![]);
        }

        // UNSAT — build counterexample
        let mut unassignable = Vec::new();
        for (nid, required) in &nodes {
            if required.is_empty() {
                continue;
            }
            let req_set: std::collections::HashSet<&str> =
                required.iter().map(|s| s.as_str()).collect();
            let mut reasons = Vec::new();
            for (pname, pcaps, exclusions) in &providers {
                let pcap_set: std::collections::HashSet<&str> =
                    pcaps.iter().map(|s| s.as_str()).collect();
                let missing: Vec<_> = req_set.difference(&pcap_set).collect();
                let conflicts: Vec<String> = exclusions
                    .iter()
                    .filter(|(a, b)| {
                        req_set.contains(a.as_str()) && req_set.contains(b.as_str())
                    })
                    .map(|(a, b)| format!("{}+{}", a, b))
                    .collect();
                if !missing.is_empty() {
                    reasons.push(format!("{}: missing {:?}", pname, missing));
                } else if !conflicts.is_empty() {
                    reasons.push(format!("{}: exclusion conflict {:?}", pname, conflicts));
                }
            }
            if !reasons.is_empty() {
                unassignable.push(format!("node '{}' ({})", nid, reasons.join(", ")));
            }
        }
        (false, unassignable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_safety_valid() {
        let v = SmtVerifier::new();
        assert!(v.prove_memory_safety(5, 100));
        assert!(v.prove_memory_safety(0, 10));
        assert!(v.prove_memory_safety(99, 100));
    }

    #[test]
    fn test_memory_safety_invalid() {
        let v = SmtVerifier::new();
        assert!(!v.prove_memory_safety(100, 100));
        assert!(!v.prove_memory_safety(-1, 100));
        assert!(!v.prove_memory_safety(200, 100));
    }

    #[test]
    fn test_loop_bound_unconstrained() {
        let v = SmtVerifier::new();
        // Unconstrained variable → cannot prove bounded → False
        assert!(!v.check_loop_bound("n", 1000000));
    }

    #[test]
    fn test_arithmetic_within_tolerance() {
        let v = SmtVerifier::new();
        assert!(v.verify_arithmetic(10, 10, 0));
        assert!(v.verify_arithmetic(11, 10, 1));
        assert!(v.verify_arithmetic(9, 10, 1));
    }

    #[test]
    fn test_arithmetic_outside_tolerance() {
        let v = SmtVerifier::new();
        assert!(!v.verify_arithmetic(12, 10, 1));
        assert!(!v.verify_arithmetic(8, 10, 1));
    }

    #[test]
    fn test_validate_mutation_bounds() {
        let v = SmtVerifier::new();
        let result = v.validate_mutation(vec![
            "bounds(5, 100)".into(),
            "bounds(200, 100)".into(),
        ]);
        assert!(!result.safe);
        assert_eq!(result.violations.len(), 1);
        assert!(result.violations[0].contains("200"));
    }

    #[test]
    fn test_validate_mutation_loop() {
        let v = SmtVerifier::new();
        let result = v.validate_mutation(vec!["loop(n, 1000000)".into()]);
        assert!(!result.safe); // Unconstrained → violation
    }

    #[test]
    fn test_provider_assignment_sat() {
        let v = SmtVerifier::new();
        let nodes = vec![("n1".into(), vec!["code".into(), "reason".into()])];
        let providers = vec![(
            "gemini".into(),
            vec!["code".into(), "reason".into(), "chat".into()],
            vec![],
        )];
        let (sat, errors) = v.verify_provider_assignment(nodes, providers);
        assert!(sat);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_provider_assignment_unsat() {
        let v = SmtVerifier::new();
        let nodes = vec![("n1".into(), vec!["code".into(), "vision".into()])];
        let providers = vec![("gemini".into(), vec!["code".into()], vec![])];
        let (sat, errors) = v.verify_provider_assignment(nodes, providers);
        assert!(!sat);
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_provider_assignment_exclusion() {
        let v = SmtVerifier::new();
        let nodes = vec![("n1".into(), vec!["search".into(), "file_search".into()])];
        let providers = vec![(
            "google".into(),
            vec!["search".into(), "file_search".into()],
            vec![("search".into(), "file_search".into())],
        )];
        let (sat, errors) = v.verify_provider_assignment(nodes, providers);
        assert!(!sat);
        assert!(!errors.is_empty());
    }
}
