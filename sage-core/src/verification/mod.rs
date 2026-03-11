//! OxiZ-backed SMT verification for contracts and evolved code.
//!
//! Pure Rust replacement for Python `z3-solver` dependency.
//! Behind `smt` feature flag. Covers:
//! - Memory safety proofs (bounds checking)
//! - Loop bound verification
//! - Arithmetic verification (concrete and symbolic expressions)
//! - Invariant verification (pre/post-condition implication via `mk_implies`)
//! - Provider assignment (SAT with integer encoding — O(N) constraints)
//!
//! All operations are QF_LIA (quantifier-free linear integer arithmetic).

use oxiz::{Solver, SolverResult, TermId, TermManager};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

/// Helper: create solver+tm, assert formula, check UNSAT.
fn is_unsat(build: impl FnOnce(&mut TermManager) -> TermId) -> bool {
    let mut solver = Solver::new();
    let mut tm = TermManager::new();
    let formula = build(&mut tm);
    solver.assert(formula, &mut tm);
    solver.check(&mut tm) == SolverResult::Unsat
}

// ──────────────────────────────────────────────────────────────────────
// Expression AST for invariant/arithmetic string parsing
// ──────────────────────────────────────────────────────────────────────

#[derive(Debug)]
enum Expr {
    Var(String),
    Int(i64),
    Cmp(Box<Expr>, CmpOp, Box<Expr>),
    Arith(Box<Expr>, ArithOp, Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Not(Box<Expr>),
}

#[derive(Debug)]
enum CmpOp {
    Gt,
    Lt,
    Ge,
    Le,
    Eq,
    Ne,
}

#[derive(Debug)]
enum ArithOp {
    Add,
    Sub,
    Mul,
}

// ──────────────────────────────────────────────────────────────────────
// Recursive descent parser
//
// Grammar (precedence low→high):
//   expr     := or_expr
//   or_expr  := and_expr ("or" and_expr)*
//   and_expr := not_expr ("and" not_expr)*
//   not_expr := "not" not_expr | cmp_expr
//   cmp_expr := add_expr (CMP add_expr)?
//   add_expr := mul_expr (("+"|"-") mul_expr)*
//   mul_expr := unary ("*" unary)*
//   unary    := "-" unary | atom
//   atom     := "(" expr ")" | integer | variable
// ──────────────────────────────────────────────────────────────────────

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            bytes: input.as_bytes(),
            pos: 0,
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn remaining(&self) -> &[u8] {
        &self.bytes[self.pos..]
    }

    /// Consume a string literal. For alphabetic keywords, ensures no trailing alnum.
    fn try_consume(&mut self, s: &str) -> bool {
        self.skip_ws();
        let sb = s.as_bytes();
        if self.remaining().starts_with(sb) {
            if sb.iter().all(|b| b.is_ascii_alphabetic()) {
                let end = self.pos + sb.len();
                if end < self.bytes.len()
                    && (self.bytes[end].is_ascii_alphanumeric() || self.bytes[end] == b'_')
                {
                    return false;
                }
            }
            self.pos += sb.len();
            true
        } else {
            false
        }
    }

    /// Parse full expression and verify all input consumed.
    fn parse_all(&mut self) -> Result<Expr, ()> {
        let expr = self.parse_or()?;
        self.skip_ws();
        if self.pos == self.bytes.len() {
            Ok(expr)
        } else {
            Err(()) // Trailing garbage
        }
    }

    fn parse_or(&mut self) -> Result<Expr, ()> {
        let mut left = self.parse_and()?;
        while self.try_consume("or") {
            let right = self.parse_and()?;
            left = Expr::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, ()> {
        let mut left = self.parse_not()?;
        while self.try_consume("and") {
            let right = self.parse_not()?;
            left = Expr::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_not(&mut self) -> Result<Expr, ()> {
        if self.try_consume("not") {
            let inner = self.parse_not()?;
            Ok(Expr::Not(Box::new(inner)))
        } else {
            self.parse_cmp()
        }
    }

    fn parse_cmp(&mut self) -> Result<Expr, ()> {
        let left = self.parse_add()?;
        self.skip_ws();
        // Order matters: >= before >, <= before <, == and != before single chars
        let op = if self.try_consume(">=") {
            Some(CmpOp::Ge)
        } else if self.try_consume("<=") {
            Some(CmpOp::Le)
        } else if self.try_consume("==") {
            Some(CmpOp::Eq)
        } else if self.try_consume("!=") {
            Some(CmpOp::Ne)
        } else if self.try_consume(">") {
            Some(CmpOp::Gt)
        } else if self.try_consume("<") {
            Some(CmpOp::Lt)
        } else {
            None
        };
        if let Some(op) = op {
            let right = self.parse_add()?;
            Ok(Expr::Cmp(Box::new(left), op, Box::new(right)))
        } else {
            Ok(left)
        }
    }

    fn parse_add(&mut self) -> Result<Expr, ()> {
        let mut left = self.parse_mul()?;
        loop {
            self.skip_ws();
            if self.try_consume("+") {
                let right = self.parse_mul()?;
                left = Expr::Arith(Box::new(left), ArithOp::Add, Box::new(right));
            } else if self.try_consume("-") {
                let right = self.parse_mul()?;
                left = Expr::Arith(Box::new(left), ArithOp::Sub, Box::new(right));
            } else {
                break;
            }
        }
        Ok(left)
    }

    fn parse_mul(&mut self) -> Result<Expr, ()> {
        let mut left = self.parse_unary()?;
        loop {
            self.skip_ws();
            if self.try_consume("*") {
                let right = self.parse_unary()?;
                left = Expr::Arith(Box::new(left), ArithOp::Mul, Box::new(right));
            } else {
                break;
            }
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr, ()> {
        self.skip_ws();
        if self.peek() == Some(b'-') {
            // Peek ahead: if next is digit, parse as negative literal
            if self.pos + 1 < self.bytes.len() && self.bytes[self.pos + 1].is_ascii_digit() {
                return self.parse_int_literal();
            }
            // Otherwise: unary minus (0 - expr)
            self.pos += 1;
            let inner = self.parse_unary()?;
            Ok(Expr::Arith(
                Box::new(Expr::Int(0)),
                ArithOp::Sub,
                Box::new(inner),
            ))
        } else {
            self.parse_atom()
        }
    }

    fn parse_int_literal(&mut self) -> Result<Expr, ()> {
        self.skip_ws();
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        if !matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos = start;
            return Err(());
        }
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos]).map_err(|_| ())?;
        let val: i64 = s.parse().map_err(|_| ())?;
        Ok(Expr::Int(val))
    }

    fn parse_atom(&mut self) -> Result<Expr, ()> {
        self.skip_ws();

        // Parenthesized expression
        if self.try_consume("(") {
            let expr = self.parse_or()?;
            self.skip_ws();
            if !self.try_consume(")") {
                return Err(());
            }
            return Ok(expr);
        }

        // Integer literal
        if matches!(self.peek(), Some(b'0'..=b'9')) {
            return self.parse_int_literal();
        }

        // Variable name
        if matches!(
            self.peek(),
            Some(b'a'..=b'z') | Some(b'A'..=b'Z') | Some(b'_')
        ) {
            let start = self.pos;
            while matches!(
                self.peek(),
                Some(b'a'..=b'z') | Some(b'A'..=b'Z') | Some(b'0'..=b'9') | Some(b'_')
            ) {
                self.pos += 1;
            }
            let name = std::str::from_utf8(&self.bytes[start..self.pos]).map_err(|_| ())?;
            // Reject Python builtins / injection vectors
            if matches!(
                name,
                "import"
                    | "exec"
                    | "eval"
                    | "open"
                    | "print"
                    | "compile"
                    | "getattr"
                    | "setattr"
                    | "delattr"
                    | "globals"
                    | "locals"
            ) {
                return Err(());
            }
            // Reject dunder names (__anything__)
            if name.starts_with("__") {
                return Err(());
            }
            return Ok(Expr::Var(name.to_string()));
        }

        Err(())
    }
}

// ──────────────────────────────────────────────────────────────────────
// AST → OxiZ TermId conversion
// ──────────────────────────────────────────────────────────────────────

fn expr_to_term(
    expr: &Expr,
    tm: &mut TermManager,
    vars: &mut HashMap<String, TermId>,
) -> Result<TermId, ()> {
    match expr {
        Expr::Var(name) => {
            if let Some(&id) = vars.get(name.as_str()) {
                Ok(id)
            } else {
                let id = tm.mk_var(name, tm.sorts.int_sort);
                vars.insert(name.clone(), id);
                Ok(id)
            }
        }
        Expr::Int(val) => Ok(tm.mk_int(*val)),
        Expr::Cmp(lhs, op, rhs) => {
            let l = expr_to_term(lhs, tm, vars)?;
            let r = expr_to_term(rhs, tm, vars)?;
            Ok(match op {
                CmpOp::Gt => tm.mk_gt(l, r),
                CmpOp::Lt => tm.mk_lt(l, r),
                CmpOp::Ge => tm.mk_ge(l, r),
                CmpOp::Le => tm.mk_le(l, r),
                CmpOp::Eq => tm.mk_eq(l, r),
                CmpOp::Ne => {
                    let eq = tm.mk_eq(l, r);
                    tm.mk_not(eq)
                }
            })
        }
        Expr::Arith(lhs, op, rhs) => {
            let l = expr_to_term(lhs, tm, vars)?;
            let r = expr_to_term(rhs, tm, vars)?;
            Ok(match op {
                ArithOp::Add => tm.mk_add(vec![l, r]),
                ArithOp::Sub => tm.mk_sub(l, r),
                ArithOp::Mul => tm.mk_mul(vec![l, r]),
            })
        }
        Expr::And(lhs, rhs) => {
            let l = expr_to_term(lhs, tm, vars)?;
            let r = expr_to_term(rhs, tm, vars)?;
            Ok(tm.mk_and(vec![l, r]))
        }
        Expr::Or(lhs, rhs) => {
            let l = expr_to_term(lhs, tm, vars)?;
            let r = expr_to_term(rhs, tm, vars)?;
            Ok(tm.mk_or(vec![l, r]))
        }
        Expr::Not(inner) => {
            let t = expr_to_term(inner, tm, vars)?;
            Ok(tm.mk_not(t))
        }
    }
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
/// - Arithmetic verification (concrete values and string expressions)
/// - Invariant verification (pre/post-condition implication)
/// - Provider assignment (integer-encoded constraint satisfaction)
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

    /// Verify an arithmetic expression string evaluates within tolerance.
    ///
    /// Parses expressions like "2 + 2", "10 * 3 + 1", "100 - 50".
    /// For constant expressions, proves the result equals expected ± tolerance.
    /// Returns false (fail-closed) on parse errors or symbolic expressions.
    pub fn verify_arithmetic_expr(&self, expr: &str, expected: i64, tolerance: i64) -> bool {
        let parsed = match Parser::new(expr).parse_all() {
            Ok(e) => e,
            Err(()) => return false,
        };
        let mut solver = Solver::new();
        let mut tm = TermManager::new();
        let mut vars = HashMap::new();
        let term = match expr_to_term(&parsed, &mut tm, &mut vars) {
            Ok(t) => t,
            Err(()) => return false,
        };
        // Symbolic expressions can't be proven to equal a constant
        if !vars.is_empty() {
            return false;
        }
        let lo = tm.mk_int(expected - tolerance);
        let hi = tm.mk_int(expected + tolerance);
        let too_low = tm.mk_lt(term, lo);
        let too_high = tm.mk_gt(term, hi);
        let violation = tm.mk_or(vec![too_low, too_high]);
        solver.assert(violation, &mut tm);
        solver.check(&mut tm) == SolverResult::Unsat
    }

    /// Verify a pre/post-condition pair (invariant implication).
    ///
    /// Parses string expressions (e.g. "x > 0", "x >= -1 and x < 100")
    /// and checks if pre → post holds for all values of free variables.
    /// Uses `mk_implies`: asserts ¬(pre → post) and checks UNSAT.
    ///
    /// Fails closed (returns False) on parse errors — no eval(), no injection.
    pub fn verify_invariant(&self, pre: &str, post: &str) -> bool {
        let pre_expr = match Parser::new(pre).parse_all() {
            Ok(e) => e,
            Err(()) => return false,
        };
        let post_expr = match Parser::new(post).parse_all() {
            Ok(e) => e,
            Err(()) => return false,
        };

        let mut solver = Solver::new();
        let mut tm = TermManager::new();
        let mut vars = HashMap::new();

        let pre_term = match expr_to_term(&pre_expr, &mut tm, &mut vars) {
            Ok(t) => t,
            Err(()) => return false,
        };
        let post_term = match expr_to_term(&post_expr, &mut tm, &mut vars) {
            Ok(t) => t,
            Err(()) => return false,
        };

        // pre → post is valid iff pre ∧ ¬post is UNSAT
        let not_post = tm.mk_not(post_term);
        let formula = tm.mk_and(vec![pre_term, not_post]);
        solver.assert(formula, &mut tm);
        solver.check(&mut tm) == SolverResult::Unsat
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
    /// For each node, creates boolean variables (one per provider).
    /// Exactly-one encoding: at-least-one (OR) + at-most-one (pairwise NOT-AND).
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
                pvars.push(var);
            }

            if !pvars.is_empty() {
                // At-least-one
                let alo = tm.mk_or(pvars.clone());
                solver.assert(alo, &mut tm);

                // At-most-one (pairwise negation)
                for i in 0..pvars.len() {
                    for j in (i + 1)..pvars.len() {
                        let both = tm.mk_and(vec![pvars[i], pvars[j]]);
                        let nb = tm.mk_not(both);
                        solver.assert(nb, &mut tm);
                    }
                }
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

    // ── Memory safety ────────────────────────────────────────────────

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

    // ── Loop bounds ──────────────────────────────────────────────────

    #[test]
    fn test_loop_bound_unconstrained() {
        let v = SmtVerifier::new();
        assert!(!v.check_loop_bound("n", 1000000));
    }

    // ── Concrete arithmetic ──────────────────────────────────────────

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

    // ── Expression arithmetic ────────────────────────────────────────

    #[test]
    fn test_arithmetic_expr_constants() {
        let v = SmtVerifier::new();
        assert!(v.verify_arithmetic_expr("42", 42, 0));
        assert!(!v.verify_arithmetic_expr("42", 43, 0));
        assert!(v.verify_arithmetic_expr("42", 43, 1));
    }

    #[test]
    fn test_arithmetic_expr_operations() {
        let v = SmtVerifier::new();
        assert!(v.verify_arithmetic_expr("2 + 2", 4, 0));
        assert!(v.verify_arithmetic_expr("10 * 3 + 1", 31, 0));
        assert!(v.verify_arithmetic_expr("100 - 50", 50, 0));
        assert!(v.verify_arithmetic_expr("7 * 6", 42, 0));
    }

    #[test]
    fn test_arithmetic_expr_negative() {
        let v = SmtVerifier::new();
        assert!(v.verify_arithmetic_expr("-5", -5, 0));
        assert!(v.verify_arithmetic_expr("10 - 15", -5, 0));
    }

    #[test]
    fn test_arithmetic_expr_symbolic_rejected() {
        let v = SmtVerifier::new();
        // Free variable → can't prove concrete value
        assert!(!v.verify_arithmetic_expr("x + 1", 5, 0));
    }

    #[test]
    fn test_arithmetic_expr_unparseable() {
        let v = SmtVerifier::new();
        assert!(!v.verify_arithmetic_expr("", 0, 0));
        assert!(!v.verify_arithmetic_expr("invalid!!", 0, 0));
    }

    // ── Invariant verification ───────────────────────────────────────

    #[test]
    fn test_invariant_simple_implication() {
        let v = SmtVerifier::new();
        // x > 0 → x > 0 (trivially true)
        assert!(v.verify_invariant("x > 0", "x > 0"));
        // x > 0 → x > -1 (true: any x > 0 is also > -1)
        assert!(v.verify_invariant("x > 0", "x > -1"));
        // x > 0 → x >= 0 (true: x > 0 means x >= 1)
        assert!(v.verify_invariant("x > 0", "x >= 0"));
        // x > 5 → x > 3 (true)
        assert!(v.verify_invariant("x > 5", "x > 3"));
    }

    #[test]
    fn test_invariant_false_implication() {
        let v = SmtVerifier::new();
        // x > 0 does NOT imply x > 5 (counterexample: x = 1)
        assert!(!v.verify_invariant("x > 0", "x > 5"));
        // x >= 0 does NOT imply x > 0 (counterexample: x = 0)
        assert!(!v.verify_invariant("x >= 0", "x > 0"));
    }

    #[test]
    fn test_invariant_compound() {
        let v = SmtVerifier::new();
        // x > 0 and x < 10 → x >= 0
        assert!(v.verify_invariant("x > 0 and x < 10", "x >= 0"));
        // x > 0 and x < 10 → x < 100
        assert!(v.verify_invariant("x > 0 and x < 10", "x < 100"));
        // x > 0 and x < 10 does NOT imply x > 5 (x could be 1)
        assert!(!v.verify_invariant("x > 0 and x < 10", "x > 5"));
    }

    #[test]
    fn test_invariant_arithmetic_in_pre() {
        let v = SmtVerifier::new();
        // x + 1 > 5 → x > 4 (true)
        assert!(v.verify_invariant("x + 1 > 5", "x > 4"));
        // x * 2 > 10 → x > 5 (true in integers)
        assert!(v.verify_invariant("x * 2 > 10", "x > 5"));
    }

    #[test]
    fn test_invariant_equality() {
        let v = SmtVerifier::new();
        // x == 5 → x > 0
        assert!(v.verify_invariant("x == 5", "x > 0"));
        // x == 0 → x >= 0
        assert!(v.verify_invariant("x == 0", "x >= 0"));
        // x == 0 does NOT imply x > 0
        assert!(!v.verify_invariant("x == 0", "x > 0"));
    }

    #[test]
    fn test_invariant_injection_blocked() {
        let v = SmtVerifier::new();
        assert!(!v.verify_invariant("__import__('os').system('ls')", "x > 0"));
        assert!(!v.verify_invariant("open('/etc/passwd')", "x > 0"));
        assert!(!v.verify_invariant("exec('malicious')", "x > 0"));
        assert!(!v.verify_invariant("eval('1+1')", "x > 0"));
    }

    #[test]
    fn test_invariant_unparseable() {
        let v = SmtVerifier::new();
        assert!(!v.verify_invariant("", "x > 0"));
        assert!(!v.verify_invariant("x > 0", ""));
        assert!(!v.verify_invariant("garbage!!!", "x > 0"));
        assert!(!v.verify_invariant("x > 0", "y > 0; drop table"));
    }

    // ── Mutation validation ──────────────────────────────────────────

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
        assert!(!result.safe);
    }

    // ── Provider assignment ──────────────────────────────────────────

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

    #[test]
    fn test_provider_assignment_multi_provider() {
        let v = SmtVerifier::new();
        let nodes = vec![
            ("n1".into(), vec!["code".into()]),
            ("n2".into(), vec!["vision".into()]),
        ];
        let providers = vec![
            ("openai".into(), vec!["code".into(), "vision".into()], vec![]),
            ("google".into(), vec!["code".into()], vec![]),
        ];
        let (sat, errors) = v.verify_provider_assignment(nodes, providers);
        assert!(sat);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_provider_assignment_no_providers_with_reqs() {
        let v = SmtVerifier::new();
        let nodes = vec![("n1".into(), vec!["code".into()])];
        let providers: Vec<(String, Vec<String>, Vec<(String, String)>)> = vec![];
        let (sat, _) = v.verify_provider_assignment(nodes, providers);
        assert!(!sat);
    }

    // ── Parser unit tests ────────────────────────────────────────────

    #[test]
    fn test_parser_simple_comparison() {
        let mut p = Parser::new("x > 0");
        assert!(p.parse_all().is_ok());
    }

    #[test]
    fn test_parser_compound_and() {
        let mut p = Parser::new("x > 0 and x < 10");
        assert!(p.parse_all().is_ok());
    }

    #[test]
    fn test_parser_compound_or() {
        let mut p = Parser::new("x < 0 or x > 100");
        assert!(p.parse_all().is_ok());
    }

    #[test]
    fn test_parser_arithmetic() {
        let mut p = Parser::new("x + 1 > 5");
        assert!(p.parse_all().is_ok());
    }

    #[test]
    fn test_parser_negative_literal() {
        let mut p = Parser::new("x > -1");
        assert!(p.parse_all().is_ok());
    }

    #[test]
    fn test_parser_parentheses() {
        let mut p = Parser::new("(x > 0) and (x < 10)");
        assert!(p.parse_all().is_ok());
    }

    #[test]
    fn test_parser_rejects_injection() {
        let mut p = Parser::new("__import__('os')");
        assert!(p.parse_all().is_err());
    }

    #[test]
    fn test_parser_rejects_trailing_garbage() {
        let mut p = Parser::new("x > 0 ; drop");
        assert!(p.parse_all().is_err());
    }
}
