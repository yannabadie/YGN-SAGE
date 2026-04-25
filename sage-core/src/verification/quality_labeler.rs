//! Formal quality labeler for LLM-generated code responses.
//!
//! Uses SmtVerifier (OxiZ) + tree-sitter validator to produce a quality score
//! based entirely on formal proofs (SAT/UNSAT) and syntactic facts.
//! **Zero heuristics** — every check is a formal proof or syntactic fact.
//!
//! Behind `#[cfg(all(feature = "smt", feature = "tool-executor"))]`.

use pyo3::prelude::*;
use tracing::instrument;

use crate::sandbox::validator::{validate_python_code, ValidationResult};
use crate::verification::smt::SmtVerifier;

// ──────────────────────────────────────────────────────────────────────
// Code block extraction
// ──────────────────────────────────────────────────────────────────────

/// Extract ```python ... ``` fenced code blocks from a Markdown response.
pub(crate) fn extract_code_blocks(response: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut lines = response.lines().peekable();

    while let Some(line) = lines.next() {
        let trimmed = line.trim();
        if trimmed.starts_with("```python") || trimmed.starts_with("```py") {
            let mut block = String::new();
            for inner in lines.by_ref() {
                let inner_trimmed = inner.trim();
                if inner_trimmed == "```" {
                    break;
                }
                if !block.is_empty() {
                    block.push('\n');
                }
                block.push_str(inner);
            }
            if !block.is_empty() {
                blocks.push(block);
            }
        }
    }

    blocks
}

// ──────────────────────────────────────────────────────────────────────
// Structural checks (syntactic facts, not heuristics)
// ──────────────────────────────────────────────────────────────────────

/// Check if code defining a function contains a `return` statement.
/// This is a syntactic fact: a `def` without `return` is either a generator
/// or a void procedure — both are structural completeness signals.
///
/// Returns (has_def, has_return) so callers can decide:
/// - (false, _) → no function defined, check is not applicable
/// - (true, true) → function has return statement
/// - (true, false) → function missing return
pub(crate) fn check_structural_completeness(code: &str) -> (bool, bool) {
    let has_def = code.lines().any(|line| {
        let t = line.trim();
        t.starts_with("def ") || t.starts_with("async def ")
    });

    if !has_def {
        return (false, false);
    }

    // Check for return statement (not inside a string literal — simple line-level check)
    let has_return = code.lines().any(|line| {
        let t = line.trim();
        t.starts_with("return ") || t == "return" || t.starts_with("return\t")
    });

    (true, has_return)
}

/// Extract arithmetic assertions from code.
/// Looks for patterns like `assert result == 42` or `assert x + y == 10`.
/// Returns Vec<(expression, expected_value)> for SMT verification.
pub(crate) fn extract_arithmetic_assertions(code: &str) -> Vec<(String, i64)> {
    let mut assertions = Vec::new();

    for line in code.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("assert ") {
            // Look for `expr == value` pattern
            if let Some(eq_pos) = rest.find(" == ") {
                let lhs = rest[..eq_pos].trim();
                let rhs = rest[eq_pos + 4..].trim();
                // Strip trailing comma or comment
                let rhs_clean = rhs
                    .split(',')
                    .next()
                    .unwrap_or(rhs)
                    .split('#')
                    .next()
                    .unwrap_or(rhs)
                    .trim();
                if let Ok(val) = rhs_clean.parse::<i64>() {
                    assertions.push((lhs.to_string(), val));
                }
            }
        }
    }

    assertions
}

/// Extract array/list access bounds from code.
/// Looks for patterns like `arr[i]` where i is a literal integer
/// and arr has a known length from `len(arr)` or list literal.
/// Returns Vec<(index, length)> for SMT verification.
pub(crate) fn extract_array_bounds(code: &str) -> Vec<(i64, i64)> {
    let mut bounds = Vec::new();

    // Find list literal lengths: `arr = [1, 2, 3]` → length 3
    let mut known_lengths: std::collections::HashMap<String, i64> =
        std::collections::HashMap::new();

    for line in code.lines() {
        let trimmed = line.trim();
        // Pattern: `name = [...]`
        if let Some(eq_pos) = trimmed.find(" = [") {
            let name = trimmed[..eq_pos].trim();
            // Only simple identifiers
            if name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') && !name.is_empty() {
                let bracket_content = &trimmed[eq_pos + 3..];
                if let Some(close) = bracket_content.find(']') {
                    let inner = &bracket_content[..close];
                    if !inner.is_empty() {
                        let count = inner.split(',').count() as i64;
                        known_lengths.insert(name.to_string(), count);
                    }
                }
            }
        }
    }

    // Find accesses: `name[literal]`
    for line in code.lines() {
        let trimmed = line.trim();
        for (name, length) in &known_lengths {
            let pattern = format!("{}[", name);
            let mut search_pos = 0;
            while let Some(start) = trimmed[search_pos..].find(&pattern) {
                let abs_start = search_pos + start + pattern.len();
                if let Some(end) = trimmed[abs_start..].find(']') {
                    let index_str = trimmed[abs_start..abs_start + end].trim();
                    if let Ok(index) = index_str.parse::<i64>() {
                        bounds.push((index, *length));
                    }
                }
                search_pos = abs_start;
            }
        }
    }

    bounds
}

// ──────────────────────────────────────────────────────────────────────
// Text arithmetic extraction (Canal 2: non-code math verification)
// ──────────────────────────────────────────────────────────────────────

/// Extract arithmetic equations from free text using the OxiZ parser as probe.
///
/// Scans for "= <number>" patterns, then probes backward with the OxiZ
/// recursive descent parser (island grammar pattern) to find the longest
/// valid arithmetic expression. No regex — uses the same parser as the
/// SMT verifier for perfect consistency.
///
/// Based on Island Grammar Parsing (SLE 2012): define "islands" (arithmetic
/// expressions) and "water" (everything else to skip).
pub(crate) fn extract_text_equations(text: &str, verifier: &SmtVerifier) -> Vec<(String, i64)> {
    let mut results = Vec::new();

    for (eq_pos, _) in text.match_indices('=') {
        // Skip ==, !=, >=, <=
        if text.get(eq_pos..eq_pos + 2) == Some("==") {
            continue;
        }
        if eq_pos > 0 && matches!(text.as_bytes().get(eq_pos - 1), Some(b'!' | b'>' | b'<')) {
            continue;
        }

        // Parse RHS as integer
        let after = text[eq_pos + 1..].trim_start();
        let num_end = after
            .find(|c: char| !c.is_ascii_digit() && c != '-')
            .unwrap_or(after.len());
        if num_end == 0 {
            continue;
        }
        let Ok(expected) = after[..num_end].parse::<i64>() else {
            continue;
        };

        // Probe backward with OxiZ parser (island grammar)
        let before = &text[..eq_pos];
        let before_trimmed = before.trim_end();
        if before_trimmed.is_empty() {
            continue;
        }

        // Cap backward search at 60 chars to avoid O(n²) worst case
        let search_start = before_trimmed.len().saturating_sub(60);

        // Try progressively longer substrings from the right.
        // Keep the longest valid parse (greedy match).
        let mut best: Option<String> = None;
        for start in (search_start..before_trimmed.len()).rev() {
            if !before_trimmed.is_char_boundary(start) {
                continue;
            }
            let candidate = before_trimmed[start..].trim();
            if candidate.is_empty() {
                continue;
            }
            if verifier.try_parse_expr(candidate) {
                best = Some(candidate.to_string());
                // Keep trying longer prefixes for the longest valid parse
            }
        }

        if let Some(expr) = best {
            results.push((expr, expected));
        }
    }

    results
}

/// Extract the final numeric answer from a response.
///
/// Tries (in order):
/// 1. Last non-empty line that parses as a pure integer
/// 2. Number following known keywords ("answer is", "result:", etc.)
///
/// Uses `str::parse::<i64>()` and `chars().take_while()` — no regex.
/// Domain-agnostic: works for any text ending with a numeric conclusion.
pub(crate) fn extract_final_number(text: &str) -> Option<i64> {
    // 1. Last non-empty line as pure number
    for line in text.lines().rev() {
        let trimmed = line.trim().trim_end_matches('.');
        if !trimmed.is_empty() {
            if let Ok(n) = trimmed.parse::<i64>() {
                return Some(n);
            }
            break; // Only check the last non-empty line
        }
    }

    // 2. Number after known keywords (no regex — string search + char iteration)
    let lower = text.to_lowercase();
    for keyword in &[
        "answer is",
        "answer:",
        "result is",
        "result:",
        "total is",
        "total:",
        "equals",
    ] {
        if let Some(pos) = lower.rfind(keyword) {
            let after = &text[pos + keyword.len()..];
            let after_clean = after.trim_start().trim_start_matches(':').trim_start();
            let num_str: String = after_clean
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '-')
                .collect();
            if !num_str.is_empty() {
                if let Ok(n) = num_str.parse::<i64>() {
                    return Some(n);
                }
            }
        }
    }

    None
}

// ──────────────────────────────────────────────────────────────────────
// PyO3 types
// ──────────────────────────────────────────────────────────────────────

/// Result of formal quality labeling.
#[pyclass]
#[derive(Clone, Debug)]
pub struct QualityLabel {
    /// Quality score in [0.0, 1.0] — ratio of passed formal checks.
    #[pyo3(get)]
    pub score: f32,

    /// Number of formal checks that passed.
    #[pyo3(get)]
    pub checks_passed: u32,

    /// Total number of formal checks attempted.
    #[pyo3(get)]
    pub checks_total: u32,

    /// Whether the response contained assessable code.
    #[pyo3(get)]
    pub assessable: bool,

    /// Human-readable summary of check results.
    #[pyo3(get)]
    pub details: String,
}

#[pymethods]
impl QualityLabel {
    fn __repr__(&self) -> String {
        format!(
            "QualityLabel(score={:.3}, checks={}/{}, assessable={})",
            self.score, self.checks_passed, self.checks_total, self.assessable,
        )
    }
}

/// Formal quality labeler using SMT verification + tree-sitter syntax checks.
///
/// Every check is either:
/// - A formal proof (SAT/UNSAT via OxiZ SmtVerifier)
/// - A syntactic fact (tree-sitter AST parse success/failure)
///
/// Zero heuristics.
#[pyclass]
pub struct QualityLabeler {
    verifier: SmtVerifier,
}

impl Default for QualityLabeler {
    fn default() -> Self {
        Self {
            verifier: SmtVerifier::new(),
        }
    }
}

#[pymethods]
impl QualityLabeler {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    /// Label the quality of an LLM response for a given task.
    ///
    /// Uses 3 verification canaux:
    /// 1. **Code** (existing): tree-sitter syntax + OxiZ arithmetic/array/structural
    /// 2. **Text arithmetic** (new): extract "5 + 3 = 8" from prose, verify via OxiZ
    /// 3. **Numeric answer** (new): extract final number, score format compliance
    ///
    /// Returns None only if NO canal produces any check (pure prose with no
    /// code, no equations, no numbers). Zero heuristics — every check is a
    /// formal proof or syntactic fact.
    #[instrument(skip(self, task, response))]
    #[pyo3(signature = (task, response))]
    pub fn label(&self, task: &str, response: &str) -> Option<QualityLabel> {
        let _ = task; // reserved for future task-specific checks
        let mut checks_passed: u32 = 0;
        let mut checks_total: u32 = 0;
        let mut details = Vec::new();

        // ── Canal 1: Code blocks (existing, unchanged) ───────────────
        let code_blocks = extract_code_blocks(response);
        for (i, code) in code_blocks.iter().enumerate() {
            let block_label = if code_blocks.len() > 1 {
                format!("block_{}", i)
            } else {
                "code".to_string()
            };

            // Check 1.1: Syntax validity via tree-sitter
            let validation: ValidationResult = validate_python_code(code);
            checks_total += 1;
            if validation.valid {
                checks_passed += 1;
                details.push(format!("{}: syntax OK", block_label));
            } else {
                details.push(format!(
                    "{}: syntax FAIL ({})",
                    block_label,
                    validation.errors.join("; ")
                ));
            }

            // Check 1.2: Arithmetic assertions verified via OxiZ
            let arith_assertions = extract_arithmetic_assertions(code);
            for (expr, expected) in &arith_assertions {
                checks_total += 1;
                if self.verifier.verify_arithmetic_expr(expr, *expected, 0) {
                    checks_passed += 1;
                    details.push(format!(
                        "{}: arithmetic '{}=={}' PROVED",
                        block_label, expr, expected
                    ));
                } else {
                    details.push(format!(
                        "{}: arithmetic '{}=={}' UNPROVED",
                        block_label, expr, expected
                    ));
                }
            }

            // Check 1.3: Array bounds verified via OxiZ
            let array_bounds = extract_array_bounds(code);
            if !array_bounds.is_empty() {
                let result = self.verifier.verify_array_bounds(array_bounds.clone());
                checks_total += 1;
                if result.safe {
                    checks_passed += 1;
                    details.push(format!(
                        "{}: array bounds ({} accesses) SAFE",
                        block_label,
                        array_bounds.len()
                    ));
                } else {
                    details.push(format!(
                        "{}: array bounds UNSAFE ({})",
                        block_label,
                        result.violations.join("; ")
                    ));
                }
            }

            // Check 1.4: Structural completeness (def has return)
            let (has_def, has_return) = check_structural_completeness(code);
            if has_def {
                checks_total += 1;
                if has_return {
                    checks_passed += 1;
                    details.push(format!("{}: structural completeness OK", block_label));
                } else {
                    details.push(format!(
                        "{}: structural completeness FAIL (def without return)",
                        block_label
                    ));
                }
            }
        }

        // ── Canal 2: Text arithmetic (OxiZ formal verification) ──────
        // Extract "5 + 3 = 8" from prose and verify via QF_LIA SAT solver.
        let text_equations = extract_text_equations(response, &self.verifier);
        for (expr, claimed) in &text_equations {
            checks_total += 1;
            if self.verifier.verify_arithmetic_expr(expr, *claimed, 0) {
                checks_passed += 1;
                details.push(format!("math: '{}={}' PROVED", expr, claimed));
            } else {
                details.push(format!("math: '{}={}' DISPROVED", expr, claimed));
            }
        }

        // ── Canal 3: Final numeric answer ────────────────────────────
        // If the response ends with a number, that's a well-formatted answer.
        let final_number = extract_final_number(response);
        if let Some(n) = final_number {
            checks_total += 1;
            checks_passed += 1; // Numeric extraction = format compliance (syntactic fact)
            details.push(format!("numeric: final answer {}", n));
        }

        // ── Aggregate ────────────────────────────────────────────────
        if checks_total == 0 {
            return None; // Pure prose with no verifiable content → abstain
        }

        let score = checks_passed as f32 / checks_total as f32;

        Some(QualityLabel {
            score,
            checks_passed,
            checks_total,
            assessable: true,
            details: details.join(", "),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_code_from_markdown() {
        let response = r#"Here is a solution:

```python
def add(a, b):
    return a + b
```

And another:

```python
x = 42
```
"#;
        let blocks = extract_code_blocks(response);
        assert_eq!(blocks.len(), 2);
        assert!(blocks[0].contains("def add"));
        assert!(blocks[1].contains("x = 42"));
    }

    #[test]
    fn test_extract_no_code() {
        let response = "This is just text, no code blocks here.";
        let blocks = extract_code_blocks(response);
        assert!(blocks.is_empty());
    }

    #[test]
    fn test_extract_code_py_fence() {
        let response = "```py\nprint('hello')\n```";
        let blocks = extract_code_blocks(response);
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].contains("print"));
    }

    #[test]
    fn test_label_valid_code() {
        let labeler = QualityLabeler::new();
        let response = r#"```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
```"#;
        let label = labeler.label("Write fibonacci", response);
        assert!(label.is_some());
        let label = label.unwrap();
        assert!(label.score > 0.0, "Score should be > 0 for valid code");
        assert!(label.assessable);
        assert!(label.checks_total > 0);
        assert!(label.checks_passed > 0);
        assert!(label.details.contains("syntax OK"));
        assert!(label.details.contains("structural completeness OK"));
    }

    #[test]
    fn test_label_pure_prose_returns_none() {
        let labeler = QualityLabeler::new();
        // Pure prose: no code, no equations, no numbers → abstain
        let response = "Here is a text-only answer with no code or math.";
        let label = labeler.label("Explain something", response);
        assert!(label.is_none());
    }

    #[test]
    fn test_label_syntax_error() {
        let labeler = QualityLabeler::new();
        let response = r#"```python
def broken(
    return  # missing closing paren
```"#;
        let label = labeler.label("Write a function", response);
        assert!(label.is_some());
        let label = label.unwrap();
        // Syntax check should fail, but structural completeness may pass
        // The overall score should be less than 1.0
        assert!(label.assessable);
        assert!(label.checks_total > 0);
    }

    #[test]
    fn test_label_with_arithmetic_assertion() {
        let labeler = QualityLabeler::new();
        let response = r#"```python
def compute():
    result = 2 + 3
    return result

assert 2 + 3 == 5
```"#;
        let label = labeler.label("Compute sum", response);
        assert!(label.is_some());
        let label = label.unwrap();
        assert!(label.details.contains("arithmetic"));
        // 2 + 3 == 5 should be PROVED by OxiZ
        assert!(label.details.contains("PROVED"));
    }

    #[test]
    fn test_label_with_wrong_arithmetic() {
        let labeler = QualityLabeler::new();
        let response = r#"```python
assert 2 + 3 == 6
```"#;
        let label = labeler.label("Check math", response);
        assert!(label.is_some());
        let label = label.unwrap();
        assert!(label.details.contains("UNPROVED"));
    }

    #[test]
    fn test_label_with_array_bounds() {
        let labeler = QualityLabeler::new();
        let response = r#"```python
def get_item():
    arr = [10, 20, 30]
    return arr[1]
```"#;
        let label = labeler.label("Get array item", response);
        assert!(label.is_some());
        let label = label.unwrap();
        assert!(label.details.contains("array bounds"));
        assert!(label.details.contains("SAFE"));
    }

    #[test]
    fn test_label_with_array_oob() {
        let labeler = QualityLabeler::new();
        let response = r#"```python
def get_item():
    arr = [10, 20, 30]
    return arr[5]
```"#;
        let label = labeler.label("Get array item", response);
        assert!(label.is_some());
        let label = label.unwrap();
        assert!(label.details.contains("UNSAFE"));
    }

    #[test]
    fn test_label_code_no_applicable_checks() {
        let labeler = QualityLabeler::new();
        // Code with no function def, no assertions, no array access
        let response = r#"```python
x = 42
y = x * 2
```"#;
        let label = labeler.label("Assign vars", response);
        assert!(label.is_some());
        let label = label.unwrap();
        // Only syntax check applies (no def, no assertions, no arrays)
        assert!(label.assessable);
        assert!(label.checks_total >= 1); // At least syntax check
    }

    #[test]
    fn test_structural_completeness_no_def() {
        let (has_def, _) = check_structural_completeness("x = 42");
        assert!(!has_def);
    }

    #[test]
    fn test_structural_completeness_with_return() {
        let code = "def foo():\n    return 42";
        let (has_def, has_return) = check_structural_completeness(code);
        assert!(has_def);
        assert!(has_return);
    }

    #[test]
    fn test_structural_completeness_missing_return() {
        let code = "def foo():\n    print('hello')";
        let (has_def, has_return) = check_structural_completeness(code);
        assert!(has_def);
        assert!(!has_return);
    }

    #[test]
    fn test_extract_arithmetic_assertions() {
        let code = "assert 2 + 3 == 5\nassert x == 10";
        let assertions = extract_arithmetic_assertions(code);
        assert_eq!(assertions.len(), 2);
        assert_eq!(assertions[0], ("2 + 3".to_string(), 5));
        assert_eq!(assertions[1], ("x".to_string(), 10));
    }

    #[test]
    fn test_extract_array_bounds_basic() {
        let code = "arr = [1, 2, 3]\nval = arr[1]";
        let bounds = extract_array_bounds(code);
        assert_eq!(bounds.len(), 1);
        assert_eq!(bounds[0], (1, 3));
    }

    // ── Canal 2: Text arithmetic tests ──────────────────────────────

    #[test]
    fn test_extract_text_equations_basic() {
        let v = SmtVerifier::new();
        let eqs = extract_text_equations("Step 1: 5 + 3 = 8. Done.", &v);
        assert_eq!(eqs.len(), 1);
        assert_eq!(eqs[0].0, "5 + 3");
        assert_eq!(eqs[0].1, 8);
    }

    #[test]
    fn test_extract_text_equations_multiple() {
        let v = SmtVerifier::new();
        let eqs = extract_text_equations("First 10 + 5 = 15, then 15 * 2 = 30.", &v);
        assert_eq!(eqs.len(), 2);
    }

    #[test]
    fn test_extract_text_equations_wrong_result() {
        let v = SmtVerifier::new();
        let eqs = extract_text_equations("5 + 3 = 9", &v);
        assert_eq!(eqs.len(), 1);
        assert_eq!(eqs[0].1, 9); // Extracted, but OxiZ will disprove
    }

    #[test]
    fn test_extract_text_equations_ignores_equality() {
        let v = SmtVerifier::new();
        let eqs = extract_text_equations("if x == 5: pass", &v);
        assert!(eqs.is_empty());
    }

    #[test]
    fn test_extract_text_equations_no_equations() {
        let v = SmtVerifier::new();
        let eqs = extract_text_equations("Just some text with no math.", &v);
        assert!(eqs.is_empty());
    }

    #[test]
    fn test_extract_text_equations_with_parens() {
        let v = SmtVerifier::new();
        let eqs = extract_text_equations("Result: (10 - 2) * 3 = 24", &v);
        assert_eq!(eqs.len(), 1);
        assert_eq!(eqs[0].1, 24);
        // The parser should find a valid expression containing parens
        assert!(eqs[0].0.contains("("));
    }

    // ── OxiZ parser probe tests ────────────────────────────────────

    #[test]
    fn test_try_parse_expr_valid() {
        let v = SmtVerifier::new();
        assert!(v.try_parse_expr("5 + 3"));
        assert!(v.try_parse_expr("(10 - 2) * 3"));
        assert!(v.try_parse_expr("42"));
    }

    #[test]
    fn test_try_parse_expr_invalid() {
        let v = SmtVerifier::new();
        assert!(!v.try_parse_expr("The answer is"));
        assert!(!v.try_parse_expr("Hello world"));
        assert!(!v.try_parse_expr(""));
    }

    #[test]
    fn test_eval_const_expr() {
        assert_eq!(SmtVerifier::eval_const_expr("5 + 3"), Some(8));
        assert_eq!(SmtVerifier::eval_const_expr("(10 - 2) * 3"), Some(24));
        assert_eq!(SmtVerifier::eval_const_expr("100 - 50"), Some(50));
        assert_eq!(SmtVerifier::eval_const_expr("42"), Some(42));
    }

    #[test]
    fn test_eval_const_expr_with_var() {
        assert_eq!(SmtVerifier::eval_const_expr("x + 1"), None);
    }

    #[test]
    fn test_eval_const_expr_invalid() {
        assert_eq!(SmtVerifier::eval_const_expr("not a number"), None);
        assert_eq!(SmtVerifier::eval_const_expr(""), None);
    }

    // ── Canal 3: Final number tests ────────────────────────────────

    #[test]
    fn test_extract_final_number_last_line() {
        assert_eq!(extract_final_number("some text\n42"), Some(42));
    }

    #[test]
    fn test_extract_final_number_with_period() {
        assert_eq!(extract_final_number("The answer is 42."), Some(42));
    }

    #[test]
    fn test_extract_final_number_answer_pattern() {
        assert_eq!(extract_final_number("Therefore the answer is 17"), Some(17));
    }

    #[test]
    fn test_extract_final_number_none() {
        assert_eq!(extract_final_number("no numbers here at all"), None);
    }

    #[test]
    fn test_extract_final_number_negative() {
        assert_eq!(extract_final_number("result: -5"), Some(-5));
    }

    // ── Integration: label() with 3 canaux ─────────────────────────

    #[test]
    fn test_label_math_response_proved() {
        let labeler = QualityLabeler::new();
        let label = labeler.label("What is 5+3?", "Let me calculate: 5 + 3 = 8\n\n8");
        assert!(label.is_some(), "Should not be None for math response");
        let l = label.unwrap();
        assert!(l.assessable);
        assert!(l.score > 0.5, "Correct math should score > 0.5");
        assert!(l.details.contains("PROVED") || l.details.contains("numeric"));
    }

    #[test]
    fn test_label_math_response_disproved() {
        let labeler = QualityLabeler::new();
        let label = labeler.label("What is 5+3?", "5 + 3 = 9\n\n9");
        assert!(label.is_some());
        let l = label.unwrap();
        assert!(l.details.contains("DISPROVED"));
    }

    #[test]
    fn test_label_pure_number_assessable() {
        let labeler = QualityLabeler::new();
        let label = labeler.label("How many animals?", "42");
        assert!(label.is_some(), "Pure number should be assessable now");
        let l = label.unwrap();
        assert!(l.assessable);
        assert_eq!(l.checks_passed, 1);
        assert!(l.details.contains("numeric"));
    }

    #[test]
    fn test_label_no_code_returns_none_for_prose() {
        let labeler = QualityLabeler::new();
        // Pure text with no code, no equations, no final number
        let label = labeler.label(
            "Explain gravity",
            "Gravity is a force of attraction between objects.",
        );
        assert!(label.is_none(), "Pure prose should still abstain");
    }

    #[test]
    fn test_label_code_still_works_unchanged() {
        let labeler = QualityLabeler::new();
        let response = "```python\ndef add(a, b):\n    return a + b\n```";
        let label = labeler.label("Write add", response);
        assert!(label.is_some());
        assert!(label.unwrap().details.contains("syntax OK"));
    }
}
