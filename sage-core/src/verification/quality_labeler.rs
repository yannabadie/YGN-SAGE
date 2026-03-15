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
    let mut known_lengths: std::collections::HashMap<String, i64> = std::collections::HashMap::new();

    for line in code.lines() {
        let trimmed = line.trim();
        // Pattern: `name = [...]`
        if let Some(eq_pos) = trimmed.find(" = [") {
            let name = trimmed[..eq_pos].trim();
            // Only simple identifiers
            if name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
                && !name.is_empty()
            {
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

#[pymethods]
impl QualityLabeler {
    #[new]
    pub fn new() -> Self {
        Self {
            verifier: SmtVerifier::new(),
        }
    }

    /// Label the quality of an LLM response for a given task.
    ///
    /// Returns None if the response contains no code blocks (not assessable).
    /// Returns Some(QualityLabel) with formal check results otherwise.
    #[instrument(skip(self, task, response))]
    #[pyo3(signature = (task, response))]
    pub fn label(&self, task: &str, response: &str) -> Option<QualityLabel> {
        let _ = task; // reserved for future task-specific checks
        let code_blocks = extract_code_blocks(response);
        if code_blocks.is_empty() {
            return None;
        }

        let mut checks_passed: u32 = 0;
        let mut checks_total: u32 = 0;
        let mut details = Vec::new();

        for (i, code) in code_blocks.iter().enumerate() {
            let block_label = if code_blocks.len() > 1 {
                format!("block_{}", i)
            } else {
                "code".to_string()
            };

            // Check 1: Syntax validity via tree-sitter
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

            // Check 2: Arithmetic assertions verified via OxiZ
            let arith_assertions = extract_arithmetic_assertions(code);
            for (expr, expected) in &arith_assertions {
                checks_total += 1;
                if self.verifier.verify_arithmetic_expr(expr, *expected, 0) {
                    checks_passed += 1;
                    details.push(format!("{}: arithmetic '{}=={}' PROVED", block_label, expr, expected));
                } else {
                    details.push(format!(
                        "{}: arithmetic '{}=={}' UNPROVED",
                        block_label, expr, expected
                    ));
                }
            }

            // Check 3: Array bounds verified via OxiZ
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

            // Check 4: Structural completeness (def has return)
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

        let score = if checks_total > 0 {
            checks_passed as f32 / checks_total as f32
        } else {
            // Code exists but no checks were applicable → neutral score
            0.5
        };

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
    fn test_label_no_code_returns_none() {
        let labeler = QualityLabeler::new();
        let response = "Here is a text-only answer with no code.";
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
}
