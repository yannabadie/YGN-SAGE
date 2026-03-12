//! Quality estimation for agent outputs.
//!
//! Ports `QualityEstimator` from `sage-python/src/sage/quality_estimator.py`
//! to Rust with PyO3 bindings.  Five signals are combined into a score in
//! [0.0, 1.0]:
//!
//! 1. Non-empty result   (+0.30)
//! 2. Length adequacy    (+0.00–0.20)
//! 3. Code presence      (+0.10–0.20)
//! 4. No error patterns  (+0.15)
//! 5. AVR convergence    (+0.05–0.15)

use pyo3::prelude::*;
use tracing::instrument;

// ── Task code-request keywords ───────────────────────────────────────────────

const TASK_CODE_KEYWORDS: &[&str] = &[
    "write",
    "code",
    "implement",
    "function",
    "class",
    "fix",
    "debug",
];

// ── Result code-presence markers ─────────────────────────────────────────────

const RESULT_CODE_MARKERS: &[&str] = &["def ", "class ", "function ", "import ", "```"];

// ── RustQualityEstimator ─────────────────────────────────────────────────────

/// Quality estimator for agent outputs.
///
/// Computes a score in [0.0, 1.0] from five cheap lexical signals — no LLM
/// call, no regex engine.  This is the Rust equivalent of the Python
/// `QualityEstimator.estimate()` static method.
#[pyclass]
#[derive(Debug, Default, Clone)]
pub struct RustQualityEstimator;

#[pymethods]
impl RustQualityEstimator {
    #[new]
    pub fn new() -> Self {
        Self
    }

    /// Estimate output quality.
    ///
    /// Parameters
    /// ----------
    /// task : str
    ///     The original task/prompt given to the agent.
    /// result : str
    ///     The agent's output to evaluate.
    /// latency_ms : float, optional
    ///     End-to-end latency in milliseconds (currently unused; reserved for
    ///     future signal weighting).
    /// had_errors : bool, optional
    ///     Whether the agent reported runtime errors during execution.
    /// avr_iterations : int, optional
    ///     Number of Act-Verify-Refine iterations consumed.
    ///
    /// Returns
    /// -------
    /// float
    ///     Quality score in [0.0, 1.0].
    #[pyo3(signature = (task, result, latency_ms=0.0, had_errors=false, avr_iterations=0))]
    #[instrument(skip(self), fields(task_len = task.len(), result_len = result.len()))]
    pub fn estimate(
        &self,
        task: &str,
        result: &str,
        latency_ms: f64,
        had_errors: bool,
        avr_iterations: u32,
    ) -> f32 {
        let _ = latency_ms; // reserved for future use

        // ── Signal 1: non-empty ──────────────────────────────────────────────
        if result.is_empty() || result.trim().is_empty() {
            return 0.0;
        }
        let mut score: f32 = 0.3;

        // ── Signal 2: length adequacy ────────────────────────────────────────
        let task_words = task.split_whitespace().count();
        let result_words = result.split_whitespace().count();
        let length_ratio = if task_words < 10 {
            (result_words as f32 / 20.0).min(1.0)
        } else {
            (result_words as f32 / 50.0).min(1.0)
        };
        score += length_ratio * 0.2;

        // ── Signal 3: code task + code presence ──────────────────────────────
        let task_lower = task.to_lowercase();
        let task_wants_code = TASK_CODE_KEYWORDS
            .iter()
            .any(|kw| task_lower.contains(kw));
        let result_has_code = RESULT_CODE_MARKERS
            .iter()
            .any(|marker| result.contains(marker));

        if task_wants_code && result_has_code {
            score += 0.2;
        } else if !task_wants_code {
            score += 0.1;
        }

        // ── Signal 4: no error patterns ──────────────────────────────────────
        // Check per line — avoids a regex dependency while preserving the
        // multiline semantics of the original Python `re.MULTILINE` flag.
        let result_lower = result.to_lowercase();
        let has_error_pattern = result_lower.lines().any(|line| {
            line.starts_with("error:")
                || line.starts_with("traceback")
                || line.starts_with("exception:")
                || line.contains("failed to")
                || line.contains("cannot ")
        });

        if !had_errors && !has_error_pattern {
            score += 0.15;
        }

        // ── Signal 5: AVR convergence ────────────────────────────────────────
        if avr_iterations > 0 {
            score += if avr_iterations <= 2 {
                0.15
            } else if avr_iterations <= 4 {
                0.10
            } else {
                0.05
            };
        }

        score.min(1.0)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn est() -> RustQualityEstimator {
        RustQualityEstimator::new()
    }

    #[test]
    fn test_empty_result_returns_zero() {
        assert_eq!(est().estimate("do something", "", 0.0, false, 0), 0.0);
    }

    #[test]
    fn test_whitespace_only_returns_zero() {
        assert_eq!(
            est().estimate("do something", "   \n\t  ", 0.0, false, 0),
            0.0
        );
    }

    #[test]
    fn test_non_empty_baseline() {
        // A short result ("ok") should yield at least the 0.3 base signal.
        let score = est().estimate("hello", "ok", 0.0, false, 0);
        assert!(
            score >= 0.3,
            "expected >= 0.3 for non-empty result, got {score}"
        );
    }

    #[test]
    fn test_code_task_with_code() {
        // "write a function" → task_wants_code=true, result contains "def foo" → code bonus.
        let task = "write a function that adds two numbers";
        let result = "def foo(a, b):\n    return a + b";
        let score = est().estimate(task, result, 0.0, false, 0);
        // Minimum expected: 0.3 (non-empty) + some length + 0.2 (code bonus) + 0.15 (no errors)
        // = at least 0.65 for a non-trivial result.
        assert!(
            score >= 0.6,
            "expected code task + code result to score >= 0.6, got {score}"
        );
    }

    #[test]
    fn test_error_in_result_reduces_score() {
        // A result starting with "Error:" should not receive Signal 4.
        let no_error_score = est().estimate("explain X", "X is a concept.", 0.0, false, 0);
        let error_score =
            est().estimate("explain X", "Error: cannot parse input.", 0.0, false, 0);
        assert!(
            error_score < no_error_score,
            "error result should score lower, got error={error_score} vs clean={no_error_score}"
        );
    }

    #[test]
    fn test_avr_convergence_fast() {
        // avr_iterations=1 (<=2) → +0.15
        let score_avr = est().estimate("task", "result text here", 0.0, false, 1);
        let score_no_avr = est().estimate("task", "result text here", 0.0, false, 0);
        let delta = score_avr - score_no_avr;
        assert!(
            (delta - 0.15).abs() < 1e-5,
            "expected +0.15 for avr_iterations=1, got delta={delta}"
        );
    }

    #[test]
    fn test_avr_convergence_slow() {
        // avr_iterations=6 (>4) → +0.05
        let score_avr = est().estimate("task", "result text here", 0.0, false, 6);
        let score_no_avr = est().estimate("task", "result text here", 0.0, false, 0);
        let delta = score_avr - score_no_avr;
        assert!(
            (delta - 0.05).abs() < 1e-5,
            "expected +0.05 for avr_iterations=6, got delta={delta}"
        );
    }

    #[test]
    fn test_max_score_capped_at_1() {
        // A perfect result with many words + code + avr should be capped at 1.0.
        let long_code_result = "def solution():\n    return 42\n".repeat(50);
        let score = est().estimate(
            "write a function implement class fix debug code",
            &long_code_result,
            0.0,
            false,
            1,
        );
        assert!(
            score <= 1.0,
            "score must not exceed 1.0, got {score}"
        );
    }
}
