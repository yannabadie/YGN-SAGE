//! Stage 0 — Structural feature extraction for adaptive routing.
//!
//! Extracts cheap lexical/structural signals from a task string in <1ms.
//! These features feed into the Stage 1 ONNX classifier (or can be used
//! directly by Python for rule-based routing).

use pyo3::prelude::*;

// ── Keyword groups ──────────────────────────────────────────────────────────

const ALGO_KEYWORDS: &[&str] = &[
    "implement",
    "build",
    "algorithm",
    "optimize",
    "compiler",
    "concurrent",
    "distributed",
    "consensus",
    "lock-free",
];

const CODE_KEYWORDS: &[&str] = &[
    "write",
    "create",
    "code",
    "function",
    "class",
    "method",
    "parse",
    "regex",
    "query",
    "endpoint",
    "decorator",
];

const DEBUG_KEYWORDS: &[&str] = &[
    "debug",
    "fix",
    "error",
    "crash",
    "bug",
    "race condition",
    "deadlock",
    "oom",
    "memory leak",
];

const DESIGN_KEYWORDS: &[&str] = &[
    "design",
    "architect",
    "refactor",
    "schema",
    "system",
    "prove",
    "induction",
    "complexity",
];

const UNCERTAINTY_KEYWORDS: &[&str] = &[
    "maybe",
    "possibly",
    "explore",
    "investigate",
    "intermittent",
    "sometimes",
    "random",
    "flaky",
];

const TOOL_KEYWORDS: &[&str] = &[
    "file",
    "search",
    "run",
    "execute",
    "compile",
    "test",
    "deploy",
    "download",
    "upload",
];

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Check whether `text` contains any keyword from `keywords`, using
/// case-insensitive matching. Multi-word keywords (e.g. "race condition")
/// are matched as substrings.
fn has_any_keyword(text: &str, keywords: &[&str]) -> bool {
    keywords.iter().any(|kw| text.contains(kw))
}

/// Count how many distinct keywords from `keywords` appear in `text`.
fn count_keywords(text: &str, keywords: &[&str]) -> usize {
    keywords.iter().filter(|kw| text.contains(**kw)).count()
}

// ── StructuralFeatures ──────────────────────────────────────────────────────

/// Cheap structural features extracted from a task string.
///
/// Used by Stage 0 of the adaptive router to produce a feature vector
/// without any LLM call. All fields are deterministic and reproducible.
#[pyclass]
#[derive(Debug, Clone)]
pub struct StructuralFeatures {
    /// Number of whitespace-delimited words in the task.
    #[pyo3(get)]
    pub word_count: usize,

    /// Whether the task contains a fenced code block (``` or ~~~).
    #[pyo3(get)]
    pub has_code_block: bool,

    /// Whether the task contains a question mark.
    #[pyo3(get)]
    pub has_question_mark: bool,

    /// Complexity score derived from keyword groups (0.0–1.0, clamped).
    #[pyo3(get)]
    pub keyword_complexity: f32,

    /// Uncertainty score derived from uncertainty keywords (0.0–1.0).
    #[pyo3(get)]
    pub keyword_uncertainty: f32,

    /// Whether tool-use keywords are present.
    #[pyo3(get)]
    pub tool_required: bool,
}

#[pymethods]
impl StructuralFeatures {
    /// Extract structural features from a task string.
    ///
    /// This is the Python-visible static constructor.
    #[staticmethod]
    pub fn extract(task: &str) -> Self {
        Self::extract_from(task)
    }

    fn __repr__(&self) -> String {
        format!(
            "StructuralFeatures(words={}, code_block={}, question={}, complexity={:.3}, uncertainty={:.3}, tool={})",
            self.word_count,
            self.has_code_block,
            self.has_question_mark,
            self.keyword_complexity,
            self.keyword_uncertainty,
            self.tool_required,
        )
    }
}

impl StructuralFeatures {
    /// Pure-Rust extraction (no Python/GIL needed).
    pub fn extract_from(task: &str) -> Self {
        let lower = task.to_lowercase();

        let word_count = task.split_whitespace().count();
        let has_code_block = lower.contains("```") || lower.contains("~~~");
        let has_question_mark = task.contains('?');
        let tool_required = has_any_keyword(&lower, TOOL_KEYWORDS);

        // ── Uncertainty score ───────────────────────────────────────────
        let uncertainty_hits = count_keywords(&lower, UNCERTAINTY_KEYWORDS);
        let keyword_uncertainty = (uncertainty_hits as f32 * 0.25).min(1.0);

        // ── Complexity score ────────────────────────────────────────────
        // Base complexity for any task.
        let mut complexity: f32 = 0.2;

        // Keyword group boosts (elif-style priority: algo > debug > design > code).
        if has_any_keyword(&lower, ALGO_KEYWORDS) {
            complexity += 0.35;
        } else if has_any_keyword(&lower, DEBUG_KEYWORDS) {
            complexity += 0.30;
        } else if has_any_keyword(&lower, DESIGN_KEYWORDS) {
            complexity += 0.20;
        } else if has_any_keyword(&lower, CODE_KEYWORDS) {
            complexity += 0.15;
        }

        // Code block presence adds complexity.
        if has_code_block {
            complexity += 0.1;
        }

        // Word count scaling: longer tasks are more complex (discrete thresholds).
        if word_count > 100 {
            complexity += 0.15;
        } else if word_count > 50 {
            complexity += 0.1;
        } else if word_count > 20 {
            complexity += 0.05;
        }

        // Clamp to [0, 1].
        let keyword_complexity = complexity.clamp(0.0, 1.0);

        Self {
            word_count,
            has_code_block,
            has_question_mark,
            keyword_complexity,
            keyword_uncertainty,
            tool_required,
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_factual() {
        let f = StructuralFeatures::extract_from("What is the capital of France?");
        assert!(f.has_question_mark);
        assert!(!f.has_code_block);
        assert!(!f.tool_required);
        // Simple factual question → base complexity only (0.2).
        assert!(
            f.keyword_complexity < 0.35,
            "expected low complexity for factual question, got {}",
            f.keyword_complexity
        );
    }

    #[test]
    fn code_task() {
        let f = StructuralFeatures::extract_from("Write a Python function to sort a list");
        assert!(!f.has_code_block);
        assert!(!f.has_question_mark);
        // "write" + "function" → CODE_KEYWORDS hit → base 0.2 + 0.15 = 0.35.
        assert!(
            (f.keyword_complexity - 0.35).abs() < 0.01,
            "expected ~0.35 for code task, got {}",
            f.keyword_complexity
        );
    }

    #[test]
    fn complex_debug() {
        let f = StructuralFeatures::extract_from(
            "Debug the race condition in the concurrent queue implementation",
        );
        // "debug" → DEBUG_KEYWORDS, "race condition" → also DEBUG, "concurrent" → ALGO.
        // ALGO takes priority (elif): base 0.2 + 0.35 = 0.55.
        assert!(
            f.keyword_complexity >= 0.50,
            "expected high complexity for debug+algo task, got {}",
            f.keyword_complexity
        );
    }

    #[test]
    fn code_block_detection() {
        let task = "Fix this code:\n```python\ndef foo():\n    pass\n```";
        let f = StructuralFeatures::extract_from(task);
        assert!(f.has_code_block);
        // "fix" → DEBUG (0.3) + code_block (0.1) → 0.2 + 0.3 + 0.1 = 0.6.
        assert!(
            f.keyword_complexity >= 0.55,
            "expected code_block to boost complexity, got {}",
            f.keyword_complexity
        );
    }

    #[test]
    fn tool_required_detection() {
        let f = StructuralFeatures::extract_from("Search the web for recent Rust async tutorials");
        assert!(f.tool_required, "expected tool_required for 'search' keyword");
    }

    #[test]
    fn long_task_scaling() {
        // Build a task with >120 words — should get word-count scaling bonus.
        let padding = "word ".repeat(130);
        let task = format!("Implement an algorithm that {}", padding);
        let f = StructuralFeatures::extract_from(&task);
        assert!(f.word_count > 120);
        // ALGO hit (0.35) + word scaling (+0.15 for >100 words) → 0.2 + 0.35 + 0.15 = 0.70.
        assert!(
            (f.keyword_complexity - 0.70).abs() < 0.01,
            "expected 0.70 for algo+long task, got {} (words={})",
            f.keyword_complexity,
            f.word_count
        );
    }
}
