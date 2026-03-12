//! CRAG-style keyword overlap relevance gate for memory injection.
//!
//! Ports `sage-python/src/sage/memory/relevance_gate.py` to Rust with PyO3 bindings.
//! Scores context vs task by keyword overlap; blocks irrelevant memory injection.

use pyo3::prelude::*;
use std::collections::HashSet;
use tracing::instrument;

/// Stop words to exclude from tokenization (mirrors the Python frozenset).
const STOP_WORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "shall",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "and", "but", "or", "not", "no",
    "nor", "this", "that", "these", "those", "it", "its", "i", "you", "he", "she", "we",
    "they", "me", "him", "her", "us", "them",
];

/// CRAG-style keyword overlap gate: scores context vs task by keyword overlap.
/// Threshold = 0.3 by default; blocks irrelevant memory injection.
#[pyclass]
pub struct RustRelevanceGate {
    #[pyo3(get)]
    pub threshold: f32,
    stop_words: HashSet<&'static str>,
}

#[pymethods]
impl RustRelevanceGate {
    #[new]
    #[pyo3(signature = (threshold=0.3))]
    pub fn new(threshold: f32) -> Self {
        let stop_words: HashSet<&'static str> = STOP_WORDS.iter().copied().collect();
        Self {
            threshold,
            stop_words,
        }
    }

    /// Score context relevance against a task by keyword overlap.
    ///
    /// Returns a float in [0.0, 1.0]: `|overlap| / |task_tokens|`.
    /// Returns 0.0 if either input is empty or tokenizes to nothing.
    #[instrument(skip(self))]
    pub fn score(&self, task: &str, context: &str) -> f32 {
        if task.is_empty() || context.is_empty() {
            return 0.0;
        }
        let task_tokens = self.tokenize(task);
        let ctx_tokens = self.tokenize(context);
        if task_tokens.is_empty() || ctx_tokens.is_empty() {
            return 0.0;
        }
        let overlap = task_tokens.intersection(&ctx_tokens).count();
        overlap as f32 / task_tokens.len() as f32
    }

    /// Return true if `context` is relevant to `task` (score >= threshold).
    ///
    /// Returns false immediately if `context` is blank.
    #[instrument(skip(self))]
    pub fn is_relevant(&self, task: &str, context: &str) -> bool {
        if context.is_empty() || context.trim().is_empty() {
            return false;
        }
        self.score(task, context) >= self.threshold
    }
}

impl RustRelevanceGate {
    /// Tokenize `text` into a set of lowercase keyword tokens.
    ///
    /// Mirrors the Python `_tokenize` method:
    /// - Lowercases the input
    /// - Extracts tokens matching `[a-z][a-z0-9_]+` (at least 2 chars after the first)
    /// - Filters out stop words and tokens shorter than 3 chars
    fn tokenize(&self, text: &str) -> HashSet<String> {
        let lower = text.to_lowercase();
        let chars: Vec<char> = lower.chars().collect();
        let mut tokens = HashSet::new();
        let len = chars.len();
        let mut i = 0;

        while i < len {
            // Must start with a-z
            if chars[i].is_ascii_lowercase() {
                let start = i;
                // Collect [a-z0-9_]* after the first char
                while i < len && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let token: String = chars[start..i].iter().collect();
                // Filter: length >= 3, not a stop word
                if token.len() >= 3 && !self.stop_words.contains(token.as_str()) {
                    tokens.insert(token);
                }
            } else {
                i += 1;
            }
        }

        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_identical_text() {
        let gate = RustRelevanceGate::new(0.3);
        let s = gate.score("hello world function", "hello world function");
        assert!(s > 0.5, "identical text should score > 0.5, got {s}");
    }

    #[test]
    fn test_score_no_overlap() {
        let gate = RustRelevanceGate::new(0.3);
        // "hello" is 5 chars, "goodbye" is 7 chars — both are single-word inputs
        // but neither is a stop word, so each tokenizes to one token with no overlap
        let s = gate.score("hello", "goodbye");
        assert_eq!(s, 0.0, "no shared tokens should score 0.0, got {s}");
    }

    #[test]
    fn test_score_empty() {
        let gate = RustRelevanceGate::new(0.3);
        assert_eq!(gate.score("", "anything"), 0.0);
        assert_eq!(gate.score("anything", ""), 0.0);
        assert_eq!(gate.score("", ""), 0.0);
    }

    #[test]
    fn test_is_relevant_above_threshold() {
        let gate = RustRelevanceGate::new(0.3);
        // "implement sorting algorithm" vs "sorting algorithm implementation"
        // task tokens: {implement, sorting, algorithm}
        // ctx tokens:  {sorting, algorithm, implementation}
        // overlap: {sorting, algorithm} → 2/3 ≈ 0.667 > 0.3
        let relevant = gate.is_relevant("implement sorting algorithm", "sorting algorithm implementation");
        assert!(relevant, "overlapping text should be relevant at threshold 0.3");
    }

    #[test]
    fn test_is_relevant_below_threshold() {
        let gate = RustRelevanceGate::new(0.9);
        // Partial overlap should fail the 0.9 threshold
        let relevant = gate.is_relevant("implement sorting algorithm", "sorting performance");
        assert!(!relevant, "partial overlap should not pass threshold 0.9");
    }

    #[test]
    fn test_stop_words_filtered() {
        let gate = RustRelevanceGate::new(0.3);
        // All tokens are stop words — tokenize returns empty set → score = 0.0
        let s = gate.score("the a is are", "the a is are");
        assert_eq!(s, 0.0, "all-stop-word inputs should score 0.0, got {s}");
    }

    #[test]
    fn test_is_relevant_blank_context() {
        let gate = RustRelevanceGate::new(0.3);
        assert!(!gate.is_relevant("some task", ""));
        assert!(!gate.is_relevant("some task", "   "));
    }

    #[test]
    fn test_tokenize_alphanumeric() {
        let gate = RustRelevanceGate::new(0.3);
        // "python3" should be tokenized as a single token
        let s = gate.score("python3 code", "python3 implementation");
        assert!(s > 0.0, "python3 should match across task and context");
    }
}
