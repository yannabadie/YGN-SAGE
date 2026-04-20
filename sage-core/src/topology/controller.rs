//! RustTopologyController — Rust port of topology_controller.py (Phase 2 of
//! the 2026-04-20 Rust-First plan). Scaffold only in this commit;
//! decision paths 1..6 are populated in commits 2.2–2.6.
//!
//! Python-facing API parity. Call sites in Python continue to go through
//! `TopologyController.evaluate_and_decide` which will delegate to the
//! Rust struct once per-path methods are populated. Until then this
//! controller is instantiated-but-dormant (imports cleanly, no behavior
//! change), so that the import guard in `sage.topology_controller` works
//! and downstream tests can start depending on its presence.
//!
//! Thresholds mirror Python `TopologyController.THETA_*` constants —
//! calibrated initial values, subject to ablation per CLAUDE.md §2.
//! Safety limits (`MAX_*`) are engineering guards (bypass-patterns.md
//! calibration table).

use pyo3::prelude::*;
use regex::Regex;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Sentinel prefix emitted by agent_loop_memory when the loop exhausts
/// its step budget without producing a final answer. Mirrors Python
/// `topology_controller._SENTINEL_PREFIX` — must stay in sync with
/// `phases/learn.py::EMPTY_STEP_SENTINEL`.
pub(crate) const SENTINEL_PREFIX: &str = "[sage: agent exited after";

/// Error-output detector — mirrors Python `_ERROR_OUTPUT` regex.
/// Lazy-compiled once per process.
fn error_output_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)^\s*(error|exception|traceback|timeout|failed)\b|\b(traceback|stack trace|timed out|no output|failed with)\b",
        )
        .expect("SENTINEL_PREFIX / _ERROR_OUTPUT regex must compile")
    })
}

/// Mirror of Python `TopologyController._is_empty_or_error` static method.
/// Public so Python tests can round-trip against the same detector.
pub fn is_empty_or_error(result: &str) -> bool {
    let stripped = result.trim();
    if stripped.is_empty() {
        return true;
    }
    if stripped.starts_with(SENTINEL_PREFIX) {
        return true;
    }
    error_output_re().is_match(stripped)
}

// Thresholds — calibrated initial values, subject to ablation.
pub const THETA_GOOD: f32 = 0.7;
pub const THETA_CRITICAL: f32 = 0.3;
pub const THETA_CONSISTENCY: f32 = 0.5;
pub const THETA_PRUNE: f32 = 0.2;

// Safety limits — engineering guards.
pub const MAX_RETRIES: u32 = 2;
pub const MAX_REROUTES: u32 = 1;
pub const MAX_GATE_TURNS: u32 = 2;
pub const MAX_SPAWNS: u32 = 3;

/// Rust mirror of Python `AdaptationDecision` dataclass.
#[pyclass(get_all)]
#[derive(Clone, Debug)]
pub struct RustAdaptationDecision {
    /// One of: "continue", "upgrade_model", "prune_node",
    /// "reroute_topology", "spawn_subagent", "open_gate".
    pub action: String,
    pub target_node: Option<usize>,
    pub reason: String,
    pub new_model_id: Option<String>,
    pub invariant_feedback: Option<String>,
    pub gate_source: Option<usize>,
    pub gate_target: Option<usize>,
}

impl RustAdaptationDecision {
    pub fn continue_at(node_idx: Option<usize>, reason: impl Into<String>) -> Self {
        Self {
            action: "continue".into(),
            target_node: node_idx,
            reason: reason.into(),
            new_model_id: None,
            invariant_feedback: None,
            gate_source: None,
            gate_target: None,
        }
    }

    pub fn reroute(node_idx: usize, reason: impl Into<String>) -> Self {
        Self {
            action: "reroute_topology".into(),
            target_node: Some(node_idx),
            reason: reason.into(),
            new_model_id: None,
            invariant_feedback: None,
            gate_source: None,
            gate_target: None,
        }
    }
}

#[pymethods]
impl RustAdaptationDecision {
    #[new]
    #[pyo3(signature = (
        action,
        target_node = None,
        reason = String::new(),
        new_model_id = None,
        invariant_feedback = None,
        gate_source = None,
        gate_target = None,
    ))]
    fn py_new(
        action: String,
        target_node: Option<usize>,
        reason: String,
        new_model_id: Option<String>,
        invariant_feedback: Option<String>,
        gate_source: Option<usize>,
        gate_target: Option<usize>,
    ) -> Self {
        Self {
            action,
            target_node,
            reason,
            new_model_id,
            invariant_feedback,
            gate_source,
            gate_target,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RustAdaptationDecision(action={:?}, target_node={:?}, reason={:?})",
            self.action, self.target_node, self.reason,
        )
    }
}

/// Rust port of Python `TopologyController` runtime state.
/// Scaffold-only for commit 2.1 — `evaluate_and_decide` stub returns None,
/// signalling the Python legacy path to handle the decision. Per-path
/// methods (`check_empty_error_reroute`, etc.) populated in 2.2–2.6.
#[pyclass]
pub struct RustTopologyController {
    reroute_count: u32,
    spawn_count: u32,
    node_retries: HashMap<usize, u32>,
    abstain_count: u32,
    node_qualities: HashMap<usize, f32>,
    gate_loops: HashMap<usize, u32>,
}

impl Default for RustTopologyController {
    fn default() -> Self {
        Self::new_inner()
    }
}

impl RustTopologyController {
    fn new_inner() -> Self {
        Self {
            reroute_count: 0,
            spawn_count: 0,
            node_retries: HashMap::new(),
            abstain_count: 0,
            node_qualities: HashMap::new(),
            gate_loops: HashMap::new(),
        }
    }
}

#[pymethods]
impl RustTopologyController {
    #[new]
    fn new() -> Self {
        Self::new_inner()
    }

    /// Scaffold stub — returns None to signal Python fallback. Populated
    /// path-by-path in 2.2..2.6 of the Rust-First plan.
    #[pyo3(signature = (node_idx, result, task))]
    fn evaluate_and_decide(
        &mut self,
        node_idx: usize,
        result: String,
        task: String,
    ) -> Option<RustAdaptationDecision> {
        let _ = (node_idx, result, task);
        None
    }

    /// Plan 2.2 — port of Python path 1 (`topology_controller.py:134-149`).
    /// Empty / sentinel / error-pattern output triggers a reroute_topology
    /// decision, up to MAX_REROUTES. When the budget is exhausted, return
    /// a continue decision citing the exhaustion. Returns None when the
    /// output passes the empty/error check — the caller (Python delegate
    /// for now) falls through to quality-cascade evaluation.
    #[pyo3(signature = (result, node_idx))]
    fn check_empty_error_reroute(
        &mut self,
        result: &str,
        node_idx: usize,
    ) -> Option<RustAdaptationDecision> {
        if !is_empty_or_error(result) {
            return None;
        }
        if self.reroute_count >= MAX_REROUTES {
            return Some(RustAdaptationDecision::continue_at(
                Some(node_idx),
                "reroute budget exhausted",
            ));
        }
        self.reroute_count += 1;
        let reason = if result.trim().is_empty() {
            "empty output"
        } else {
            "error-like output"
        };
        Some(RustAdaptationDecision::reroute(node_idx, reason))
    }

    #[getter]
    fn reroute_count(&self) -> u32 {
        self.reroute_count
    }

    #[getter]
    fn spawn_count(&self) -> u32 {
        self.spawn_count
    }

    #[getter]
    fn abstain_count(&self) -> u32 {
        self.abstain_count
    }

    /// Diagnostic view mirroring Python `quality_stats()`. Expose state
    /// early so tests and the Python delegate can both observe it.
    fn quality_stats(&self, py: Python<'_>) -> PyObject {
        let dict = pyo3::types::PyDict::new(py);
        let _ = dict.set_item("abstain_count", self.abstain_count);
        let _ = dict.set_item("reroute_count", self.reroute_count);
        let _ = dict.set_item("spawn_count", self.spawn_count);
        let node_qualities = pyo3::types::PyDict::new(py);
        for (k, v) in &self.node_qualities {
            let _ = node_qualities.set_item(*k, *v);
        }
        let _ = dict.set_item("node_qualities", &node_qualities);
        dict.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_has_expected_initial_state() {
        let ctrl = RustTopologyController::new();
        assert_eq!(ctrl.reroute_count, 0);
        assert_eq!(ctrl.spawn_count, 0);
        assert_eq!(ctrl.abstain_count, 0);
        assert!(ctrl.node_retries.is_empty());
        assert!(ctrl.node_qualities.is_empty());
        assert!(ctrl.gate_loops.is_empty());
    }

    #[test]
    fn adaptation_decision_continue_factory_sets_action() {
        let d = RustAdaptationDecision::continue_at(Some(3), "ok");
        assert_eq!(d.action, "continue");
        assert_eq!(d.target_node, Some(3));
        assert_eq!(d.reason, "ok");
    }

    #[test]
    fn adaptation_decision_reroute_factory_sets_action() {
        let d = RustAdaptationDecision::reroute(2, "empty output");
        assert_eq!(d.action, "reroute_topology");
        assert_eq!(d.target_node, Some(2));
        assert_eq!(d.reason, "empty output");
    }

    #[test]
    fn is_empty_or_error_detects_blanks() {
        assert!(is_empty_or_error(""));
        assert!(is_empty_or_error("   "));
        assert!(is_empty_or_error("\n\t"));
    }

    #[test]
    fn is_empty_or_error_detects_sentinel() {
        assert!(is_empty_or_error(
            "[sage: agent exited after 20 steps, no final content]"
        ));
    }

    #[test]
    fn is_empty_or_error_detects_error_patterns() {
        assert!(is_empty_or_error("Error: something went wrong"));
        assert!(is_empty_or_error("Exception during call"));
        assert!(is_empty_or_error("Traceback (most recent call last):"));
        assert!(is_empty_or_error("The job timed out after 60s"));
        assert!(is_empty_or_error("operation failed with code 42"));
        assert!(is_empty_or_error("no output from tool"));
    }

    #[test]
    fn is_empty_or_error_passes_normal_content() {
        assert!(!is_empty_or_error("Here is the answer: 42"));
        assert!(!is_empty_or_error("def solve(): return 42"));
        assert!(!is_empty_or_error("The quick brown fox jumped"));
    }

    #[test]
    fn check_empty_error_reroute_reroutes_empty_then_exhausts() {
        let mut c = RustTopologyController::new();
        let d = c.check_empty_error_reroute("", 2).unwrap();
        assert_eq!(d.action, "reroute_topology");
        assert_eq!(d.target_node, Some(2));
        assert_eq!(d.reason, "empty output");
        assert_eq!(c.reroute_count, 1);

        // Second empty: budget exhausted (MAX_REROUTES=1) → continue with reason
        let d2 = c.check_empty_error_reroute("", 2).unwrap();
        assert_eq!(d2.action, "continue");
        assert_eq!(d2.reason, "reroute budget exhausted");
        assert_eq!(c.reroute_count, 1, "must NOT increment past the budget");
    }

    #[test]
    fn check_empty_error_reroute_tags_error_reason() {
        let mut c = RustTopologyController::new();
        let d = c.check_empty_error_reroute("Error: boom", 0).unwrap();
        assert_eq!(d.action, "reroute_topology");
        assert_eq!(d.reason, "error-like output");
    }

    #[test]
    fn check_empty_error_reroute_passes_through_normal() {
        let mut c = RustTopologyController::new();
        assert!(c
            .check_empty_error_reroute("normal answer text here", 0)
            .is_none());
        assert_eq!(c.reroute_count, 0);
    }

    #[test]
    fn thresholds_mirror_python_constants() {
        assert_eq!(THETA_GOOD, 0.7);
        assert_eq!(THETA_CRITICAL, 0.3);
        assert_eq!(THETA_CONSISTENCY, 0.5);
        assert_eq!(THETA_PRUNE, 0.2);
        assert_eq!(MAX_RETRIES, 2);
        assert_eq!(MAX_REROUTES, 1);
        assert_eq!(MAX_GATE_TURNS, 2);
        assert_eq!(MAX_SPAWNS, 3);
    }
}
