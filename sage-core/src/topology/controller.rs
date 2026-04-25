//! RustTopologyController — Rust primitives that back the Python
//! `TopologyController` (Phase 2 of the 2026-04-20 Rust-First plan).
//!
//! This struct owns the runtime counters (`reroute_count`, `spawn_count`,
//! `node_retries`, `node_qualities`, `gate_loops`, `abstain_count`) and
//! exposes per-path decision primitives (`check_empty_error_reroute`,
//! `check_quality_cascade`, `is_in_gate_band`, `check_parallel_inconsistency`,
//! `check_low_importance_prune`, `should_trigger_emergent_spawn`). Python
//! orchestrates which primitive to invoke per agent-loop step because the
//! cascade depends on Python-resident subsystems (embedder, SmtVerifier,
//! topology graph, gate management, upgrade-model resolution).
//!
//! The top-level `evaluate_and_decide` entry stayed on Python per
//! 2026-04-23 B8 decision (advisor + gpt-5.4-high-reasoning converged):
//! a Rust wrapper would be a thin PyO3 shim back into the Python
//! subsystems with no measurable win, and the semantic-drift surface
//! outweighs the "Rust-primary controller" branding. See ADR-012.
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

/// Rust-side state + per-path decision primitives for Python
/// `TopologyController`. Python owns orchestration (which primitive to
/// invoke per step), Rust owns state + threshold math. See module-level
/// doc comment + ADR-012 for the split rationale.
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

    /// Plan 2.3 — port of Python path 2 (quality cascade) + the debate-gate
    /// threshold check (scope-clipped per plan; `_open_gate` helper stays
    /// Python for now because it walks the topology graph via predecessors).
    /// Returns:
    ///   - Some(continue) when quality >= THETA_GOOD
    ///   - Some(upgrade_model) (partial — caller fills new_model_id +
    ///     invariant_feedback via Python resolver) when quality <
    ///     THETA_CRITICAL AND node retries < retry_limit; increments
    ///     `node_retries[node_idx]` as a side effect.
    ///   - None when quality is in the critical band [THETA_CRITICAL,
    ///     THETA_GOOD) — caller falls through to debate-gate logic. Also
    ///     None when quality < THETA_CRITICAL but retry budget exhausted
    ///     (caller continues cascade to parallel inconsistency / prune /
    ///     spawn).
    ///
    /// `retry_limit` is passed in by the caller because it depends on the
    /// node's `max_retries` attribute — reading that from Python here
    /// would couple Rust to the Python topology-graph shape. Keep it in
    /// Python until 2.6 ports `_max_retries_for_node`.
    #[pyo3(signature = (quality, node_idx, retry_limit))]
    fn check_quality_cascade(
        &mut self,
        quality: f32,
        node_idx: usize,
        retry_limit: u32,
    ) -> Option<RustAdaptationDecision> {
        self.node_qualities.insert(node_idx, quality);
        if quality >= THETA_GOOD {
            return Some(RustAdaptationDecision::continue_at(Some(node_idx), ""));
        }
        if quality < THETA_CRITICAL {
            let retries = *self.node_retries.get(&node_idx).unwrap_or(&0);
            if retries < retry_limit {
                self.node_retries.insert(node_idx, retries + 1);
                return Some(RustAdaptationDecision {
                    action: "upgrade_model".into(),
                    target_node: Some(node_idx),
                    reason: format!("quality={:.2} < {}", quality, THETA_CRITICAL),
                    new_model_id: None, // filled by Python _resolve_upgrade_model
                    invariant_feedback: None, // filled by Python _get_invariant_feedback
                    gate_source: None,
                    gate_target: None,
                });
            }
            // retry budget exhausted → caller continues cascade
            return None;
        }
        // In the critical band [THETA_CRITICAL, THETA_GOOD) — caller will
        // try debate gate, then fall through if gate returns None.
        None
    }

    /// Plan 2.3 helper — "is this quality in the debate-gate band?"
    /// Keeps the threshold check in Rust so Python delegation from 2.6
    /// doesn't need to hard-code the constants; `_open_gate` itself
    /// stays Python for now.
    #[pyo3(signature = (quality))]
    fn is_in_gate_band(&self, quality: f32) -> bool {
        (THETA_CRITICAL..THETA_GOOD).contains(&quality)
    }

    /// Plan 2.4 — port of Python path 4 (parallel inconsistency reroute,
    /// `topology_controller.py:202-211`). Consistency scoring stays in
    /// Python (it requires the embedder) — Rust takes the pre-computed
    /// `consistency` float and applies the threshold + state machine.
    /// Same pattern as `check_quality_cascade` taking pre-computed quality.
    /// Does nothing for debate topologies (multi-turn disagreement is
    /// part of the intended flow, not a reroute signal) or when the
    /// reroute budget is exhausted.
    #[pyo3(signature = (node_idx, consistency, is_debate))]
    fn check_parallel_inconsistency(
        &mut self,
        node_idx: usize,
        consistency: f32,
        is_debate: bool,
    ) -> Option<RustAdaptationDecision> {
        if is_debate || self.reroute_count >= MAX_REROUTES {
            return None;
        }
        if consistency >= THETA_CONSISTENCY {
            return None;
        }
        self.reroute_count += 1;
        Some(RustAdaptationDecision {
            action: "reroute_topology".into(),
            target_node: Some(node_idx),
            reason: format!("consistency={:.2} < {}", consistency, THETA_CONSISTENCY),
            new_model_id: None,
            invariant_feedback: None,
            gate_source: None,
            gate_target: None,
        })
    }

    /// Plan 2.4 — port of Python path 5 (importance prune,
    /// `topology_controller.py:213-222`). Importance scoring stays in
    /// Python (embedder-backed); Rust threshold-checks. Quality must be
    /// KNOWN (not abstained) — an abstained quality means the estimator
    /// had no signal, pruning would be premature. Caller passes that
    /// bool; Rust doesn't track abstain_count at this scope. Debate
    /// topologies also suppress prune (the multi-node structure is
    /// deliberate).
    #[pyo3(signature = (node_idx, importance, is_debate, quality_is_known))]
    fn check_importance_prune(
        &self,
        node_idx: usize,
        importance: f32,
        is_debate: bool,
        quality_is_known: bool,
    ) -> Option<RustAdaptationDecision> {
        if is_debate || !quality_is_known {
            return None;
        }
        if importance >= THETA_PRUNE {
            return None;
        }
        Some(RustAdaptationDecision {
            action: "prune_node".into(),
            target_node: Some(node_idx),
            reason: format!("importance={:.2} < {}", importance, THETA_PRUNE),
            new_model_id: None,
            invariant_feedback: None,
            gate_source: None,
            gate_target: None,
        })
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

    /// Plan 2.6 helpers — setters so Python callers (tests, legacy
    /// state-injection patterns) can seed Rust state before invoking
    /// a decision path. Intentionally not made `#[setter]` so misuse
    /// is grep-able as an explicit call site rather than an attribute
    /// assignment.
    fn set_reroute_count(&mut self, value: u32) {
        self.reroute_count = value;
    }

    fn set_spawn_count(&mut self, value: u32) {
        self.spawn_count = value;
    }

    fn set_node_retries(&mut self, node_idx: usize, value: u32) {
        self.node_retries.insert(node_idx, value);
    }

    /// H11 audit (2026-04-20): expose per-node retry counter for
    /// cross-path equivalence tests and any caller that needs to
    /// observe the Rust-side state independently of quality_stats.
    fn get_node_retries(&self, node_idx: usize) -> u32 {
        *self.node_retries.get(&node_idx).unwrap_or(&0)
    }

    /// Dict-shaped state: expose as Vec<(idx, count)> since PyO3 dict
    /// conversion would need the GIL token. Python caller wraps in dict().
    fn node_retries_view(&self) -> Vec<(usize, u32)> {
        self.node_retries.iter().map(|(k, v)| (*k, *v)).collect()
    }

    fn node_qualities_view(&self) -> Vec<(usize, f32)> {
        self.node_qualities.iter().map(|(k, v)| (*k, *v)).collect()
    }

    /// Budget-only check: can another emergent spawn happen?
    /// `_node_idx` is accepted for forward compat (future per-node budgets)
    /// but ignored under the current global MAX_SPAWNS policy.
    fn should_trigger_emergent_spawn(&self, _node_idx: usize) -> bool {
        self.spawn_count < MAX_SPAWNS
    }

    /// Increment the abstain counter. Used by Python `_compute_quality`
    /// when QualityEstimator falls back to heuristic (signals an abstain).
    /// No cap — abstain is diagnostic, not budget-limiting.
    fn record_abstain(&mut self) {
        self.abstain_count += 1;
    }

    /// Increment the spawn counter under the MAX_SPAWNS gate.
    /// Returns PyValueError if already at cap (defensive — caller should
    /// check `should_trigger_emergent_spawn` first).
    fn record_emergent_spawn(&mut self, _node_idx: usize) -> PyResult<()> {
        if self.spawn_count >= MAX_SPAWNS {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "spawn budget exhausted ({}/{} reached)",
                self.spawn_count, MAX_SPAWNS
            )));
        }
        self.spawn_count += 1;
        Ok(())
    }

    /// Test-only entry point — set all counters and retry dict in one call.
    /// Used to replace the Python `__setattr__` mirror that let legacy tests
    /// do `controller._reroute_count = N` directly. Validates implausible
    /// values (> MAX_REROUTES + 10) to catch typos.
    fn seed_state_for_legacy_tests(
        &mut self,
        reroute_count: u32,
        spawn_count: u32,
        node_retries: Vec<(usize, u32)>,
        abstain_count: u32,
    ) -> PyResult<()> {
        if reroute_count > MAX_REROUTES + 10 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "reroute_count={} implausible (MAX_REROUTES={})",
                reroute_count, MAX_REROUTES
            )));
        }
        self.reroute_count = reroute_count;
        self.spawn_count = spawn_count;
        self.node_retries = node_retries.into_iter().collect();
        self.abstain_count = abstain_count;
        Ok(())
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
        let node_retries = pyo3::types::PyDict::new(py);
        for (k, v) in &self.node_retries {
            let _ = node_retries.set_item(*k, *v);
        }
        let _ = dict.set_item("node_retries", &node_retries);
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
    fn quality_cascade_good_returns_continue() {
        let mut c = RustTopologyController::new();
        let d = c.check_quality_cascade(0.85, 0, 2).unwrap();
        assert_eq!(d.action, "continue");
        assert_eq!(d.target_node, Some(0));
        assert_eq!(c.node_qualities.get(&0), Some(&0.85));
    }

    #[test]
    fn quality_cascade_at_good_threshold_is_inclusive() {
        let mut c = RustTopologyController::new();
        let d = c.check_quality_cascade(THETA_GOOD, 0, 2).unwrap();
        assert_eq!(d.action, "continue");
    }

    #[test]
    fn quality_cascade_critical_upgrades_and_increments_retries() {
        let mut c = RustTopologyController::new();
        let d = c.check_quality_cascade(0.1, 3, 2).unwrap();
        assert_eq!(d.action, "upgrade_model");
        assert_eq!(d.target_node, Some(3));
        assert!(d.reason.starts_with("quality="));
        assert!(d.reason.contains("< 0.3"));
        assert_eq!(c.node_retries.get(&3), Some(&1));
    }

    #[test]
    fn quality_cascade_critical_retry_exhaustion_returns_none() {
        let mut c = RustTopologyController::new();
        // Two upgrades allowed (MAX_RETRIES via retry_limit param)
        c.check_quality_cascade(0.1, 3, 2).unwrap();
        c.check_quality_cascade(0.1, 3, 2).unwrap();
        // Third call — budget exhausted, falls through
        assert!(c.check_quality_cascade(0.1, 3, 2).is_none());
        // Retries capped at the budget (no over-increment)
        assert_eq!(c.node_retries.get(&3), Some(&2));
    }

    #[test]
    fn quality_cascade_middle_band_returns_none_for_gate() {
        let mut c = RustTopologyController::new();
        // Between THETA_CRITICAL=0.3 and THETA_GOOD=0.7 → None so the
        // Python debate-gate helper can decide
        assert!(c.check_quality_cascade(0.5, 0, 2).is_none());
        // Per-node quality tracking still happens
        assert_eq!(c.node_qualities.get(&0), Some(&0.5));
    }

    #[test]
    fn is_in_gate_band_detects_critical_range() {
        let c = RustTopologyController::new();
        assert!(c.is_in_gate_band(0.3)); // lower inclusive
        assert!(c.is_in_gate_band(0.5));
        assert!(c.is_in_gate_band(0.699));
        assert!(!c.is_in_gate_band(0.7)); // upper exclusive
        assert!(!c.is_in_gate_band(0.29)); // below
        assert!(!c.is_in_gate_band(0.9));
    }

    #[test]
    fn parallel_inconsistency_reroutes_below_threshold() {
        let mut c = RustTopologyController::new();
        let d = c.check_parallel_inconsistency(1, 0.3, false).unwrap();
        assert_eq!(d.action, "reroute_topology");
        assert_eq!(d.target_node, Some(1));
        assert!(d.reason.starts_with("consistency="));
        assert_eq!(c.reroute_count, 1);
    }

    #[test]
    fn parallel_inconsistency_above_threshold_is_none() {
        let mut c = RustTopologyController::new();
        assert!(c
            .check_parallel_inconsistency(1, THETA_CONSISTENCY, false)
            .is_none());
        assert!(c.check_parallel_inconsistency(1, 0.8, false).is_none());
        assert_eq!(c.reroute_count, 0);
    }

    #[test]
    fn parallel_inconsistency_debate_topology_skips_reroute() {
        let mut c = RustTopologyController::new();
        // Would normally reroute, but debate topology suppresses
        assert!(c.check_parallel_inconsistency(1, 0.1, true).is_none());
        assert_eq!(c.reroute_count, 0);
    }

    #[test]
    fn parallel_inconsistency_respects_reroute_budget() {
        let mut c = RustTopologyController::new();
        c.check_parallel_inconsistency(0, 0.2, false).unwrap();
        assert_eq!(c.reroute_count, 1);
        // Budget exhausted (MAX_REROUTES=1) — next call returns None
        assert!(c.check_parallel_inconsistency(0, 0.1, false).is_none());
        assert_eq!(c.reroute_count, 1);
    }

    #[test]
    fn importance_prune_below_threshold() {
        let c = RustTopologyController::new();
        let d = c.check_importance_prune(2, 0.1, false, true).unwrap();
        assert_eq!(d.action, "prune_node");
        assert_eq!(d.target_node, Some(2));
        assert!(d.reason.starts_with("importance="));
    }

    #[test]
    fn importance_prune_above_threshold_is_none() {
        let c = RustTopologyController::new();
        assert!(c
            .check_importance_prune(2, THETA_PRUNE, false, true)
            .is_none());
        assert!(c.check_importance_prune(2, 0.8, false, true).is_none());
    }

    #[test]
    fn importance_prune_abstain_skips() {
        let c = RustTopologyController::new();
        // Quality unknown (estimator abstained) — never prune
        assert!(c.check_importance_prune(2, 0.1, false, false).is_none());
    }

    #[test]
    fn importance_prune_debate_skips() {
        let c = RustTopologyController::new();
        assert!(c.check_importance_prune(2, 0.1, true, true).is_none());
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

    // --- Task A: view methods + spawn gate + seed ----------------------------

    #[test]
    fn getters_match_internal_state_after_decisions() {
        let mut c = RustTopologyController::new_inner();
        c.reroute_count = 2;
        c.spawn_count = 1;
        c.abstain_count = 3;
        c.node_retries.insert(5, 7);
        c.node_qualities.insert(5, 0.42);

        // Getters (existing) return the current state.
        assert_eq!(c.reroute_count(), 2);
        assert_eq!(c.spawn_count(), 1);
        assert_eq!(c.abstain_count(), 3);

        // New view methods expose HashMap state as Vec<(K, V)>.
        let retries: std::collections::HashMap<usize, u32> =
            c.node_retries_view().into_iter().collect();
        assert_eq!(retries.get(&5).copied(), Some(7));
        let qualities: std::collections::HashMap<usize, f32> =
            c.node_qualities_view().into_iter().collect();
        assert_eq!(qualities.get(&5).copied(), Some(0.42));
    }

    #[test]
    fn should_trigger_emergent_spawn_allows_within_budget() {
        let c = RustTopologyController::new_inner();
        assert!(c.should_trigger_emergent_spawn(0));
    }

    #[test]
    fn should_trigger_emergent_spawn_refuses_at_cap() {
        let mut c = RustTopologyController::new_inner();
        c.spawn_count = MAX_SPAWNS;
        assert!(!c.should_trigger_emergent_spawn(0));
    }

    #[test]
    fn should_trigger_emergent_spawn_refuses_past_cap() {
        let mut c = RustTopologyController::new_inner();
        c.spawn_count = MAX_SPAWNS + 5;
        assert!(!c.should_trigger_emergent_spawn(99));
    }

    #[test]
    fn record_abstain_increments_counter() {
        let mut c = RustTopologyController::new_inner();
        assert_eq!(c.abstain_count, 0);
        c.record_abstain();
        c.record_abstain();
        assert_eq!(c.abstain_count, 2);
    }

    #[test]
    fn record_emergent_spawn_increments_counter() {
        pyo3::prepare_freethreaded_python();
        let mut c = RustTopologyController::new_inner();
        assert_eq!(c.spawn_count, 0);
        pyo3::Python::with_gil(|_py| {
            c.record_emergent_spawn(0).expect("first record ok");
            c.record_emergent_spawn(1).expect("second record ok");
        });
        assert_eq!(c.spawn_count, 2);
    }

    #[test]
    fn record_emergent_spawn_errors_at_cap() {
        pyo3::prepare_freethreaded_python();
        let mut c = RustTopologyController::new_inner();
        c.spawn_count = MAX_SPAWNS;
        pyo3::Python::with_gil(|_py| {
            let err = c.record_emergent_spawn(0).expect_err("should error at cap");
            let err_str = format!("{err}");
            assert!(
                err_str.contains("spawn budget exhausted"),
                "expected budget message, got: {err_str}"
            );
        });
        assert_eq!(c.spawn_count, MAX_SPAWNS);
    }

    #[test]
    fn seed_state_for_legacy_tests_populates_all_fields() {
        pyo3::prepare_freethreaded_python();
        let mut c = RustTopologyController::new_inner();
        pyo3::Python::with_gil(|_py| {
            c.seed_state_for_legacy_tests(1, 2, vec![(0, 2), (3, 1)], 4)
                .expect("seed ok");
        });
        assert_eq!(c.reroute_count, 1);
        assert_eq!(c.spawn_count, 2);
        assert_eq!(c.abstain_count, 4);
        assert_eq!(c.node_retries.get(&0).copied(), Some(2));
        assert_eq!(c.node_retries.get(&3).copied(), Some(1));
    }

    #[test]
    fn seed_state_for_legacy_tests_rejects_implausible_reroute() {
        pyo3::prepare_freethreaded_python();
        let mut c = RustTopologyController::new_inner();
        pyo3::Python::with_gil(|_py| {
            let result = c.seed_state_for_legacy_tests(MAX_REROUTES + 20, 0, vec![], 0);
            assert!(result.is_err());
        });
    }
}
