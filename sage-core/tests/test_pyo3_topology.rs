//! Integration tests for PyO3 topology wrappers (PyTopologyEngine, PyTopologyExecutor, PyGenerateResult).
//!
//! These tests verify the public API surface that Python will consume,
//! without requiring a Python interpreter.

use sage_core::topology::pyo3_wrappers::{PyTopologyEngine, PyTopologyExecutor};
use sage_core::topology::templates;

// ── PyTopologyEngine integration tests ──────────────────────────────────────

#[test]
fn test_engine_defaults_are_empty() {
    let engine = PyTopologyEngine::new();
    assert_eq!(engine.topology_count(), 0);
    assert_eq!(engine.archive_cell_count(), 0);
    assert_eq!(engine.smmu_chunk_count(), 0);
    assert!((engine.archive_coverage() - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_engine_generate_returns_valid_result() {
    let mut engine = PyTopologyEngine::new();

    // S1 task should produce sequential template
    let result = engine.generate("say hello", None, 1, 0.0).unwrap();
    assert_eq!(result.source(), "template_fallback");
    assert_eq!(result.topology().template_type, "sequential");
    assert!(result.confidence() > 0.0);
    assert!(result.confidence() <= 1.0);
}

#[test]
fn test_engine_generate_s2_avr() {
    let mut engine = PyTopologyEngine::new();

    let result = engine
        .generate("Write a binary search function in Python", None, 2, 0.5)
        .unwrap();
    assert_eq!(result.source(), "template_fallback");
    assert_eq!(result.topology().template_type, "avr");
}

#[test]
fn test_engine_generate_s3_debate() {
    let mut engine = PyTopologyEngine::new();

    let result = engine.generate("Prove that P != NP", None, 3, 0.0).unwrap();
    assert_eq!(result.source(), "template_fallback");
    assert_eq!(result.topology().template_type, "debate");
}

#[test]
fn test_engine_cache_roundtrip() {
    let mut engine = PyTopologyEngine::new();
    let graph = templates::sequential("gemini-2.5-flash");
    let id = graph.id.clone();

    engine.cache_topology(&graph);
    assert_eq!(engine.topology_count(), 1);

    let retrieved = engine.get_topology(&id);
    assert!(retrieved.is_some());

    let g = retrieved.unwrap();
    assert_eq!(g.id, id);
    assert_eq!(g.template_type, "sequential");
    assert_eq!(g.py_node_count(), 3);
}

#[test]
fn test_engine_cache_multiple_topologies() {
    let mut engine = PyTopologyEngine::new();

    let g1 = templates::sequential("model-a");
    let g2 = templates::avr("model-b", "model-c");
    let g3 = templates::debate("model-d", "model-e");

    engine.cache_topology(&g1);
    engine.cache_topology(&g2);
    engine.cache_topology(&g3);

    assert_eq!(engine.topology_count(), 3);
    assert!(engine.get_topology(&g1.id).is_some());
    assert!(engine.get_topology(&g2.id).is_some());
    assert!(engine.get_topology(&g3.id).is_some());
}

#[test]
fn test_engine_get_nonexistent_returns_none() {
    let engine = PyTopologyEngine::new();
    assert!(engine.get_topology("does-not-exist").is_none());
}

#[test]
fn test_engine_evolve_empty_is_noop() {
    let mut engine = PyTopologyEngine::new();
    engine.evolve(10, 5);
    assert_eq!(engine.archive_cell_count(), 0);
}

#[test]
fn test_engine_record_outcome_populates_smmu() {
    let mut engine = PyTopologyEngine::new();

    // Generate + cache a topology first
    let result = engine.generate("sort numbers", None, 2, 0.0).unwrap();
    let topology = result.topology();
    let id = topology.id.clone();
    engine.cache_topology(&topology);

    // Record outcome
    engine.record_outcome(
        &id,
        "sorted numbers successfully",
        vec!["sort".into(), "algorithm".into()],
        None,
        0.85,
        0.005,
        120.0,
    );

    // S-MMU should have gained at least one chunk
    assert!(engine.smmu_chunk_count() > 0);
}

#[test]
fn test_engine_repr_format() {
    let engine = PyTopologyEngine::new();
    let repr = engine.__repr__();
    assert!(repr.starts_with("TopologyEngine("));
    assert!(repr.contains("cached=0"));
    assert!(repr.contains("archive_cells=0"));
    assert!(repr.contains("smmu_chunks=0"));
}

// ── PyTopologyExecutor integration tests ────────────────────────────────────

#[test]
fn test_executor_sequential_mode() {
    let graph = templates::sequential("model");
    let exec = PyTopologyExecutor::new(&graph);
    assert_eq!(exec.mode(), "static");
    assert!(!exec.is_done());
}

#[test]
fn test_executor_avr_dynamic_mode() {
    let graph = templates::avr("model-a", "model-b");
    let exec = PyTopologyExecutor::new(&graph);
    assert_eq!(exec.mode(), "dynamic");
    assert!(!exec.is_done());
}

#[test]
fn test_executor_complete_sequential_flow() {
    let graph = templates::sequential("model");
    let mut exec = PyTopologyExecutor::new(&graph);

    // Sequential: 3 nodes, one at a time
    let w1 = exec.next_ready(&graph);
    assert_eq!(w1.len(), 1);
    assert_eq!(w1[0], 0);

    exec.mark_running(0);
    exec.mark_completed(0);

    let w2 = exec.next_ready(&graph);
    assert_eq!(w2.len(), 1);
    assert_eq!(w2[0], 1);

    exec.mark_running(1);
    exec.mark_completed(1);

    let w3 = exec.next_ready(&graph);
    assert_eq!(w3.len(), 1);
    assert_eq!(w3[0], 2);

    exec.mark_running(2);
    exec.mark_completed(2);

    assert!(exec.is_done());
    assert!(exec.next_ready(&graph).is_empty());
}

#[test]
fn test_executor_parallel_concurrent_workers() {
    let graph = templates::parallel("model", 3);
    let mut exec = PyTopologyExecutor::new(&graph);

    // Source node first
    let w1 = exec.next_ready(&graph);
    assert_eq!(w1, vec![0]);
    exec.mark_completed(0);

    // All 3 workers ready simultaneously
    let mut w2 = exec.next_ready(&graph);
    w2.sort();
    assert_eq!(w2, vec![1, 2, 3]);

    for &w in &w2 {
        exec.mark_completed(w);
    }

    // Aggregator
    let w3 = exec.next_ready(&graph);
    assert_eq!(w3, vec![4]);
    exec.mark_completed(4);

    assert!(exec.is_done());
}

#[test]
fn test_executor_skip_all_nodes() {
    let graph = templates::sequential("model");
    let mut exec = PyTopologyExecutor::new(&graph);

    exec.mark_skipped(0);
    exec.mark_skipped(1);
    exec.mark_skipped(2);

    assert!(exec.is_done());
}

#[test]
fn test_executor_repr_format() {
    let graph = templates::sequential("model");
    let exec = PyTopologyExecutor::new(&graph);
    let repr = exec.__repr__();
    assert!(repr.starts_with("TopologyExecutor("));
    assert!(repr.contains("mode='static'"));
    assert!(repr.contains("done=false"));
}

// ── PyGenerateResult integration tests ──────────────────────────────────────

#[test]
fn test_generate_result_getters() {
    let mut engine = PyTopologyEngine::new();
    let result = engine.generate("test", None, 1, 0.0).unwrap();

    // All getters should work
    let t = result.topology();
    assert!(t.py_node_count() > 0);

    let s = result.source();
    assert!(!s.is_empty());

    let c = result.confidence();
    assert!(c >= 0.0 && c <= 1.0);
}

#[test]
fn test_generate_result_repr() {
    let mut engine = PyTopologyEngine::new();
    let result = engine.generate("test", None, 2, 0.0).unwrap();

    let repr = result.__repr__();
    assert!(repr.starts_with("GenerateResult("));
    assert!(repr.contains("source='template_fallback'"));
    assert!(repr.contains("template='avr'"));
}

// ── Cross-component integration tests ───────────────────────────────────────

#[test]
fn test_engine_generate_then_execute() {
    let mut engine = PyTopologyEngine::new();

    // Generate a topology
    let result = engine.generate("parse CSV file", None, 1, 0.0).unwrap();
    let graph = result.topology();

    // Create an executor for it
    let mut exec = PyTopologyExecutor::new(&graph);
    assert_eq!(exec.mode(), "static"); // sequential is static

    // Execute all nodes
    let node_count = graph.py_node_count();
    let mut completed = 0;

    loop {
        let ready = exec.next_ready(&graph);
        if ready.is_empty() {
            break;
        }
        for idx in ready {
            exec.mark_completed(idx);
            completed += 1;
        }
    }

    assert!(exec.is_done());
    assert_eq!(completed, node_count);
}

#[test]
fn test_full_cycle_generate_execute_record() {
    let mut engine = PyTopologyEngine::new();

    // 1. Generate
    let result = engine.generate("merge sort", None, 2, 0.5).unwrap();
    let graph = result.topology();
    let id = graph.id.clone();

    // 2. Cache
    engine.cache_topology(&graph);

    // 3. Execute
    let mut exec = PyTopologyExecutor::new(&graph);
    loop {
        let ready = exec.next_ready(&graph);
        if ready.is_empty() {
            break;
        }
        for idx in ready {
            exec.mark_completed(idx);
        }
    }
    assert!(exec.is_done());

    // 4. Record outcome
    engine.record_outcome(
        &id,
        "merge sort implemented successfully",
        vec!["sort".into(), "merge".into(), "algorithm".into()],
        None,
        0.95,
        0.008,
        200.0,
    );

    // Verify state
    assert_eq!(engine.topology_count(), 1);
    assert!(engine.smmu_chunk_count() > 0);
}
