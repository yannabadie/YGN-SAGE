#![cfg(feature = "cognitive")]

use sage_core::routing::bandit::ContextualBandit;
use std::path::PathBuf;

fn temp_db() -> PathBuf {
    let dir = std::env::temp_dir();
    dir.join(format!("sage_bandit_test_{}.db", ulid::Ulid::new()))
}

// ── Test 1: Save and load roundtrip ─────────────────────────────────────────

#[test]
fn test_save_and_load_roundtrip() {
    let db = temp_db();
    let mut bandit = ContextualBandit::create(0.99, 0.2);
    bandit.add_arm("model-a", "sequential");
    bandit.add_arm("model-b", "avr");

    // Record some observations to change posteriors from defaults
    let d1 = bandit.choose(0.0).unwrap();
    bandit
        .record_outcome(&d1.decision_id, 0.9, 0.5, 100.0)
        .unwrap();

    // Save
    sage_core::routing::persistence::save_bandit(&bandit, db.to_str().unwrap()).unwrap();

    // Load
    let loaded =
        sage_core::routing::persistence::load_bandit(db.to_str().unwrap()).unwrap();
    assert_eq!(loaded.arm_count(), 2);
    assert_eq!(loaded.total_observations(), 1);

    // Cleanup
    let _ = std::fs::remove_file(&db);
}

// ── Test 2: Save empty bandit ───────────────────────────────────────────────

#[test]
fn test_save_empty_bandit() {
    let db = temp_db();
    let bandit = ContextualBandit::create(0.995, 0.1);
    sage_core::routing::persistence::save_bandit(&bandit, db.to_str().unwrap()).unwrap();
    let loaded =
        sage_core::routing::persistence::load_bandit(db.to_str().unwrap()).unwrap();
    assert_eq!(loaded.arm_count(), 0);
    let _ = std::fs::remove_file(&db);
}

// ── Test 3: Posteriors preserved after roundtrip ────────────────────────────

#[test]
fn test_posteriors_preserved() {
    let db = temp_db();
    let mut bandit = ContextualBandit::create(0.99, 0.15);
    bandit.add_arm("gpt-5", "avr");

    // Feed multiple observations
    for _ in 0..10 {
        let d = bandit.choose(0.0).unwrap();
        bandit
            .record_outcome(&d.decision_id, 0.8, 1.0, 200.0)
            .unwrap();
    }

    // Get summaries before save
    let before = bandit.arm_summaries();

    sage_core::routing::persistence::save_bandit(&bandit, db.to_str().unwrap()).unwrap();
    let loaded =
        sage_core::routing::persistence::load_bandit(db.to_str().unwrap()).unwrap();
    let after = loaded.arm_summaries();

    // Quality mean should be approximately preserved
    assert!(
        (before[0].2 - after[0].2).abs() < 0.01,
        "Quality mean should match: before={}, after={}",
        before[0].2,
        after[0].2
    );
    // Observation count must be exact
    assert_eq!(before[0].5, after[0].5);

    let _ = std::fs::remove_file(&db);
}

// ── Test 4: Config preserved ────────────────────────────────────────────────

#[test]
fn test_config_preserved() {
    let db = temp_db();
    let bandit = ContextualBandit::create(0.987, 0.42);
    sage_core::routing::persistence::save_bandit(&bandit, db.to_str().unwrap()).unwrap();
    let loaded =
        sage_core::routing::persistence::load_bandit(db.to_str().unwrap()).unwrap();

    // Verify config was preserved via accessor methods
    assert!(
        (loaded.decay_factor() - 0.987).abs() < 1e-10,
        "decay_factor should be preserved: got {}",
        loaded.decay_factor()
    );
    assert!(
        (loaded.exploration_bonus() - 0.42).abs() < 1e-10,
        "exploration_bonus should be preserved: got {}",
        loaded.exploration_bonus()
    );

    let _ = std::fs::remove_file(&db);
}

// ── Test 5: Overwrite existing DB ───────────────────────────────────────────

#[test]
fn test_overwrite_existing_db() {
    let db = temp_db();

    // Save once with one arm
    let mut bandit = ContextualBandit::create(0.99, 0.1);
    bandit.add_arm("model-a", "sequential");
    sage_core::routing::persistence::save_bandit(&bandit, db.to_str().unwrap()).unwrap();

    // Save again with different arms
    let mut bandit2 = ContextualBandit::create(0.99, 0.1);
    bandit2.add_arm("model-b", "parallel");
    bandit2.add_arm("model-c", "debate");
    sage_core::routing::persistence::save_bandit(&bandit2, db.to_str().unwrap()).unwrap();

    // Load should have model-a (from first save, still in table) + model-b, model-c
    let loaded =
        sage_core::routing::persistence::load_bandit(db.to_str().unwrap()).unwrap();
    assert!(
        loaded.arm_count() >= 2,
        "Should have at least 2 arms, got {}",
        loaded.arm_count()
    );

    let _ = std::fs::remove_file(&db);
}

// ── Test 6: Load nonexistent DB ─────────────────────────────────────────────

#[test]
fn test_load_nonexistent_db() {
    // rusqlite creates the file if it doesn't exist (but the directory must exist).
    // On a path with nonexistent directory, it should fail gracefully (no panic).
    let result =
        sage_core::routing::persistence::load_bandit("/nonexistent/path/bandit.db");
    // Either way, no panic — just assert we got through
    let _ = result;
}

// ── Test 7: Loaded bandit is functional ─────────────────────────────────────

#[test]
fn test_loaded_bandit_is_functional() {
    let db = temp_db();
    let mut bandit = ContextualBandit::create(0.99, 0.1);
    bandit.add_arm("model-a", "sequential");
    bandit.add_arm("model-b", "avr");

    // Record a few observations
    for _ in 0..5 {
        let d = bandit.choose(0.5).unwrap();
        bandit
            .record_outcome(&d.decision_id, 0.7, 0.3, 150.0)
            .unwrap();
    }

    sage_core::routing::persistence::save_bandit(&bandit, db.to_str().unwrap()).unwrap();
    let mut loaded =
        sage_core::routing::persistence::load_bandit(db.to_str().unwrap()).unwrap();

    // The loaded bandit should be fully functional: choose + record
    let decision = loaded.choose(0.0).unwrap();
    assert!(!decision.decision_id.is_empty());
    assert!(!decision.model_id.is_empty());
    loaded
        .record_outcome(&decision.decision_id, 0.9, 0.1, 50.0)
        .unwrap();

    // Observation count should now be original (5) + 1
    assert_eq!(loaded.total_observations(), 6);

    let _ = std::fs::remove_file(&db);
}

// ── Test 8: restore_arm creates correct posteriors ──────────────────────────

#[test]
fn test_restore_arm_posteriors() {
    let mut bandit = ContextualBandit::create(0.995, 0.1);
    bandit.restore_arm(
        "test-model".to_string(),
        "avr".to_string(),
        5.0,  // quality alpha
        2.0,  // quality beta
        3.0,  // cost shape
        10.0, // cost rate
        4.0,  // latency shape
        20.0, // latency rate
        42,   // observation count
    );

    assert_eq!(bandit.arm_count(), 1);
    assert_eq!(bandit.total_observations(), 42);

    let summaries = bandit.arm_summaries();
    let (model_id, template, quality_mean, cost_mean, latency_mean, obs) = &summaries[0];
    assert_eq!(model_id, "test-model");
    assert_eq!(template, "avr");
    // quality mean = 5.0 / (5.0 + 2.0) = 0.714...
    assert!((*quality_mean - 0.714).abs() < 0.01);
    // cost mean = 3.0 / 10.0 = 0.3
    assert!((*cost_mean - 0.3).abs() < 0.01);
    // latency mean = 4.0 / 20.0 = 0.2
    assert!((*latency_mean - 0.2).abs() < 0.01);
    assert_eq!(*obs, 42);
}
