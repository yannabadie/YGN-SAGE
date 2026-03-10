use sage_core::routing::bandit::ContextualBandit;

// ── Test 1: Register and count arms ────────────────────────────────────────

#[test]
fn test_register_and_count_arms() {
    let mut bandit = ContextualBandit::create(0.995, 0.1);
    assert_eq!(bandit.arm_count(), 0);

    bandit.add_arm("gemini-2.5-flash", "sequential");
    bandit.add_arm("gpt-5.3-codex", "avr");
    bandit.add_arm("gemini-3.1-pro", "parallel");
    assert_eq!(bandit.arm_count(), 3);

    // Duplicate registration is a no-op
    bandit.add_arm("gemini-2.5-flash", "sequential");
    assert_eq!(bandit.arm_count(), 3);
}

// ── Test 2: Select returns a valid decision ────────────────────────────────

#[test]
fn test_select_returns_decision() {
    let mut bandit = ContextualBandit::create(0.995, 0.1);
    bandit.add_arm("gemini-2.5-flash", "sequential");
    bandit.add_arm("gpt-5.3-codex", "avr");

    let decision = bandit.choose(0.0).unwrap();

    // Decision has all expected fields populated
    assert!(!decision.decision_id.is_empty());
    assert_eq!(decision.decision_id.len(), 26); // ULID is 26 chars
    assert!(!decision.model_id.is_empty());
    assert!(!decision.template.is_empty());
    assert!(decision.expected_quality >= 0.0);
    assert!(decision.expected_cost > 0.0);
    assert!(decision.expected_latency > 0.0);
}

// ── Test 3: Select with no arms errors ─────────────────────────────────────

#[test]
fn test_select_no_arms_errors() {
    let mut bandit = ContextualBandit::create(0.995, 0.1);

    let result = bandit.choose(0.0);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("No arms"),
        "Error should mention no arms: {}",
        err_msg
    );
}

// ── Test 4: Record updates posteriors ──────────────────────────────────────

#[test]
fn test_record_updates_posteriors() {
    let mut bandit = ContextualBandit::create(0.995, 0.1);
    bandit.add_arm("model-a", "sequential");

    // Get initial quality mean
    let initial_summaries = bandit.arm_summaries();
    let initial_quality = initial_summaries[0].2;

    // Select and record a perfect outcome
    let decision = bandit.choose(0.0).unwrap();
    bandit
        .record_outcome(&decision.decision_id, 1.0, 0.01, 100.0)
        .unwrap();

    // Quality mean should increase after a perfect observation
    let updated_summaries = bandit.arm_summaries();
    let updated_quality = updated_summaries[0].2;
    assert!(
        updated_quality > initial_quality,
        "Quality mean should increase after perfect observation: {} > {}",
        updated_quality,
        initial_quality
    );
}

// ── Test 5: Record with unknown decision_id errors ─────────────────────────

#[test]
fn test_record_unknown_decision_errors() {
    let mut bandit = ContextualBandit::create(0.995, 0.1);
    bandit.add_arm("model-a", "sequential");

    let result = bandit.record_outcome("FAKE_DECISION_ID_123456789", 0.9, 0.01, 100.0);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("FAKE_DECISION_ID"),
        "Error should mention the decision_id: {}",
        err_msg
    );
}

// ── Test 6: Decay reduces old observations' weight ─────────────────────────

#[test]
fn test_decay_reduces_old_observations() {
    let mut bandit = ContextualBandit::create(0.99, 0.1); // Aggressive decay for testing
    bandit.add_arm("model-a", "sequential");

    // Feed many high-quality observations
    for _ in 0..20 {
        let decision = bandit.choose(0.0).unwrap();
        bandit
            .record_outcome(&decision.decision_id, 1.0, 0.01, 100.0)
            .unwrap();
    }

    let quality_after_good = bandit.arm_summaries()[0].2;

    // Now feed low-quality observations — decay should reduce old weight
    for _ in 0..20 {
        let decision = bandit.choose(0.0).unwrap();
        bandit
            .record_outcome(&decision.decision_id, 0.0, 0.01, 100.0)
            .unwrap();
    }

    let quality_after_bad = bandit.arm_summaries()[0].2;

    assert!(
        quality_after_bad < quality_after_good,
        "Quality should decrease after bad observations: {} < {}",
        quality_after_bad,
        quality_after_good
    );
}

// ── Test 7: Exploration budget zero exploits ───────────────────────────────

#[test]
fn test_exploration_budget_zero_exploits() {
    let mut bandit = ContextualBandit::create(0.995, 0.1);
    bandit.add_arm("good-model", "avr");
    bandit.add_arm("bad-model", "sequential");

    // Train the bandit: good-model gets quality=1.0, bad-model gets quality=0.0
    for _ in 0..30 {
        let decision = bandit.choose(1.0).unwrap(); // explore to hit both
        if decision.model_id == "good-model" {
            bandit
                .record_outcome(&decision.decision_id, 1.0, 0.01, 100.0)
                .unwrap();
        } else {
            bandit
                .record_outcome(&decision.decision_id, 0.0, 0.5, 500.0)
                .unwrap();
        }
    }

    // With exploration_budget=0.0, should exploit (pick highest quality)
    let mut good_count = 0;
    for _ in 0..20 {
        let decision = bandit.choose(0.0).unwrap();
        if decision.model_id == "good-model" {
            good_count += 1;
        }
        // Record to clear pending
        bandit
            .record_outcome(&decision.decision_id, 0.5, 0.01, 100.0)
            .unwrap();
    }

    // Thompson sampling should strongly prefer the good arm
    assert!(
        good_count >= 14,
        "With exploit mode, good-model should be picked most often: got {}/20",
        good_count
    );
}

// ── Test 8: Exploration budget one explores ────────────────────────────────

#[test]
fn test_exploration_budget_one_explores() {
    let mut bandit = ContextualBandit::create(0.995, 0.1);
    bandit.add_arm("model-a", "sequential");
    bandit.add_arm("model-b", "avr");
    bandit.add_arm("model-c", "parallel");

    // Train model-a to be clearly best
    for _ in 0..20 {
        let decision = bandit.choose(1.0).unwrap();
        let quality = if decision.model_id == "model-a" {
            1.0
        } else {
            0.0
        };
        bandit
            .record_outcome(&decision.decision_id, quality, 0.01, 100.0)
            .unwrap();
    }

    // With exploration_budget=1.0, should explore (pick random arms)
    let mut selections: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    for _ in 0..60 {
        let decision = bandit.choose(1.0).unwrap();
        *selections.entry(decision.model_id.clone()).or_default() += 1;
        bandit
            .record_outcome(&decision.decision_id, 0.5, 0.01, 100.0)
            .unwrap();
    }

    // With pure exploration over 3 arms and 60 trials, each arm should
    // be picked at least a few times (extremely unlikely to miss one)
    assert!(
        selections.len() >= 2,
        "Pure exploration should hit multiple arms: got {:?}",
        selections
    );
}

// ── Test 9: Arm summaries format ───────────────────────────────────────────

#[test]
fn test_arm_summaries_format() {
    let mut bandit = ContextualBandit::create(0.995, 0.1);
    bandit.add_arm("model-a", "sequential");
    bandit.add_arm("model-b", "avr");

    let summaries = bandit.arm_summaries();
    assert_eq!(summaries.len(), 2);

    for (model_id, template, quality_mean, cost_mean, latency_mean, obs_count) in &summaries {
        assert!(!model_id.is_empty());
        assert!(!template.is_empty());
        // Initial means from priors
        assert!(*quality_mean >= 0.0 && *quality_mean <= 1.0);
        assert!(*cost_mean > 0.0);
        assert!(*latency_mean > 0.0);
        assert_eq!(*obs_count, 0);
    }
}

// ── Test 10: Many observations converge ────────────────────────────────────

#[test]
fn test_many_observations_converge() {
    let mut bandit = ContextualBandit::create(0.999, 0.1); // Slow decay to accumulate
    bandit.add_arm("model-a", "sequential");

    // Feed 100 observations with quality=0.9
    for _ in 0..100 {
        let decision = bandit.choose(0.0).unwrap();
        bandit
            .record_outcome(&decision.decision_id, 0.9, 0.05, 200.0)
            .unwrap();
    }

    let summaries = bandit.arm_summaries();
    let quality_mean = summaries[0].2;

    assert!(
        quality_mean > 0.7,
        "After 100 observations of quality=0.9, mean should be > 0.7, got {}",
        quality_mean
    );
}

// ── Test 11: Multiple arms, best arm selected most often ───────────────────

#[test]
fn test_multiple_arms_selection() {
    let mut bandit = ContextualBandit::create(0.995, 0.1);

    // Register 5 arms
    let arms = vec![
        ("model-a", "sequential", 0.9_f32),
        ("model-b", "avr", 0.7),
        ("model-c", "parallel", 0.5),
        ("model-d", "loop", 0.3),
        ("model-e", "z3", 0.1),
    ];
    for (model, template, _) in &arms {
        bandit.add_arm(model, template);
    }

    // Train each arm with its designated quality
    for _ in 0..40 {
        let decision = bandit.choose(1.0).unwrap(); // explore to reach all arms
        let quality = arms
            .iter()
            .find(|(m, t, _)| *m == decision.model_id && *t == decision.template)
            .map(|(_, _, q)| *q)
            .unwrap_or(0.5);
        bandit
            .record_outcome(&decision.decision_id, quality, 0.01, 100.0)
            .unwrap();
    }

    // Now exploit: count how often the best arm (model-a) is selected
    let mut best_count = 0;
    for _ in 0..40 {
        let decision = bandit.choose(0.0).unwrap();
        if decision.model_id == "model-a" {
            best_count += 1;
        }
        // Record to clear pending
        let quality = arms
            .iter()
            .find(|(m, t, _)| *m == decision.model_id && *t == decision.template)
            .map(|(_, _, q)| *q)
            .unwrap_or(0.5);
        bandit
            .record_outcome(&decision.decision_id, quality, 0.01, 100.0)
            .unwrap();
    }

    assert!(
        best_count >= 15,
        "Best arm (model-a, quality=0.9) should be picked most often in exploit mode: got {}/40",
        best_count
    );
}

// ── Test 12: repr output ───────────────────────────────────────────────────

#[test]
fn test_bandit_repr() {
    let mut bandit = ContextualBandit::create(0.995, 0.1);
    bandit.add_arm("model-a", "sequential");
    bandit.add_arm("model-b", "avr");

    let repr = bandit.repr();
    assert!(
        repr.contains("ContextualBandit"),
        "repr should contain class name: {}",
        repr
    );
    assert!(repr.contains("arms=2"), "repr should show arm count: {}", repr);
    assert!(
        repr.contains("observations=0"),
        "repr should show observation count: {}",
        repr
    );
    assert!(
        repr.contains("0.9950"),
        "repr should show decay factor: {}",
        repr
    );
}
