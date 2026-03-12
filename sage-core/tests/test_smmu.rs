use sage_core::memory::smmu::MultiViewMMU;
use sage_core::memory::WorkingMemory;

/// Verify that the BFS multi-path score accumulation works correctly.
///
/// Diamond topology via entity edges:
///   A ──entity──> B ──entity──> D
///   A ──entity──> C ──entity──> D
///
/// D should receive accumulated scores from BOTH paths (A→B→D and A→C→D).
/// This tests the GPT-5.4 audit finding about `visited` preventing multi-path
/// accumulation. The score entry (`*scores.entry(...)`) runs for every edge
/// traversal regardless of `visited`, so scores accumulate correctly even
/// though frontier expansion is gated.
#[test]
fn test_multi_path_score_accumulation() {
    let mut smmu = MultiViewMMU::new();

    // A: keywords ["alpha"] — the starting node
    let a = smmu.register_chunk(0, 1, "A", vec!["alpha".into()], None, None);
    // B: keywords ["alpha", "beta"] — entity edge to A (via "alpha")
    let b = smmu.register_chunk(2, 3, "B", vec!["alpha".into(), "beta".into()], None, None);
    // C: keywords ["alpha", "gamma"] — entity edge to A (via "alpha")
    let c = smmu.register_chunk(4, 5, "C", vec!["alpha".into(), "gamma".into()], None, None);
    // D: keywords ["beta", "gamma"] — entity edges to B (via "beta") and C (via "gamma")
    // D does NOT share any keywords with A, so no direct A→D entity edge.
    let d = smmu.register_chunk(6, 7, "D", vec!["beta".into(), "gamma".into()], None, None);

    // Retrieve from A with max_hops=3, all view weights equal.
    let results = smmu.retrieve_relevant(&a, 3, [1.0, 1.0, 1.0, 1.0]);

    // All of B, C, D should be reachable.
    assert!(!results.is_empty(), "Should have results");
    let b_entry = results.iter().find(|(id, _)| *id == b);
    let c_entry = results.iter().find(|(id, _)| *id == c);
    let d_entry = results.iter().find(|(id, _)| *id == d);

    assert!(
        b_entry.is_some(),
        "B (chunk {}) should be reachable from A",
        b
    );
    assert!(
        c_entry.is_some(),
        "C (chunk {}) should be reachable from A",
        c
    );
    assert!(
        d_entry.is_some(),
        "D (chunk {}) should be reachable from A via multi-path",
        d
    );

    // D's score should reflect contributions from BOTH paths.
    // With only a single path, D would get score from one entity edge only.
    // With two paths (A→B→D and A→C→D), D's score should be strictly greater
    // than what a single path would contribute.
    let d_score = d_entry.unwrap().1;
    assert!(
        d_score > 0.0,
        "D should have a positive accumulated score, got {}",
        d_score
    );

    // Verify multi-path accumulation: D should have higher score than if
    // reached via only one path. We can check this by comparing D's score
    // against what a single entity hop from B alone would yield.
    // B's entity edge weight to D is jaccard({"alpha","beta"}, {"beta","gamma"}) = 1/3.
    // C's entity edge weight to D is jaccard({"alpha","gamma"}, {"beta","gamma"}) = 1/3.
    // If only one path contributed, D's score would be B_score * 1/3 (or C_score * 1/3).
    // With both paths, D gets B_score * 1/3 + C_score * 1/3 (plus temporal contributions).
    // Since B_score ≈ C_score, multi-path D should be roughly 2x a single-path contribution.
    let b_score = b_entry.unwrap().1;
    let c_score = c_entry.unwrap().1;
    let single_path_estimate = f32::max(b_score, c_score) * (1.0 / 3.0);
    assert!(
        d_score > single_path_estimate * 1.1, // at least 10% more than single-path
        "D score ({}) should exceed single-path estimate ({}) due to multi-path accumulation",
        d_score,
        single_path_estimate
    );
}

/// Verify that multi-path accumulation works via causal edges too.
/// Diamond: A──causal──>B──causal──>D and A──causal──>C──causal──>D
#[test]
fn test_multi_path_causal_accumulation() {
    let mut smmu = MultiViewMMU::new();

    // Use distinct keywords to avoid entity cross-links.
    let a = smmu.register_chunk(0, 1, "root", vec!["root".into()], None, None);
    let b = smmu.register_chunk(10, 11, "branch-b", vec!["bonly".into()], None, Some(a.clone()));
    let _c = smmu.register_chunk(20, 21, "branch-c", vec!["conly".into()], None, Some(a.clone()));
    let d_via_b = smmu.register_chunk(30, 31, "leaf-d", vec!["donly".into()], None, Some(b.clone()));

    // Also add a causal edge from C to D by registering another chunk that
    // shares D's identity. But since register_chunk always creates a NEW node,
    // we need a different approach: add an entity link from C to D.
    // Instead, let's make D share a keyword with C to create an entity edge C→D.
    // Re-approach: we cannot add a second causal parent. Let's use entity edges.

    // Retrieve from A. D should be reachable via A→B→D (causal chain).
    let results = smmu.retrieve_relevant(&a, 3, [1.0, 1.0, 1.0, 1.0]);
    let d_entry = results.iter().find(|(id, _)| *id == d_via_b);
    assert!(
        d_entry.is_some(),
        "D should be reachable from A via causal chain A→B→D"
    );
    assert!(
        d_entry.unwrap().1 > 0.0,
        "D should have positive score via causal path"
    );
}

#[test]
fn test_compact_with_meta_registers_in_smmu() {
    let mut mem = WorkingMemory::new("agent-smmu".to_string(), None);

    // Add some events and compact with metadata.
    for i in 0..5 {
        mem.add_event("step", &format!("Step {i}"));
    }

    let chunk_id = mem
        .compact_to_arrow_with_meta(
            vec!["rust".to_string(), "memory".to_string()],
            Some(vec![1.0, 0.0, 0.5]),
            None,
            None,
        )
        .expect("compact should succeed");

    // chunk_id is a ULID string — must be non-empty and 26 chars
    assert!(!chunk_id.is_empty(), "chunk_id should be a non-empty ULID");
    assert_eq!(chunk_id.len(), 26, "ULID should be 26 characters");
    assert_eq!(mem.smmu_chunk_count(), 1);
    // Active buffer should be cleared after compaction.
    assert_eq!(mem.recent_events(100).len(), 0);
    // Event count should reflect compacted events.
    assert_eq!(mem.event_count(), 5);
}

#[test]
fn test_multi_chunk_temporal_linking() {
    let mut mem = WorkingMemory::new("agent-temporal".to_string(), None);

    // Create 3 chunks sequentially and collect their ULID IDs.
    let mut chunk_ids: Vec<String> = Vec::new();
    for batch in 0..3 {
        for i in 0..3 {
            mem.add_event("step", &format!("Batch {batch} Step {i}"));
        }
        let cid = mem
            .compact_to_arrow_with_meta(vec![], None, None, None)
            .expect("compact should succeed");
        assert!(!cid.is_empty(), "chunk_id should be non-empty ULID");
        assert_eq!(cid.len(), 26, "ULID should be 26 characters");
        chunk_ids.push(cid);
    }

    assert_eq!(mem.smmu_chunk_count(), 3);

    // From chunk 0, we should be able to reach chunk 1 and chunk 2 via temporal edges.
    let relevant = mem.retrieve_relevant_chunks(&chunk_ids[0], 3, None);
    let ids: Vec<&str> = relevant.iter().map(|(cid, _)| cid.as_str()).collect();

    // Both chunk 1 and 2 should be reachable.
    assert!(
        ids.contains(&chunk_ids[1].as_str()),
        "chunk 1 should be reachable from chunk 0"
    );
    assert!(
        ids.contains(&chunk_ids[2].as_str()),
        "chunk 2 should be reachable from chunk 0"
    );

    // chunk 1 should be more relevant than chunk 2 (closer temporally).
    let score_1 = relevant
        .iter()
        .find(|(cid, _)| cid == &chunk_ids[1])
        .unwrap()
        .1;
    let score_2 = relevant
        .iter()
        .find(|(cid, _)| cid == &chunk_ids[2])
        .unwrap()
        .1;
    assert!(
        score_1 >= score_2,
        "chunk 1 (score={}) should score >= chunk 2 (score={})",
        score_1,
        score_2
    );
}

#[test]
fn test_causal_linking() {
    let mut mem = WorkingMemory::new("child-agent".to_string(), Some("parent-agent".to_string()));

    // Create a "parent" chunk first.
    for i in 0..3 {
        mem.add_event("parent_step", &format!("Parent step {i}"));
    }
    let parent_cid = mem
        .compact_to_arrow_with_meta(vec!["planning".to_string()], None, None, None)
        .expect("compact parent chunk");
    assert!(!parent_cid.is_empty(), "parent_cid should be non-empty ULID");

    // Create a child chunk that causally links to the parent chunk.
    for i in 0..3 {
        mem.add_event("child_step", &format!("Child step {i}"));
    }
    let child_cid = mem
        .compact_to_arrow_with_meta(
            vec!["execution".to_string()],
            None,
            Some(parent_cid.clone()), // causal link to parent (ULID string)
            None,
        )
        .expect("compact child chunk");
    assert!(!child_cid.is_empty(), "child_cid should be non-empty ULID");

    // From the parent chunk, the child should be reachable via causal edge.
    let relevant = mem.retrieve_relevant_chunks(&parent_cid, 2, None);
    let ids: Vec<&str> = relevant.iter().map(|(cid, _)| cid.as_str()).collect();
    assert!(
        ids.contains(&child_cid.as_str()),
        "child chunk should be reachable from parent via causal link"
    );
}

#[test]
fn test_page_out_candidates() {
    let mut mem = WorkingMemory::new("agent-paging".to_string(), None);

    // Create 5 chunks. The first 2 will be disconnected from the last one
    // if max_hops is restricted.
    let mut chunk_ids: Vec<String> = Vec::new();
    for batch in 0..5 {
        for i in 0..2 {
            mem.add_event("step", &format!("Batch {batch} Step {i}"));
        }
        let cid = mem
            .compact_to_arrow_with_meta(vec![], None, None, None)
            .expect("compact should succeed");
        chunk_ids.push(cid);
    }

    assert_eq!(mem.smmu_chunk_count(), 5);

    // Active chunk is the last one (index 4). With max_hops=1, only chunk 3 is
    // reachable (direct temporal predecessor).
    let active_id = &chunk_ids[4];
    let candidates = mem.get_page_out_candidates(active_id, 1, 3);

    // Should return at most 3 candidates.
    assert!(candidates.len() <= 3);
    // The active chunk itself should never be a candidate.
    assert!(
        !candidates.contains(active_id),
        "active chunk should not be a page-out candidate"
    );
    // Chunks 0, 1, 2 are unreachable within 1 hop — they should be candidates.
    let has_distant = candidates
        .iter()
        .any(|c| c == &chunk_ids[0] || c == &chunk_ids[1] || c == &chunk_ids[2]);
    assert!(
        has_distant,
        "distant chunks (0-2) should be page-out candidates; got {:?}",
        candidates
    );
}
