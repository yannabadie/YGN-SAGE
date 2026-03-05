use sage_core::memory::WorkingMemory;

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
        )
        .expect("compact should succeed");

    assert_eq!(chunk_id, 0);
    assert_eq!(mem.smmu_chunk_count(), 1);
    // Active buffer should be cleared after compaction.
    assert_eq!(mem.recent_events(100).len(), 0);
    // Event count should reflect compacted events.
    assert_eq!(mem.event_count(), 5);
}

#[test]
fn test_multi_chunk_temporal_linking() {
    let mut mem = WorkingMemory::new("agent-temporal".to_string(), None);

    // Create 3 chunks sequentially.
    for batch in 0..3 {
        for i in 0..3 {
            mem.add_event("step", &format!("Batch {batch} Step {i}"));
        }
        let cid = mem
            .compact_to_arrow_with_meta(vec![], None, None)
            .expect("compact should succeed");
        assert_eq!(cid, batch);
    }

    assert_eq!(mem.smmu_chunk_count(), 3);

    // From chunk 0, we should be able to reach chunk 1 and chunk 2 via temporal edges.
    let relevant = mem.retrieve_relevant_chunks(0, 3, None);
    let ids: Vec<usize> = relevant.iter().map(|&(cid, _)| cid).collect();

    // Both chunk 1 and 2 should be reachable.
    assert!(ids.contains(&1), "chunk 1 should be reachable from chunk 0");
    assert!(ids.contains(&2), "chunk 2 should be reachable from chunk 0");

    // chunk 1 should be more relevant than chunk 2 (closer temporally).
    let score_1 = relevant.iter().find(|&&(cid, _)| cid == 1).unwrap().1;
    let score_2 = relevant.iter().find(|&&(cid, _)| cid == 2).unwrap().1;
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

    // Create a "parent" chunk first (chunk 0).
    for i in 0..3 {
        mem.add_event("parent_step", &format!("Parent step {i}"));
    }
    let parent_cid = mem
        .compact_to_arrow_with_meta(
            vec!["planning".to_string()],
            None,
            None,
        )
        .expect("compact parent chunk");
    assert_eq!(parent_cid, 0);

    // Create a child chunk that causally links to the parent chunk.
    for i in 0..3 {
        mem.add_event("child_step", &format!("Child step {i}"));
    }
    let child_cid = mem
        .compact_to_arrow_with_meta(
            vec!["execution".to_string()],
            None,
            Some(parent_cid), // causal link to parent
        )
        .expect("compact child chunk");
    assert_eq!(child_cid, 1);

    // From the parent chunk, the child should be reachable via causal edge.
    let relevant = mem.retrieve_relevant_chunks(parent_cid, 2, None);
    let ids: Vec<usize> = relevant.iter().map(|&(cid, _)| cid).collect();
    assert!(
        ids.contains(&child_cid),
        "child chunk should be reachable from parent via causal link"
    );
}

#[test]
fn test_page_out_candidates() {
    let mut mem = WorkingMemory::new("agent-paging".to_string(), None);

    // Create 5 chunks. The first 2 will be disconnected from the last one
    // if max_hops is restricted.
    for batch in 0..5 {
        for i in 0..2 {
            mem.add_event("step", &format!("Batch {batch} Step {i}"));
        }
        mem.compact_to_arrow_with_meta(vec![], None, None)
            .expect("compact should succeed");
    }

    assert_eq!(mem.smmu_chunk_count(), 5);

    // Active chunk is the last one (4). With max_hops=1, only chunk 3 is
    // reachable (direct temporal predecessor).
    let candidates = mem.get_page_out_candidates(4, 1, 3);

    // Should return at most 3 candidates.
    assert!(candidates.len() <= 3);
    // The active chunk itself should never be a candidate.
    assert!(
        !candidates.contains(&4),
        "active chunk should not be a page-out candidate"
    );
    // Chunk 3 is the most relevant (direct predecessor), so it should NOT
    // be among the first candidates.
    // Chunks 0, 1 are unreachable within 1 hop — they should be candidates.
    // (Chunk 2 is also unreachable within 1 hop.)
    let has_distant = candidates.iter().any(|&c| c <= 2);
    assert!(
        has_distant,
        "distant chunks (0-2) should be page-out candidates; got {:?}",
        candidates
    );
}
