use sage_core::memory::WorkingMemory;

#[test]
fn test_add_and_get_event() {
    let mut mem = WorkingMemory::new("agent-1".to_string());

    let event_id = mem.add_event("tool_call", "Called bash with 'ls'");
    let event = mem.get_event(&event_id);

    assert!(event.is_some());
    let e = event.unwrap();
    assert_eq!(e.event_type, "tool_call");
    assert_eq!(e.content, "Called bash with 'ls'");
}

#[test]
fn test_add_child_agent() {
    let mut mem = WorkingMemory::new("parent".to_string());
    mem.add_child_agent("child-1".to_string());

    let children = mem.child_agents();
    assert_eq!(children.len(), 1);
    assert_eq!(children[0], "child-1");
}

#[test]
fn test_get_recent_events() {
    let mut mem = WorkingMemory::new("agent".to_string());
    for i in 0..10 {
        mem.add_event("step", &format!("Step {i}"));
    }

    let recent = mem.recent_events(3);
    assert_eq!(recent.len(), 3);
    assert_eq!(recent[0].content, "Step 7");
    assert_eq!(recent[2].content, "Step 9");
}

#[test]
fn test_summarize_compresses() {
    let mut mem = WorkingMemory::new("agent".to_string());
    for i in 0..20 {
        mem.add_event("step", &format!("Step {i}"));
    }

    assert_eq!(mem.event_count(), 20);
    mem.compress_old_events(5, "Summary of steps 0-14");
    // Should have 5 recent events + 1 summary event
    assert_eq!(mem.event_count(), 6);
}
