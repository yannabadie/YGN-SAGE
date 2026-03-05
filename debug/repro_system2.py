import asyncio
import os
import json
import logging
from pathlib import Path
from sage.boot import boot_agent_system
from sage.agent_loop import LoopPhase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("repro-s2")

async def test_s2_routing_and_validation():
    """Verify System 2 routing and Level 2 empirical validation."""
    print("\n=== SYSTEM 2 REPRODUCTION TEST ===\n")
    
    # 1. Boot the system
    # Force mock for test reproducibility if needed, but here we want to see the routing logic
    system = boot_agent_system(use_mock_llm=True)
    
    # 2. Define a moderate task that SHOULD route to System 2
    # Task requires planning, tool consideration (but not necessarily formal proof)
    task = "Design a multi-threaded web scraper in Python that includes a robust exponential backoff retry mechanism and a clear separation between the fetching and parsing layers. Provide the implementation plan and key code structures."
    
    # 3. Track events to verify routing
    events = []
    def on_event(evt):
        events.append(evt)
        if evt.phase == LoopPhase.PERCEIVE:
            print(f"[PERCEIVE] System: {evt.data.get('system')}, Tier: {evt.data.get('routed_tier')}, Validation: {evt.data.get('validation_level')}")
        elif evt.phase == LoopPhase.THINK:
            if "content" in evt.data:
                print(f"[THINK] Content emitted (length: {len(evt.data['content'])})")
            if "validation" in evt.data:
                print(f"[THINK] Validation type: {evt.data['validation']}")

    system.agent_loop._on_event = on_event
    
    # 4. Run the task
    print(f"Running task: {task}")
    result = await system.run(task)
    
    # 5. Assertions
    perceive_evt = next((e for e in events if e.phase == LoopPhase.PERCEIVE), None)
    assert perceive_evt is not None, "Missing PERCEIVE event"
    
    system_routed = perceive_evt.data.get('system')
    validation_level = perceive_evt.data.get('validation_level')
    
    print(f"\nFinal Routing -> System: {system_routed}, Validation Level: {validation_level}")
    
    if system_routed == 2:
        print("SUCCESS: Task correctly routed to System 2.")
    else:
        print(f"WARNING: Task routed to System {system_routed} instead of 2. Check thresholds.")

    if validation_level == 2:
        print("SUCCESS: Validation level set to 2 (Empirical).")
    else:
        print(f"FAILED: Validation level is {validation_level}, expected 2.")

    # Check for THINK content emission
    think_evts = [e for e in events if e.phase == LoopPhase.THINK and "content" in e.data]
    if think_evts:
        print(f"SUCCESS: {len(think_evts)} THINK events emitted real-time content.")
    else:
        print("FAILED: No THINK events emitted content for real-time dashboard updates.")

if __name__ == "__main__":
    # Ensure we are in the right directory to import sage
    import sys
    sys.path.append("sage-python/src")
    
    asyncio.run(test_s2_routing_and_validation())
