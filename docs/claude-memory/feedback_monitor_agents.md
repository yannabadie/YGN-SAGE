---
name: Monitor background agents before deciding
description: Always check agent status before making implementation decisions — agents may have critical findings
type: feedback
---

NEVER proceed to implementation without checking ALL background agents first.

**Why:** During the April 7 session, I launched 2 research agents then proceeded to implement
the MiniMax fix without waiting. Agent 2 had a complete synthesis with 6 actionable tactics
(including "stronger model for repair" = +4-6pp) that could have changed the priority order.

**How to apply:**
1. Before any PLAN or EVOLVE step, check all running agents
2. If an agent seems stuck (>5min no new output), extract partial findings manually
3. If an agent has findings, integrate them BEFORE deciding what to implement
4. The LEARN phase isn't done until ALL research agents have reported
