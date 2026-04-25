---
name: Improvement Loop Methodology
description: User-mandated workflow for every improvement cycle — research→analyze→plan→evolve→learn→loop
type: feedback
---

Every improvement cycle MUST follow this loop:

1. **LEARN** — gather knowledge from multiple sources:
   - arXiv papers (relevant to the problem)
   - Codebase analysis (grep, read, understand current state)
   - Context7 (library docs for any framework/SDK involved)
   - GitHub + web research (competitors, implementations, benchmarks)

2. **ANALYZE** — synthesize + adversarial:
   - Synthesize findings across all sources
   - Be adversarial: question assumptions, challenge claims, find contradictions
   - Compare paper claims vs actual code reality

3. **PLAN** — define implementation strategy

4. **EVOLVE** — implement the changes

5. **ANALYZE LEARNS** — reflect on what worked/failed

6. **LOOP()** — repeat with new learnings

**Why:** The user wants research-backed, evidence-driven development. No blind implementation.
Never skip the research phase. Always question before coding.

**How to apply:** Before any major feature or fix, run the full loop. For small fixes, a lightweight version (codebase analysis + plan + implement) suffices.
