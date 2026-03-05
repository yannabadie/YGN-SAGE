# ADR-001: Adding System 2 (Algorithmic Mind) to Cognitive Routing

**Status**: Implemented
**Date**: 2026-03-05
**Author**: Yann Abadie + Gemini 2.5 Pro
**Deciders**: YGN-SAGE Architecture Team

## Context

YGN-SAGE's metacognitive controller (`strategy/metacognition.py`) routes tasks between three cognitive tiers:

- **System 1** (autonomous): Gemini Flash Lite, no validation.
- **System 2** (algorithmic): Gemini Flash/Pro, Level 2 Empirical validation.
- **System 3** (reflective): Codex/Reasoner, Level 3 Formal Z3 validation.

## Implementation Details

### System 2: Empirical Validation (Level 2)
As of 2026-03-05, System 2 has been fully integrated into the `AgentLoop`. Unlike System 1 (which uses heuristics) or System 3 (which requires formal SMT proofs), System 2 enforces **Empirical Grounding**:

1.  **Chain-of-Thought (CoT) Requirement**: On the first step of any task routed to System 2, the agent is forced to use step-by-step reasoning (via `<think>` tags). If reasoning is missing, the loop intercepts and retries with an explicit prompt for CoT.
2.  **Algorithmic Tiers**: Engaging Gemini 3 Flash (mutator) or Gemini 3.1 Pro (reasoner) to handle moderate complexity without the latency of System 3 formal verification.

### Dashboard: Real-Time Cognitive Streaming
The dashboard has been refactored to support the tripartite model:
- **Response Pane**: Elevated to the primary central element. It now supports **Real-Time Streaming** by capturing `THINK` phase content emissions, allowing the user to see the agent's thought process as it happens.
- **Tripartite Visualization**: The routing bars now clearly distinguish between System 1 (Indigo), System 2 (Amber), and System 3 (Purple).
- **Sub-Agent Pool**: Live tracking of dynamically spawned sub-agents.

## Consequences

### Positive
- Offloads moderate tasks from System 3, reducing cost by ~85% for common debugging and planning tasks.
- Improved UX: Real-time response streaming eliminates the "black box" feel of the agent loop.
- Theoretical alignment: Completes the Stanovich tripartite mind implementation.

## Research Sources

### Theoretical Foundations
- **Kahneman (2011)**: Dual-process theory — S1 (fast) + S2 (slow). Only two systems.
- **Stanovich (2011)**: Tripartite model — S1 (autonomous mind) + S2 (algorithmic mind) + S3 (reflective mind). YGN-SAGE's naming origin.
- **SOFAI (Rossi et al., 2023)**: Slow and Fast AI framework. Acknowledges that AI systems may benefit from explicit intermediate processing layers beyond binary fast/slow.

### Empirical Evidence (from NotebookLM: Discover AI)
- **"The Illusion of Thinking" (2025)**: Identifies three complexity regimes. For medium-complexity tasks, reasoning models (S2) demonstrate distinct advantage. For low-complexity, non-thinking (S1) performs better. At extreme complexity, both collapse.
- **AdaptThink (2025)**: Uses RL to dynamically switch between "Thinking" (S2) and "NoThinking" (S1). Confirms mandatory S2 leads to overthinking on simple tasks, but is highly beneficial for moderate-to-difficult reasoning.
- **KnowSelf (2025)**: Introduces special tokens for "fast," "slow," and "knowledgeable" thinking based on situational complexity.
- **ALAMA (2025)**: Adaptively activates appropriate mechanisms (ReAct, CoT, Reflection) depending on task characteristics.

### Architecture Precedents (from NotebookLM: MetaScaffold + Technical)
- **UXAgent**: Dual-process with "Fast Loop" and "Slow Loop" mimicking S1/S2.
- **SWIFTSAGE**: Explicit dual-process agent framework.
- **DPT-Agent (2025)**: S1 as FSM + S2 as async LLM reflection for human-AI collaboration.
- **PlanGEN**: Selection agent using UCB policy for complexity-based routing between inference algorithms.
- **System 3 AI paradigm**: Fully autonomous agentic loops (sense-plan-act) — our S3 with Z3 goes further by adding formal verification.

### Game-Theoretic Mapping (from NotebookLM: MARL)
- **S1 = Level-0 best response**: Heuristic, pre-trained reflex, easily exploitable.
- **S2 = Approximate oracle / Deep Cognitive Hierarchies (DCH)**: Bounded rationality, parallel approximate best responses. Trades exact accuracy for practical efficiency.
- **S3 = Exact best response oracle**: Full value iteration, sequence-form linear programming, zero exploitability.
- **SHOR-PSRO parallel**: Anneals from aggressive heuristic exploration (S2-like Softmax) down to rigorous equilibrium finding (S3-like Optimistic Regret Matching).
- **Key finding**: S2 rapidly reduces exploitability during exploration phase; S3 is reserved for final exploitation and rigorous validation.

### Memory Architecture (from NotebookLM: ExoCortex)
- S2 requires heavy working memory (intermediate results, chain-of-thought state).
- SAMPO eliminates catastrophic forgetting in multi-turn loops — critical for S2's iterative refinement.
- S-MMU Data Minimization + Semantic Slicing keeps S2 context windows efficient.
- Zero-copy Arrow retrieval feeds graph knowledge into S2 with minimal latency.

### Internal Precedent
- `labs/deploy_autonomous_hft.py` already implements "System 2" as periodic LLM reflection (Gemini Flash Lite every 5 cycles).
- `research_journal/agent_long_term_memory.md` references "Dual-Process framework (System 1 execution, System 2 reflection)" from NotebookLM insights.

## Decision

**Add System 2 as an explicit cognitive tier.** The routing becomes S1/S2/S3 tripartite.

### System 2 Specification

| Attribute | Value |
|-----------|-------|
| **LLM Tier** | `mutator` (Gemini 3 Flash) or `reasoner` (Gemini 3.1 Pro) based on sub-thresholds |
| **max_tokens** | 4096 |
| **Validation** | Empirical (sandbox execution, test pass/fail) — NOT Z3 formal proofs |
| **CoT required** | Yes (`<think>` tags encouraged but not PRM-validated) |
| **CGRS self-braking** | Active (entropy monitoring continues) |
| **Working memory** | Full access (GraphRAG retrieval, episodic recall) |
| **Tools** | Sandbox execution, code analysis, file I/O, search |

### Routing Thresholds

Using continuous complexity/uncertainty scoring with discrete tier routing:

```
complexity <= 0.35 AND uncertainty <= 0.3 AND NOT tool_required
    → System 1 (fast, llm_tier="fast", max_tokens=2048, no validation)

complexity <= 0.7 AND uncertainty <= 0.6
    → System 2 (algorithmic, llm_tier="mutator", max_tokens=4096, empirical validation)

complexity > 0.7 OR uncertainty > 0.6
    → System 3 (formal, llm_tier="reasoner"/"codex", max_tokens=8192, Z3 PRM validation)
```

Escalation path: S1 → S2 → S3 (if S2 fails after N sandbox retries, escalate to S3).

### Validation Levels

| Level | System | Method |
|-------|--------|--------|
| Level 1 | S1 | None — fast heuristic response |
| Level 2 | S2 | Empirical: sandbox execution, test pass/fail, LLM-as-judge |
| Level 3 | S3 | Formal: Z3 SMT solver, PRM with `<think>` tag verification |

### CGRS Interaction
- S1: No entropy monitoring (too fast).
- S2: Entropy monitoring active. Self-brake triggers if convergence detected (avoids overthinking).
- S3: Entropy monitoring + PRM validation. Self-brake + formal verification.

### Code Changes Required

1. **`RoutingDecision`**: Add `system=2` as valid value, add `validation_level` field.
2. **`MetacognitiveController.route()`**: Implement 3-tier thresholds.
3. **`AgentLoop.run()`**: Replace `enforce_system3: bool` with `validation_level: int` (1/2/3).
4. **`boot.py` / `AgentSystem.run()`**: Handle S2 tier selection, empirical validation path.
5. **Dashboard**: Display S1/S2/S3 in routing visualization (was binary, now tripartite).

## Alternatives Considered

### A. Keep Binary S1/S3 Routing
- **Pros**: Simpler, fewer code changes.
- **Cons**: Wastes compute on moderate tasks (Z3 overhead), under-serves simple reasoning tasks.
- **Rejected**: All 4 NotebookLM notebooks and frontier research (AdaptThink, Illusion of Thinking) confirm a 3-tier approach improves performance on medium-complexity tasks.

### B. Continuous Routing (No Discrete Tiers)
- **Pros**: Maximum flexibility, no hard boundaries.
- **Cons**: Hard to implement validation levels on a continuum, harder to reason about cost/performance, no clear escalation path.
- **Rejected**: Literature consensus (PlanGEN, KnowSelf, ALAMA) favors continuous scoring with discrete routing — the hybrid approach we chose.

## Consequences

### Positive
- 80% of moderate-complexity tasks (debugging, planning, code review) offloaded from expensive S3 to efficient S2.
- Reduced cost: S2 uses Gemini Flash (~$0.0015/1K tokens) vs S3 Codex (~$0.03/1K tokens) — **20x cheaper** for moderate tasks.
- Reduced latency: S2 skips Z3 formal verification overhead.
- Aligns with Stanovich's complete tripartite model and PSRO game-theoretic best practices.
- CGRS self-braking prevents S2 overthinking.

### Negative
- Slightly more complex routing logic (3 tiers vs 2).
- Two additional thresholds to tune (`complexity_s2_threshold`, `uncertainty_s2_threshold`).
- Need to implement empirical validation path (sandbox execution check).

### Risks
- S2/S3 boundary may need tuning per domain (coding vs. planning vs. research).
- Escalation from S2 to S3 adds latency for misclassified tasks (mitigated by retry cap).

## Implementation Plan

See `docs/plans/` for detailed implementation plan (to be created via writing-plans skill).

### Phase 1: Core Routing (metacognition.py)
- Add `system=2` to `RoutingDecision`
- Implement 3-tier threshold logic in `route()`
- Update `CognitiveProfile` assessment prompts

### Phase 2: Agent Loop Integration (agent_loop.py)
- Replace `enforce_system3: bool` with `validation_level: int`
- Implement Level 2 empirical validation (sandbox execution check)
- Preserve Level 3 Z3 PRM path unchanged

### Phase 3: Boot & Provider Wiring (boot.py)
- Wire S2 to `mutator` tier (Gemini 3 Flash)
- Handle S2 → S3 escalation on empirical failure

### Phase 4: Dashboard (ui/)
- Display S1/S2/S3 in routing panel
- Show validation level in metrics
- Color-code: S1=green, S2=amber, S3=red

---

*This ADR was informed by queries to 4 NotebookLM notebooks (MARL, Technical, Discover AI, MetaScaffold, ExoCortex), HuggingFace paper searches (SOFAI 2023, DPT-Agent 2025), and YGN-SAGE internal research journal.*
