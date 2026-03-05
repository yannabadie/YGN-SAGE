# System 2 Cognitive Routing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add System 2 (algorithmic mind) as an explicit cognitive tier between System 1 (fast) and System 3 (formal), per ADR-001.

**Architecture:** 3-tier routing (S1/S2/S3) using continuous complexity/uncertainty scoring with discrete thresholds. S2 uses Gemini 3 Flash with empirical validation (no Z3). Replace boolean `enforce_system3` with integer `validation_level` (1/2/3).

**Tech Stack:** Python 3.12+, existing Gemini/Codex providers, Z3 PRM (unchanged for S3), FastAPI dashboard.

---

### Task 1: Update RoutingDecision and Add Validation Level

**Files:**
- Modify: `sage-python/src/sage/strategy/metacognition.py:25-31`
- Test: `sage-python/tests/test_metacognition.py`

**Step 1: Write the failing tests**

Add to `sage-python/tests/test_metacognition.py`:

```python
def test_routing_decision_has_validation_level():
    from sage.strategy.metacognition import RoutingDecision
    rd = RoutingDecision(system=2, llm_tier="mutator", max_tokens=4096, use_z3=False, validation_level=2)
    assert rd.validation_level == 2

def test_route_moderate_to_system2():
    ctrl = MetacognitiveController()
    # Moderate complexity, moderate uncertainty, no tools
    decision = ctrl.route(CognitiveProfile(complexity=0.5, uncertainty=0.4, tool_required=False))
    assert decision.system == 2
    assert decision.llm_tier == "mutator"
    assert decision.validation_level == 2
    assert not decision.use_z3

def test_route_moderate_with_tools_to_system2():
    ctrl = MetacognitiveController()
    decision = ctrl.route(CognitiveProfile(complexity=0.4, uncertainty=0.3, tool_required=True))
    assert decision.system == 2
    assert decision.validation_level == 2
```

**Step 2: Run tests to verify they fail**

Run: `cd sage-python && python -m pytest tests/test_metacognition.py::test_routing_decision_has_validation_level tests/test_metacognition.py::test_route_moderate_to_system2 tests/test_metacognition.py::test_route_moderate_with_tools_to_system2 -v`
Expected: FAIL (validation_level not in RoutingDecision, system=2 never returned)

**Step 3: Update RoutingDecision dataclass**

In `sage-python/src/sage/strategy/metacognition.py`, replace the `RoutingDecision` class (lines 25-31):

```python
@dataclass
class RoutingDecision:
    """Which system and LLM tier to use."""
    system: int           # 1 = fast/intuitive, 2 = algorithmic/deliberate, 3 = formal/verified
    llm_tier: str         # fast, mutator, reasoner, codex
    max_tokens: int
    use_z3: bool          # Whether to validate with Z3 PRM
    validation_level: int = 1  # 1=none, 2=empirical, 3=formal(Z3)
```

**Step 4: Run tests to verify they still fail (routing not yet 3-tier)**

Run: `cd sage-python && python -m pytest tests/test_metacognition.py::test_routing_decision_has_validation_level -v`
Expected: PASS (dataclass updated), the routing tests still FAIL

**Step 5: Commit**

```bash
git add sage-python/src/sage/strategy/metacognition.py sage-python/tests/test_metacognition.py
git commit -m "feat(metacognition): add validation_level to RoutingDecision, add S2 test stubs"
```

---

### Task 2: Implement 3-Tier Routing Logic

**Files:**
- Modify: `sage-python/src/sage/strategy/metacognition.py:67-111`
- Test: `sage-python/tests/test_metacognition.py`

**Step 1: Write additional boundary tests**

Add to `sage-python/tests/test_metacognition.py`:

```python
def test_route_low_complexity_to_system1():
    ctrl = MetacognitiveController()
    decision = ctrl.route(CognitiveProfile(complexity=0.2, uncertainty=0.1, tool_required=False))
    assert decision.system == 1
    assert decision.validation_level == 1
    assert decision.llm_tier == "fast"

def test_route_high_complexity_to_system3():
    ctrl = MetacognitiveController()
    decision = ctrl.route(CognitiveProfile(complexity=0.8, uncertainty=0.7, tool_required=True))
    assert decision.system == 3
    assert decision.validation_level == 3
    assert decision.use_z3

def test_route_high_complexity_codex_tier():
    ctrl = MetacognitiveController()
    decision = ctrl.route(CognitiveProfile(complexity=0.9, uncertainty=0.5, tool_required=False))
    assert decision.system == 3
    assert decision.llm_tier == "codex"

def test_route_s2_uses_reasoner_for_higher_complexity():
    ctrl = MetacognitiveController()
    decision = ctrl.route(CognitiveProfile(complexity=0.65, uncertainty=0.55, tool_required=False))
    assert decision.system == 2
    assert decision.llm_tier == "reasoner"
```

**Step 2: Implement 3-tier routing in MetacognitiveController**

Replace the `__init__` and `route` methods in `metacognition.py`:

```python
class MetacognitiveController:
    """Routes tasks between System 1, 2, and 3 (Stanovich tripartite model).

    System 1: Fast/intuitive (Gemini Flash Lite, no validation)
    System 2: Algorithmic/deliberate (Gemini Flash/Pro, empirical validation)
    System 3: Formal/verified (Codex/Reasoner, Z3 PRM validation)

    Self-braking (CGRS): monitors output entropy to detect convergence.
    """

    def __init__(
        self,
        s1_complexity_ceil: float = 0.35,
        s1_uncertainty_ceil: float = 0.3,
        s3_complexity_floor: float = 0.7,
        s3_uncertainty_floor: float = 0.6,
        brake_window: int = 3,
        brake_entropy_threshold: float = 0.15,
    ):
        self.s1_complexity_ceil = s1_complexity_ceil
        self.s1_uncertainty_ceil = s1_uncertainty_ceil
        self.s3_complexity_floor = s3_complexity_floor
        self.s3_uncertainty_floor = s3_uncertainty_floor
        self.brake_window = brake_window
        self.brake_entropy_threshold = brake_entropy_threshold
        self._entropy_history: deque[float] = deque(maxlen=10)
        self._llm_available: bool | None = None

    def route(self, profile: CognitiveProfile) -> RoutingDecision:
        """Decide which cognitive system to engage (S1/S2/S3)."""
        c, u = profile.complexity, profile.uncertainty

        # System 3: high complexity OR high uncertainty
        if c > self.s3_complexity_floor or u > self.s3_uncertainty_floor:
            tier = "codex" if c > 0.8 else "reasoner"
            return RoutingDecision(
                system=3, llm_tier=tier,
                max_tokens=8192, use_z3=True, validation_level=3,
            )

        # System 1: low complexity AND low uncertainty AND no tools
        if (c <= self.s1_complexity_ceil
                and u <= self.s1_uncertainty_ceil
                and not profile.tool_required):
            return RoutingDecision(
                system=1, llm_tier="fast",
                max_tokens=2048, use_z3=False, validation_level=1,
            )

        # System 2: everything in between
        tier = "reasoner" if c > 0.55 else "mutator"
        return RoutingDecision(
            system=2, llm_tier=tier,
            max_tokens=4096, use_z3=False, validation_level=2,
        )
```

**Step 3: Update existing tests for new constructor params**

The old `MetacognitiveController()` call with no args still works (defaults changed but compatible). Update `test_route_simple_to_system1`:

```python
def test_route_simple_to_system1():
    ctrl = MetacognitiveController()
    decision = ctrl.route(CognitiveProfile(complexity=0.1, uncertainty=0.1, tool_required=False))
    assert decision.system == 1
    assert decision.llm_tier == "fast"
    assert decision.validation_level == 1
```

Update `test_route_complex_to_system3`:

```python
def test_route_complex_to_system3():
    ctrl = MetacognitiveController()
    decision = ctrl.route(CognitiveProfile(complexity=0.9, uncertainty=0.8, tool_required=True))
    assert decision.system == 3
    assert decision.llm_tier in ("reasoner", "codex")
    assert decision.validation_level == 3
```

**Step 4: Run all metacognition tests**

Run: `cd sage-python && python -m pytest tests/test_metacognition.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add sage-python/src/sage/strategy/metacognition.py sage-python/tests/test_metacognition.py
git commit -m "feat(metacognition): implement 3-tier S1/S2/S3 routing with validation_level"
```

---

### Task 3: Update Docstring and LLM Routing Prompt

**Files:**
- Modify: `sage-python/src/sage/strategy/metacognition.py:1-5,58-64`

**Step 1: Update module docstring**

```python
"""Metacognitive Controller: Stanovich tripartite S1/S2/S3 routing + self-braking.

System 1 (autonomous mind): Gemini Flash Lite, no validation (~$0.0005/call, <1s).
System 2 (algorithmic mind): Gemini Flash/Pro, empirical validation (~$0.0015/1K).
System 3 (reflective mind): Codex/Reasoner, Z3 PRM formal verification (~$0.03/1K).

Uses Gemini 2.5 Flash Lite for LLM-based task assessment.
Falls back to heuristic if GOOGLE_API_KEY is not set.
"""
```

**Step 2: No tests needed (docstring only)**

**Step 3: Commit**

```bash
git add sage-python/src/sage/strategy/metacognition.py
git commit -m "docs(metacognition): update docstring for tripartite S1/S2/S3 model"
```

---

### Task 4: Replace enforce_system3 with validation_level in AgentConfig

**Files:**
- Modify: `sage-python/src/sage/agent.py:17-28`
- Modify: `sage-python/src/sage/agent_loop.py:144-149,207-222`
- Test: `sage-python/tests/test_metacognition.py`

**Step 1: Write failing test**

Add to `sage-python/tests/test_metacognition.py`:

```python
def test_agent_config_validation_level():
    from sage.agent import AgentConfig
    from sage.llm.base import LLMConfig
    cfg = AgentConfig(
        name="test",
        llm=LLMConfig(provider="mock", model="mock"),
        validation_level=2,
    )
    assert cfg.validation_level == 2
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_metacognition.py::test_agent_config_validation_level -v`
Expected: FAIL (no validation_level field)

**Step 3: Update AgentConfig**

In `sage-python/src/sage/agent.py`, replace the `enforce_system3` field:

```python
@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    llm: LLMConfig
    system_prompt: str = "You are a helpful AI assistant."
    max_steps: int = 100
    tools: list[str] | None = None
    use_docker_sandbox: bool = False
    snapshot_to_restore: str | None = None
    validation_level: int = 1  # 1=none(S1), 2=empirical(S2), 3=formal/Z3(S3)
```

**Step 4: Update AgentLoop to use validation_level**

In `sage-python/src/sage/agent_loop.py`, replace the system prompt augmentation (lines 144-149):

```python
        system_prompt = self.config.system_prompt
        if self.config.validation_level >= 3:
            system_prompt += (
                "\n\nCRITICAL: Use <think>...</think> tags to reason step-by-step. "
                "Your reasoning is evaluated by a Process Reward Model."
            )
        elif self.config.validation_level >= 2:
            system_prompt += (
                "\n\nUse step-by-step reasoning to solve this task. "
                "Show your work clearly."
            )
```

Replace the Z3 PRM validation block (lines 207-222):

```python
            # Validation based on cognitive tier
            if self.config.validation_level >= 3 and content:
                # Level 3: Formal Z3 PRM verification
                r_path, details = self.prm.calculate_r_path(content)
                self._emit(LoopPhase.THINK, r_path=r_path, details=details)
                if r_path < 0.0 and "error" in details:
                    self._prm_retries += 1
                    if self._prm_retries <= self._max_prm_retries:
                        messages.append(Message(
                            role=Role.USER,
                            content="SYSTEM: Use <think> tags for structured reasoning.",
                        ))
                        continue
                    log.warning("PRM retry limit reached, accepting response without <think> tags.")
                else:
                    self._prm_retries = 0
            elif self.config.validation_level >= 2 and content:
                # Level 2: Empirical validation (log entropy, no Z3)
                self._emit(LoopPhase.THINK, validation="empirical", entropy=round(entropy, 3))
```

**Step 5: Update Agent.run() in agent.py similarly**

In `sage-python/src/sage/agent.py`, replace `self.config.enforce_system3` references (lines 80-81, 117-126):

Line 80-81:
```python
        if self.config.validation_level >= 3:
            system_prompt += "\n\nCRITICAL: You MUST use <think>...</think> tags to reason step-by-step before answering. Your reasoning is evaluated by a Process Reward Model."
        elif self.config.validation_level >= 2:
            system_prompt += "\n\nUse step-by-step reasoning to solve this task. Show your work clearly."
```

Lines 117-126:
```python
                if self.config.validation_level >= 3 and response.content:
                    r_path, details = self.prm.calculate_r_path(response.content)
                    if r_path < 0.0:
                        warning = "SYSTEM: You failed to use <think> tags. You must break down the problem structurally before answering."
                        self._messages.append(Message(role=Role.USER, content=warning))
                        self.working_memory.add_event("SYSTEM_WARNING", warning)
                        continue
                    else:
                        self.working_memory.add_event("REWARD", f"R_path={r_path:.2f} Verifiable={details['verifiable_ratio']:.0%}")
```

**Step 6: Run all tests**

Run: `cd sage-python && python -m pytest tests/ -v --tb=short`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add sage-python/src/sage/agent.py sage-python/src/sage/agent_loop.py sage-python/tests/test_metacognition.py
git commit -m "feat(agent): replace enforce_system3 with validation_level (1/2/3)"
```

---

### Task 5: Update boot.py to Wire S2

**Files:**
- Modify: `sage-python/src/sage/boot.py:44-81,125-140`

**Step 1: Update AgentSystem.run() for 3-tier routing**

Replace the enforce_system3 logic in `boot.py` lines 49-60:

```python
        # 2. Apply routing decision
        current_provider = self.agent_loop.config.llm.provider

        # Set validation level based on routing decision
        actual_provider = self.agent_loop.config.llm.provider
        if actual_provider == "codex":
            # Codex CLI refuses <think> tags — it reasons internally
            self.agent_loop.config.validation_level = 1
        else:
            self.agent_loop.config.validation_level = decision.validation_level
```

**Step 2: Update boot_agent_system() default config**

Replace line 139:

```python
        validation_level=1 if llm_config.provider == "codex" else 2,
```

**Step 3: Run full test suite**

Run: `cd sage-python && python -m pytest tests/ -v --tb=short`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add sage-python/src/sage/boot.py
git commit -m "feat(boot): wire S2 routing, validation_level in AgentSystem"
```

---

### Task 6: Update Dashboard State for S1/S2/S3

**Files:**
- Modify: `ui/app.py:96-117`

**Step 1: Update dashboard_state default**

In `ui/app.py`, change line 111:

```python
    "metacognitive_system": 1,     # 1, 2, or 3
```

No other backend changes needed — `meta["system"]` already passes through whatever value metacognition emits (now 1, 2, or 3).

**Step 2: Add validation_level to state tracking**

After line 152 (`"metacognitive_system"` assignment), add:

```python
    # Validation level (from routing decision)
    if "validation_level" in meta:
        dashboard_state["validation_level"] = meta["validation_level"]
```

Add to `dashboard_state` initial dict:

```python
    "validation_level": 1,         # 1=none, 2=empirical, 3=formal
```

**Step 3: Emit validation_level from agent_loop PERCEIVE event**

In `sage-python/src/sage/agent_loop.py`, after line 136 (`perceive_meta["use_z3"]`), add:

```python
            perceive_meta["validation_level"] = decision.validation_level
```

**Step 4: Commit**

```bash
git add ui/app.py sage-python/src/sage/agent_loop.py
git commit -m "feat(dashboard): track validation_level and S1/S2/S3 in backend state"
```

---

### Task 7: Update Dashboard HTML for 3-Tier Routing Display

**Files:**
- Modify: `ui/static/index.html`

**Step 1: Replace 2-bar routing panel with 3-bar (lines 570-600)**

Replace the "Metacognitive routing" card:

```html
            <!-- Metacognitive routing -->
            <div class="card p-3">
                <h3 class="text-xs font-semibold uppercase tracking-wider mb-3" style="color: var(--text-muted);">Cognitive Routing</h3>
                <div class="flex flex-col gap-2">
                    <!-- System 1 bar -->
                    <div class="card-inset p-2.5">
                        <div class="flex items-center justify-between mb-1.5">
                            <span class="text-xs font-semibold" style="color: var(--accent);">System 1</span>
                            <span class="text-xs font-mono" style="color: var(--text-muted);">Fast / Heuristic</span>
                        </div>
                        <div class="w-full rounded-full h-1.5" style="background: var(--border);">
                            <div id="bar-s1" class="h-1.5 rounded-full transition-all duration-500" style="width: 100%; background: var(--accent);"></div>
                        </div>
                    </div>
                    <!-- System 2 bar -->
                    <div class="card-inset p-2.5">
                        <div class="flex items-center justify-between mb-1.5">
                            <span class="text-xs font-semibold" style="color: #f59e0b;">System 2</span>
                            <span class="text-xs font-mono" style="color: var(--text-muted);">Algorithmic / Empirical</span>
                        </div>
                        <div class="w-full rounded-full h-1.5" style="background: var(--border);">
                            <div id="bar-s2" class="h-1.5 rounded-full transition-all duration-500" style="width: 0%; background: #f59e0b;"></div>
                        </div>
                    </div>
                    <!-- System 3 bar -->
                    <div class="card-inset p-2.5">
                        <div class="flex items-center justify-between mb-1.5">
                            <span class="text-xs font-semibold" style="color: var(--phase-perceive);">System 3</span>
                            <span class="text-xs font-mono" style="color: var(--text-muted);">Formal / Z3</span>
                        </div>
                        <div class="w-full rounded-full h-1.5" style="background: var(--border);">
                            <div id="bar-s3" class="h-1.5 rounded-full transition-all duration-500" style="width: 0%; background: var(--phase-perceive);"></div>
                        </div>
                    </div>
                    <!-- Active indicator -->
                    <div class="flex items-center justify-center mt-1">
                        <span class="text-xs" style="color: var(--text-muted);">Active:</span>
                        <span id="active-system" class="ml-2 text-sm font-mono font-bold" style="color: var(--accent);">System 1</span>
                    </div>
                </div>
            </div>
```

**Step 2: Add bar-s2 to DOM references (after line 707)**

```javascript
            barS2:           $('bar-s2'),
```

**Step 3: Update system routing JS (replace lines 1037-1051)**

```javascript
            // System routing (S1/S2/S3)
            if (meta.system !== undefined) {
                const sys = meta.system;
                const colors = { 1: '#38bdf8', 2: '#f59e0b', 3: '#818cf8' };
                const color = colors[sys] || '#38bdf8';
                dom.valSystem.textContent = 'S' + sys;
                dom.valSystem.style.color = color;
                dom.activeSystem.textContent = 'System ' + sys;
                dom.activeSystem.style.color = color;
                // Highlight active bar, dim others
                dom.barS1.style.width = sys === 1 ? '100%' : '15%';
                dom.barS2.style.width = sys === 2 ? '100%' : '15%';
                dom.barS3.style.width = sys === 3 ? '100%' : '15%';
            }
```

**Step 4: Update state poll handler (replace lines 1185-1188)**

```javascript
                    if (state.metacognitive_system !== undefined) {
                        const sys = state.metacognitive_system;
                        const colors = { 1: '#38bdf8', 2: '#f59e0b', 3: '#818cf8' };
                        const color = colors[sys] || '#38bdf8';
                        dom.valSystem.textContent = 'S' + sys;
                        dom.valSystem.style.color = color;
                        dom.activeSystem.textContent = 'System ' + sys;
                        dom.activeSystem.style.color = color;
                        dom.barS1.style.width = sys === 1 ? '100%' : '15%';
                        dom.barS2.style.width = sys === 2 ? '100%' : '15%';
                        dom.barS3.style.width = sys === 3 ? '100%' : '15%';
                    }
```

**Step 5: Update reset handler (line 1308)**

```javascript
                    dom.valSystem.textContent = 'S1';
                    dom.valSystem.style.color = '#38bdf8';
                    dom.barS1.style.width = '100%';
                    dom.barS2.style.width = '0%';
                    dom.barS3.style.width = '0%';
```

**Step 6: Commit**

```bash
git add ui/static/index.html
git commit -m "feat(dashboard): 3-tier S1/S2/S3 cognitive routing display with color coding"
```

---

### Task 8: Run Full Test Suite and Verify

**Files:**
- All modified files

**Step 1: Run full test suite**

Run: `cd sage-python && python -m pytest tests/ -v --tb=short`
Expected: ALL PASS (59+ tests)

**Step 2: Run ruff linter**

Run: `cd sage-python && ruff check src/sage/strategy/metacognition.py src/sage/agent.py src/sage/agent_loop.py src/sage/boot.py`
Expected: No errors

**Step 3: Visual test dashboard**

Run: `cd C:/Code/YGN-SAGE && python ui/app.py`
Open: `http://localhost:8000`
Verify: 3 routing bars visible (S1 green, S2 amber, S3 purple)

**Step 4: Final commit with all changes**

```bash
git add -A
git commit -m "feat(cognitive-routing): complete S1/S2/S3 tripartite routing per ADR-001

- metacognition.py: 3-tier routing with validation_level (1/2/3)
- agent.py: replace enforce_system3 with validation_level
- agent_loop.py: tiered validation (none/empirical/formal)
- boot.py: wire S2 to mutator tier
- dashboard: 3-bar routing display with color coding
- ADR: docs/ADR-cognitive-routing-system2.md

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Add `validation_level` to `RoutingDecision` | metacognition.py, tests |
| 2 | Implement 3-tier routing logic | metacognition.py, tests |
| 3 | Update docstrings | metacognition.py |
| 4 | Replace `enforce_system3` with `validation_level` | agent.py, agent_loop.py, tests |
| 5 | Wire S2 in boot.py | boot.py |
| 6 | Dashboard backend state | app.py, agent_loop.py |
| 7 | Dashboard HTML 3-bar display | index.html |
| 8 | Full test suite + verify | all files |
