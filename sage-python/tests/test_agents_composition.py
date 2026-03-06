"""Tests for multi-agent composition patterns: Sequential, Parallel, Loop, Handoff."""
from __future__ import annotations

import asyncio

import pytest


class MockRunnable:
    """Lightweight mock that satisfies the agent protocol (has .name and async .run)."""

    def __init__(self, name: str, response: str):
        self.name = name
        self._response = response
        self.last_input: str | None = None

    async def run(self, task: str) -> str:
        self.last_input = task
        return self._response


# ---------------------------------------------------------------------------
# SequentialAgent
# ---------------------------------------------------------------------------

class TestSequentialAgent:
    @pytest.mark.asyncio
    async def test_chains_output_correctly(self):
        """Output of each agent feeds as input to the next."""
        from sage.agents.sequential import SequentialAgent

        a = MockRunnable("step1", "alpha")
        b = MockRunnable("step2", "bravo")
        c = MockRunnable("step3", "charlie")

        seq = SequentialAgent(name="seq", agents=[a, b, c])
        result = await seq.run("start")

        # First agent receives the original task
        assert a.last_input == "start"
        # Second agent receives output of first
        assert b.last_input == "alpha"
        # Third agent receives output of second
        assert c.last_input == "bravo"
        # Final result is the output of the last agent
        assert result == "charlie"

    @pytest.mark.asyncio
    async def test_single_agent(self):
        from sage.agents.sequential import SequentialAgent

        a = MockRunnable("only", "done")
        seq = SequentialAgent(name="seq-single", agents=[a])
        result = await seq.run("go")
        assert result == "done"
        assert a.last_input == "go"

    @pytest.mark.asyncio
    async def test_shared_state_passed(self):
        """shared_state dict is accessible on the SequentialAgent instance."""
        from sage.agents.sequential import SequentialAgent

        state = {"counter": 0}
        seq = SequentialAgent(name="seq-state", agents=[MockRunnable("a", "ok")], shared_state=state)
        assert seq.shared_state is state


# ---------------------------------------------------------------------------
# ParallelAgent
# ---------------------------------------------------------------------------

class TestParallelAgent:
    @pytest.mark.asyncio
    async def test_runs_concurrently_and_aggregates(self):
        """All agents run and default aggregator joins results."""
        from sage.agents.parallel import ParallelAgent

        a = MockRunnable("agent_a", "result_a")
        b = MockRunnable("agent_b", "result_b")

        par = ParallelAgent(name="par", agents=[a, b])
        result = await par.run("task")

        # Both agents received the same task
        assert a.last_input == "task"
        assert b.last_input == "task"

        # Default aggregator format
        assert "[agent_a]: result_a" in result
        assert "[agent_b]: result_b" in result

    @pytest.mark.asyncio
    async def test_custom_aggregator(self):
        """Custom aggregator receives dict and returns combined string."""
        from sage.agents.parallel import ParallelAgent

        a = MockRunnable("x", "10")
        b = MockRunnable("y", "20")

        def sum_agg(results: dict[str, str]) -> str:
            total = sum(int(v) for v in results.values())
            return str(total)

        par = ParallelAgent(name="par-custom", agents=[a, b], aggregator=sum_agg)
        result = await par.run("compute")
        assert result == "30"

    @pytest.mark.asyncio
    async def test_true_concurrent_execution(self):
        """Verify that agents truly run concurrently via asyncio.gather."""
        from sage.agents.parallel import ParallelAgent

        class SlowRunnable:
            def __init__(self, name: str, delay: float, response: str):
                self.name = name
                self._delay = delay
                self._response = response

            async def run(self, task: str) -> str:
                await asyncio.sleep(self._delay)
                return self._response

        a = SlowRunnable("slow_a", 0.1, "a_done")
        b = SlowRunnable("slow_b", 0.1, "b_done")

        par = ParallelAgent(name="par-conc", agents=[a, b])

        start = asyncio.get_event_loop().time()
        result = await par.run("go")
        elapsed = asyncio.get_event_loop().time() - start

        # If sequential it would take ~0.2s; concurrent should take ~0.1s
        assert elapsed < 0.18
        assert "a_done" in result
        assert "b_done" in result


# ---------------------------------------------------------------------------
# LoopAgent
# ---------------------------------------------------------------------------

class TestLoopAgent:
    @pytest.mark.asyncio
    async def test_runs_until_exit_condition(self):
        """Loop terminates when exit_condition returns True."""
        from sage.agents.loop_agent import LoopAgent

        call_count = 0

        class CountingRunnable:
            def __init__(self):
                self.name = "counter"

            async def run(self, task: str) -> str:
                nonlocal call_count
                call_count += 1
                return f"iteration_{call_count}"

        agent = CountingRunnable()
        loop = LoopAgent(
            name="loop-exit",
            agent=agent,
            max_iterations=10,
            exit_condition=lambda r: r == "iteration_3",
        )

        result = await loop.run("go")
        assert result == "iteration_3"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_respects_max_iterations(self):
        """Loop stops at max_iterations even if exit_condition never triggers."""
        from sage.agents.loop_agent import LoopAgent

        a = MockRunnable("looper", "never_done")
        loop = LoopAgent(
            name="loop-max",
            agent=a,
            max_iterations=5,
            exit_condition=lambda r: False,  # never exits
        )
        result = await loop.run("go")
        assert result == "never_done"

    @pytest.mark.asyncio
    async def test_no_exit_condition_runs_max(self):
        """Without exit_condition, runs exactly max_iterations times."""
        from sage.agents.loop_agent import LoopAgent

        call_count = 0

        class CountingRunnable:
            def __init__(self):
                self.name = "counter"

            async def run(self, task: str) -> str:
                nonlocal call_count
                call_count += 1
                return f"step_{call_count}"

        loop = LoopAgent(name="loop-no-cond", agent=CountingRunnable(), max_iterations=4)
        result = await loop.run("go")
        assert call_count == 4
        assert result == "step_4"

    @pytest.mark.asyncio
    async def test_feeds_output_back_as_input(self):
        """Each iteration receives the output of the previous iteration."""
        from sage.agents.loop_agent import LoopAgent

        inputs_seen: list[str] = []

        class TrackingRunnable:
            def __init__(self):
                self.name = "tracker"

            async def run(self, task: str) -> str:
                inputs_seen.append(task)
                return task + "+"

        loop = LoopAgent(name="loop-chain", agent=TrackingRunnable(), max_iterations=3)
        result = await loop.run("x")
        assert inputs_seen == ["x", "x+", "x++"]
        assert result == "x+++"


# ---------------------------------------------------------------------------
# Handoff
# ---------------------------------------------------------------------------

class TestHandoff:
    @pytest.mark.asyncio
    async def test_transfers_to_target(self):
        """Handoff executes the target agent and returns HandoffResult."""
        from sage.agents.handoff import Handoff, HandoffResult

        target = MockRunnable("specialist", "expert_answer")
        h = Handoff(target=target, description="Route to specialist")

        result = await h.execute("question")
        assert isinstance(result, HandoffResult)
        assert result.output == "expert_answer"
        assert result.target_name == "specialist"
        assert target.last_input == "question"

    @pytest.mark.asyncio
    async def test_input_filter_transforms_input(self):
        """input_filter modifies the task before sending to target."""
        from sage.agents.handoff import Handoff

        target = MockRunnable("filtered", "filtered_answer")
        h = Handoff(
            target=target,
            description="With filter",
            input_filter=lambda s: s.upper(),
        )

        result = await h.execute("hello")
        assert target.last_input == "HELLO"
        assert result.output == "filtered_answer"

    @pytest.mark.asyncio
    async def test_on_handoff_callback_fires(self):
        """on_handoff callback is invoked with target_name and task."""
        from sage.agents.handoff import Handoff

        callback_log: list[tuple[str, str]] = []

        def on_handoff_cb(target_name: str, task: str) -> None:
            callback_log.append((target_name, task))

        target = MockRunnable("receiver", "received")
        h = Handoff(
            target=target,
            description="With callback",
            on_handoff=on_handoff_cb,
        )

        await h.execute("payload")
        assert len(callback_log) == 1
        assert callback_log[0] == ("receiver", "payload")

    @pytest.mark.asyncio
    async def test_handoff_without_optional_params(self):
        """Handoff works fine with no input_filter or on_handoff."""
        from sage.agents.handoff import Handoff

        target = MockRunnable("basic", "basic_result")
        h = Handoff(target=target, description="Simple handoff")
        result = await h.execute("task")
        assert result.output == "basic_result"
        assert result.target_name == "basic"
