"""Candidate: override sink-node prompts for code-patch tasks.

Hypothesis:
The Rust template store hardcodes a generic sink prompt that asks for a
"definitive answer" and says nothing about preserving unified diffs. On
SWE-bench-style tasks that final node is often the last hop before patch
extraction, so making it explicitly forward or salvage a fenced diff
should raise the real-patch rate and reduce empty/non-diff finals.
"""
from __future__ import annotations

from typing import Any

from reference_examples.ygn_sage.sage_candidate import SageCandidate

_GENERIC_SINK_PREFIX = "You are the final synthesizer."
_SINK_ROLE_KEYWORDS = (
    "synthesizer",
    "aggregator",
    "output_formatter",
    "formatter",
    "verifier",
    "judge",
    "mixer",
    "solver",
)

_CODE_SINK_PROMPT = """\
You are the FINAL PATCH SYNTHESIZER for a code-repair multi-agent pipeline.

Your job is to emit the best unified diff patch from predecessor outputs.

Priority order:
1. If any predecessor already produced a complete diff, forward that diff verbatim.
2. If multiple predecessors produced diffs, choose the one with the strongest source grounding: real file paths, concrete hunks, and evidence of repository inspection.
3. If no predecessor produced a full diff but one produced near-complete patch content, salvage it into a single unified diff.

Hard rules:
- Output exactly one fenced ```diff block and nothing else.
- Preserve paths, hunk headers, context lines, and patch text exactly whenever a predecessor already gave them.
- Do not summarize, explain, apologize, or emit verification labels.
- Do not emit prose before or after the diff block.
- If you must salvage an incomplete patch, prefer a minimal patch over a broad speculative rewrite.
"""


def _looks_like_code_patch_task(task: str) -> bool:
    if not isinstance(task, str):
        return False
    lower = task.lower()
    return (
        "unified diff patch" in lower
        or "```diff" in lower
        or "diff --git" in lower
        or ("issue description" in lower and "repository" in lower)
        or "pytest" in lower
    )


def _is_sink_node(role: str, prompt: str) -> bool:
    role_lower = role.lower() if isinstance(role, str) else ""
    prompt_text = prompt.strip() if isinstance(prompt, str) else ""
    return (
        any(keyword in role_lower for keyword in _SINK_ROLE_KEYWORDS)
        or prompt_text.startswith(_GENERIC_SINK_PREFIX)
    )


class SinkDiffPromptCandidate(SageCandidate):
    name = "sink_diff_prompt"
    hypothesis = (
        "Override the generic final-node prompt on code tasks so the last topology "
        "hop forwards or salvages a fenced unified diff instead of collapsing the "
        "pipeline output into a generic final answer."
    )
    axis = "prompts"

    def build_system(self, hints: dict[str, Any] | None = None) -> Any:
        from sage.boot import boot_agent_system
        from sage.topology import runner as runner_mod

        system = boot_agent_system()
        if getattr(system, "pipeline", None) is None:
            return system

        if getattr(runner_mod.TopologyRunner, "_meta_sink_diff_prompt_installed", False):
            return system

        original = runner_mod.TopologyRunner._execute_node_via_agent_loop

        async def _patched_execute_node_via_agent_loop(
            self, node_idx: int, task: str, context_override: str | None = None,
        ) -> str:
            node = self.graph.get_node(node_idx)
            role = getattr(node, "role", f"node-{node_idx}")
            custom_prompt = getattr(node, "prompt", "")

            if not (_looks_like_code_patch_task(task) and _is_sink_node(role, custom_prompt)):
                return await original(self, node_idx, task, context_override)

            node_model_id = getattr(node, "model_id", "")
            if node_model_id and self._provider_pool:
                provider, config = self._provider_pool.resolve(node_model_id)
            else:
                provider, config = self._llm, self._config

            system_prompt = _CODE_SINK_PROMPT
            if self._harness:
                if self._harness.prompts.global_prefix:
                    system_prompt = self._harness.prompts.global_prefix + "\n" + system_prompt
                if self._harness.prompts.global_suffix:
                    system_prompt = system_prompt + "\n" + self._harness.prompts.global_suffix
            system_prompt = self._maybe_planner_injection(node_idx, system_prompt)

            loop = self._agent_loop_factory(
                node_role=role,
                node_name=f"node-{node_idx}-{role}",
                llm_provider=provider,
                llm_config=config,
                system_prompt=system_prompt,
            )

            context = (
                context_override
                if context_override is not None
                else self._gather_predecessor_context(node_idx)
            )
            if context:
                full_task = (
                    f"## Previous agent output:\n{context}\n\n"
                    f"## Task:\n{task}"
                )
            else:
                full_task = task

            result = await loop.run(full_task)
            self._node_outputs[node_idx] = result
            self.tool_call_count += int(getattr(loop, "tool_call_count", 0) or 0)
            self.tool_turn_count += int(getattr(loop, "tool_turn_count", 0) or 0)
            node_commands = list(getattr(loop, "executed_commands", []) or [])
            if node_commands:
                self.executed_commands.extend(f"[{role}] {command}" for command in node_commands)

            if self._controller and result:
                try:
                    node_ctx = {
                        "node_idx": node_idx,
                        "latency_ms": 0.0,
                        "model_id": getattr(node, "model_id", ""),
                        "output_length": len(result),
                        "axis_hint": self._axis_hint,
                    }
                    self._controller.evaluate_and_decide(
                        node_idx=node_idx,
                        result=result,
                        task=task,
                        topology=self.graph,
                        ctx=node_ctx,
                    )
                except Exception:
                    pass

            return result

        runner_mod.TopologyRunner._execute_node_via_agent_loop = _patched_execute_node_via_agent_loop
        runner_mod.TopologyRunner._meta_sink_diff_prompt_installed = True
        runner_mod.TopologyRunner._meta_sink_diff_prompt_original = original
        return system


CANDIDATE = SinkDiffPromptCandidate()
