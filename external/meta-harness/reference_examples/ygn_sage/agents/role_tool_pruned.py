"""Candidate: prune per-role tool exposure to a valid SWE-bench subset.

Hypothesis:
The current per-node factory gives planner/coder roles the full registry
(including tool creation, recursive self-invocation, and research tools)
while verifier/formatter filters reference stale tool names that no longer
exist. On SWE-bench this widens the action space for the nodes that must
stay repository-grounded and wastes scarce tool/step budget. Constraining
roles to a small set of real, relevant tools should raise the real-patch
rate and reduce sentinels.
"""
from __future__ import annotations

from typing import Any

from reference_examples.ygn_sage.sage_candidate import SageCandidate

_READONLY_MEMORY = (
    "retrieve_context",
    "summarize_context",
    "search_memory",
)
_PLANNER_TOOLS = ("execute_bash", *_READONLY_MEMORY)
_CODER_TOOLS = ("execute_bash", *_READONLY_MEMORY)
_VERIFIER_TOOLS = ("execute_bash", *_READONLY_MEMORY)
_SYNTH_TOOLS = ("retrieve_context", "summarize_context")


def _tools_for_role(role: str) -> list[str] | None:
    role_lower = role.lower() if isinstance(role, str) else ""
    if any(token in role_lower for token in ("planner", "input_processor", "decomposer")):
        return list(_PLANNER_TOOLS)
    if any(token in role_lower for token in ("coder", "actor", "worker")):
        return list(_CODER_TOOLS)
    if any(token in role_lower for token in ("verif", "validator", "critic", "judge")):
        return list(_VERIFIER_TOOLS)
    if any(token in role_lower for token in ("synthesizer", "aggregat", "format", "output")):
        return list(_SYNTH_TOOLS)
    return None


class RoleToolPrunedCandidate(SageCandidate):
    name = "role_tool_pruned"
    hypothesis = (
        "Constrain SWE-bench topology nodes to a small set of valid repository and "
        "memory tools instead of the full registry; this should reduce wasted tool "
        "turns and prevent stale tool-filter mismatches from starving verifier and "
        "formatter roles of the intended read-only context."
    )
    axis = "tools"

    def build_system(self, hints: dict[str, Any] | None = None) -> Any:
        from sage.boot import boot_agent_system
        from sage import agent_loop_factory as factory_mod

        system = boot_agent_system()

        if not getattr(factory_mod, "_meta_role_tool_pruned_installed", False):
            original = factory_mod.create_node_agent_loop

            def _patched_create_node_agent_loop(
                node_role: str,
                node_name: str,
                llm_provider: Any,
                llm_config: Any,
                tool_registry: Any,
                system_prompt: str,
                system_level: int,
                task_domain: str = "",
                on_event: Any = None,
            ) -> Any:
                loop = original(
                    node_role=node_role,
                    node_name=node_name,
                    llm_provider=llm_provider,
                    llm_config=llm_config,
                    tool_registry=tool_registry,
                    system_prompt=system_prompt,
                    system_level=system_level,
                    task_domain=task_domain,
                    on_event=on_event,
                )
                selected_tools = _tools_for_role(node_role)
                if selected_tools is not None:
                    loop.config.tools = selected_tools
                return loop

            factory_mod.create_node_agent_loop = _patched_create_node_agent_loop
            factory_mod._meta_role_tool_pruned_installed = True
            factory_mod._meta_role_tool_pruned_original = original

        # If the pipeline falls back to direct single-agent execution, keep the
        # tool surface similarly focused for the same benchmark setting.
        if getattr(system, "agent_loop", None) is not None:
            system.agent_loop.config.tools = list(_CODER_TOOLS)

        return system


CANDIDATE = RoleToolPrunedCandidate()
