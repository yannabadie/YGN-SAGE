"""Tool registry for managing available tools."""
from __future__ import annotations

from sage.tools.base import Tool


class ToolRegistry:
    """Registry for managing tools available to agents."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._usage: dict[str, dict] = {}  # name -> {usage_count, success_count, source}

    def register(self, tool: Tool, *, replace: bool = False) -> None:
        """Register a tool.

        Phase 1.5c+ cycle-13 K (cgpro VERIFY 2026-05-06): registration
        resolves the tool's ToolCapability via
        `sage.policy.manifest.resolve_tool_capability`, whose runtime
        chain is:

            explicit `tool.capability` -> class default -> raise

        The built-in manifest is documentation/audit only and is NOT
        consulted at runtime to accept unlabeled tools. If the chain
        falls through to "raise", `ToolPolicyDeclarationError` is
        raised — there is intentionally NO default-tag-dangerous
        fallback per cgpro DESIGN trap "creates an illusion of
        security". The resolved capability is normalised back onto
        `tool.capability` so downstream consumers (Tool.execute,
        audit CLI) can rely on a non-None, enum-valued attribute.

        Phase 1.5b (cgpro VERIFY 2026-05-06 EDIT_REQUIRED): adds a
        duplicate-name guard. Registering a tool whose name is already
        taken raises `ToolPolicyDeclarationError` unless the caller
        passes `replace=True`. Closes the silent-overwrite vector
        where an external impostor could replace a trusted built-in
        (e.g. `read_file`) with its own handler AFTER the trusted
        version had been registered. Tests / forge paths that want
        to swap a tool by name pass `replace=True` to make the intent
        visible at the call site.
        """
        from sage.policy.errors import ToolPolicyDeclarationError
        from sage.policy.manifest import resolve_tool_capability

        # `resolve_tool_capability` raises ToolPolicyDeclarationError on
        # an unresolvable tool — propagate so the boot path fails fast.
        resolved = resolve_tool_capability(tool)
        # Normalise the resolved enum back onto the Tool so all later
        # code paths (Tool.execute, audit CLI) see the same value even
        # when the tool was originally constructed with `capability=None`
        # and the class default supplied the resolution.
        tool.capability = resolved

        if not replace and tool.spec.name in self._tools:
            raise ToolPolicyDeclarationError(
                f"Tool {tool.spec.name!r} is already registered. Refusing "
                f"silent overwrite: pass `replace=True` to swap an existing "
                f"registration explicitly. Phase 1.5b anti-spoof guard: "
                f"this prevents an external impostor from quietly replacing "
                f"a trusted built-in after boot."
            )

        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def search(self, query: str) -> list[Tool]:
        """Search tools by name or description."""
        query_lower = query.lower()
        return [
            tool
            for tool in self._tools.values()
            if query_lower in tool.spec.name.lower()
            or query_lower in tool.spec.description.lower()
        ]

    def describe_for_prompt(
        self,
        names: list[str] | None = None,
        max_desc_chars: int = 200,
    ) -> str:
        """Render a Markdown block listing registered tools for prompt inclusion.

        Single source of truth for "what tools does the agent have?" — any
        tool registered at boot (including runtime-forged ToolForge tools)
        is auto-documented without bench templates needing to hand-roll
        their own lists.

        Motivated by the 2026-04-21 ExoCortex usage audit
        (``docs/audits/2026-04-21-exocortex-swebench-usage.md``): the
        search_exocortex tool was registered in the structured tool-call
        schema but invisible in the prompt because SWE-bench's
        ``_TASK_TEMPLATE`` only named ``execute_bash``. The agent called
        ``search_exocortex`` zero times across v13/v15/v17 smoke runs
        (157/157 tool calls were bash). Auto-injecting the tool list into
        the system prompt removes the per-bench anti-affordance.

        Args:
            names: Filter to a subset of tool names. None = all.
            max_desc_chars: Truncate each description to this length to
                bound token use.

        Returns:
            Empty string if no tools registered. Otherwise a Markdown
            section starting with "## Available Tools" and one bullet
            per tool.
        """
        tools = (
            [self._tools[n] for n in names if n in self._tools]
            if names is not None
            else list(self._tools.values())
        )
        if not tools:
            return ""
        lines = ["## Available Tools", ""]
        for tool in sorted(tools, key=lambda t: t.spec.name):
            desc = (tool.spec.description or "").strip().replace("\n", " ")
            if max_desc_chars > 0 and len(desc) > max_desc_chars:
                desc = desc[: max_desc_chars - 3] + "..."
            lines.append(f"- **{tool.spec.name}** — {desc}")
        return "\n".join(lines) + "\n"

    def get_tool_defs(self, names: list[str] | None = None) -> list:
        """Get tool definitions in OpenAI function-calling format."""
        specs = (
            [t.spec for t in self._tools.values()]
            if names is None
            else [self._tools[n].spec for n in names if n in self._tools]
        )
        return [
            {
                "type": "function",
                "function": {
                    "name": s.name,
                    "description": s.description,
                    "parameters": s.parameters,
                },
            }
            for s in specs
        ]

    # ── Usage tracking (ToolForge axis) ────────────────────────────────────

    def record_usage(self, name: str, success: bool = True) -> None:
        """Record a tool invocation for usage tracking."""
        if name not in self._usage:
            self._usage[name] = {"usage_count": 0, "success_count": 0, "source": "builtin"}
        self._usage[name]["usage_count"] += 1
        if success:
            self._usage[name]["success_count"] += 1

    def get_usage(self, name: str) -> dict:
        """Get usage stats for a tool. Returns default dict if unknown."""
        return self._usage.get(name, {"usage_count": 0, "success_count": 0, "source": "builtin"})

    def mark_source(
        self,
        name: str,
        source: str,
        approved_by: str | None = None,
    ) -> None:
        """Mark the origin of a tool (builtin, forged, user)."""
        if name not in self._usage:
            self._usage[name] = {
                "usage_count": 0,
                "success_count": 0,
                "source": source,
            }
        else:
            self._usage[name]["source"] = source

        if source == "forged":
            self._usage[name]["approved_by"] = approved_by
        else:
            self._usage[name].pop("approved_by", None)
