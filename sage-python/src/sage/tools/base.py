"""Base tool types and decorator."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, TYPE_CHECKING

from sage.llm.base import ToolDef

if TYPE_CHECKING:
    from pydantic import BaseModel

    from sage.policy.tool_policy import ToolCapability

log = logging.getLogger(__name__)


# AUDIT3 #17 / A14 — opt-in ToolResult v2 Pydantic output validator.
# Default behaviour unchanged: output remains a free-form string and
# no validation happens unless the caller attaches a schema via
# `Tool(..., output_schema=MySchema)` AND either calls
# `result.validate_output(MySchema)` or sets SAGE_TOOLRESULT_VALIDATE=1.
#
# Example:
#     from pydantic import BaseModel
#
#     class FileReadResult(BaseModel):
#         path: str
#         size: int
#         content: str
#
#     tool = Tool(spec=spec, handler=read_file, output_schema=FileReadResult)
#     r = await tool.execute({"path": "x.txt"})
#     parsed = r.validate_output(FileReadResult)   # None on failure
#
# `SAGE_TOOLRESULT_VALIDATE=1` only raises when a schema is attached
# AND validation fails — default off preserves back-compat.


def _env_validate_strict() -> bool:
    return os.environ.get("SAGE_TOOLRESULT_VALIDATE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass
class ToolResult:
    output: str
    is_error: bool = False
    # When set, the tool's attached output_schema instance after
    # `validate_output` succeeds. None until validated.
    validated: Any | None = field(default=None, repr=False)

    def validate_output(self, schema: type["BaseModel"]) -> "BaseModel | None":
        """Parse `output` as JSON and validate against `schema`.

        Returns the validated instance on success, None on JSON decode
        failure. Raises `pydantic.ValidationError` on schema violation.
        When env SAGE_TOOLRESULT_VALIDATE=1, JSON decode failure also
        raises (promotes warn-silent to hard-fail for production).
        """
        try:
            payload = json.loads(self.output)
        except (json.JSONDecodeError, TypeError) as exc:
            if _env_validate_strict():
                raise ValueError(
                    f"ToolResult.output is not valid JSON: {exc}. "
                    f"Set SAGE_TOOLRESULT_VALIDATE=0 to downgrade."
                ) from exc
            return None

        instance = schema.model_validate(payload)
        self.validated = instance
        return instance


class Tool:
    """A tool that an agent can use.

    `output_schema` (optional, AUDIT3 #17 / A14): when provided, callers
    can invoke `result.validate_output(tool.output_schema)` to get a
    typed Pydantic instance. Leaving it None preserves free-form output.

    `capability` (Phase 1.5c cycle-13 K, cgpro VERIFY 2026-05-06):
    declared ToolCapability tier (`pure` / `read_local` / `write_local` /
    `network` / `subprocess` / `dangerous`). Single label per tool;
    multi-effect tools classify as `dangerous`. Resolved by
    `sage.policy.manifest.resolve_tool_capability()` at registration.
    Phase 1.5c collapsed the resolution chain to: explicit kwarg ->
    class default -> raise. Built-ins MUST declare `capability=`
    inline at the factory call site (the manifest is documentation/
    audit only, not a runtime trust anchor — `function.__module__`
    is writable in Python and cannot anchor trust).
    """

    def __init__(
        self,
        spec: ToolDef,
        handler: Callable[..., Awaitable[str]],
        *,
        output_schema: type["BaseModel"] | None = None,
        capability: "ToolCapability | str | None" = None,
    ):
        from sage.policy.tool_policy import ToolCapability as _ToolCapability

        self.spec = spec
        self._handler = handler
        self.output_schema = output_schema
        # Coerce string -> enum at construction so downstream consumers
        # (registry, ToolPolicy.assert_allowed) can rely on the enum.
        if capability is None:
            self.capability: "_ToolCapability | None" = None
        elif isinstance(capability, _ToolCapability):
            self.capability = capability
        elif isinstance(capability, str):
            try:
                self.capability = _ToolCapability(capability)
            except ValueError:
                from sage.policy.errors import ToolPolicyDeclarationError

                raise ToolPolicyDeclarationError(
                    f"Tool {spec.name!r} got invalid capability {capability!r}. "
                    f"Must be one of {[c.value for c in _ToolCapability]}."
                )
        else:
            from sage.policy.errors import ToolPolicyDeclarationError

            raise ToolPolicyDeclarationError(
                f"Tool {spec.name!r} got capability of type "
                f"{type(capability).__name__}; expected ToolCapability or str."
            )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the tool with given arguments.

        Exception handling (2026-04-23, ALIRE2 §6 "Traceback leakage"):
        the model-visible ToolResult.output contains ONLY the exception
        type + message. The full traceback goes to the operator-side
        log via log.exception — not into the agent's prompt, where it
        previously leaked internal file paths, module names, and other
        structural info about the host. Exception type names are kept
        because the model often steers on them (e.g. FileNotFoundError
        → "create it first" vs PermissionError → "request approval").

        ToolPolicy gate (Phase 1.5 cycle-13 K): per cgpro DESIGN trap
        "Tool.execute MUST be the last-resort common boundary"
        (because agent_loop_execution can re-lookup ToolForge-
        synthesised tools after the AgentLoop pre-check). The gate
        resolves the tool's effective capability AND the ambient
        ToolPolicy via ContextVar; on denial, returns
        `ToolResult(is_error=True)` without invoking the handler.
        """
        from sage.observability.spans import sage_span, _safe_str
        from sage.policy.errors import (
            ToolPolicyDeclarationError,
            ToolPolicyDenied,
        )
        from sage.policy.manifest import resolve_tool_capability
        from sage.policy.tool_policy import get_effective_tool_policy

        with sage_span(
            "sage.tool",
            op="execute_tool",
            record_exception=False,  # We handle exceptions inside; redact before re-raise
            **{
                "gen_ai.tool.name": self.spec.name,
                "gen_ai.tool.call.arguments": _safe_str(arguments),
            },
        ) as _tool_span:
            # Phase 1.5 last-resort capability gate. Resolution failure
            # AND policy denial both produce `is_error=True` results
            # (no handler call, no payload exfiltration). The OracleStack
            # treats both as `trainable=False` per cgpro DESIGN Q4.
            try:
                cap = resolve_tool_capability(self)
                policy = get_effective_tool_policy()
                policy.assert_allowed(cap, tool_name=self.spec.name)
            except (ToolPolicyDenied, ToolPolicyDeclarationError) as policy_exc:
                log.warning(
                    "Tool %s blocked by ToolPolicy: %s",
                    self.spec.name,
                    type(policy_exc).__name__,
                )
                if _tool_span is not None:
                    _tool_span.set_attribute("error.type", type(policy_exc).__name__)
                    _tool_span.set_attribute("sage.tool.policy.denied", True)
                return ToolResult(
                    output=f"Error: {type(policy_exc).__name__}: {policy_exc}",
                    is_error=True,
                )

            try:
                output = await self._handler(**arguments)
                if _tool_span is not None:
                    _tool_span.set_attribute(
                        "gen_ai.tool.call.result", _safe_str(output)
                    )
                return ToolResult(output=output, is_error=False)
            except Exception as e:
                log.exception("Tool %s raised %s", self.spec.name, type(e).__name__)
                if _tool_span is not None:
                    _tool_span.set_attribute("error.type", type(e).__name__)
                return ToolResult(
                    output=f"Error: {type(e).__name__}: {e}",
                    is_error=True,
                )

    async def run(self, arguments: dict[str, Any]) -> str:
        """Execute and return raw output string."""
        result = await self.execute(arguments)
        return result.output

    @staticmethod
    def define(
        name: str,
        description: str,
        parameters: dict[str, Any],
        output_schema: type["BaseModel"] | None = None,
        capability: "ToolCapability | str | None" = None,
    ) -> Callable[[Callable[..., Awaitable[str]]], Tool]:
        """Decorator to define a tool from an async function.

        ``output_schema`` (AUDIT3 #17 / A14) threads through to the
        resulting Tool so callers can invoke
        ``result.validate_output(tool.output_schema)`` to get a typed
        Pydantic instance. Only meaningful for tools whose handler
        emits valid JSON — free-form string handlers should leave it
        at None (opt-in per-tool policy, 2026-04-24).

        ``capability`` (Phase 1.5c cycle-13 K) declares the
        ToolCapability tier for this tool. Required for ALL tools
        post-Phase-1.5c — built-ins must declare `capability=` inline
        at the factory call site (the built-in manifest is doc/audit
        only, not a runtime trust anchor). Class-level defaults still
        apply (e.g. `AgentTool` -> dangerous); otherwise
        `Registry.register` raises `ToolPolicyDeclarationError`.
        """

        def decorator(func: Callable[..., Awaitable[str]]) -> Tool:
            spec = ToolDef(name=name, description=description, parameters=parameters)
            return Tool(
                spec=spec,
                handler=func,
                output_schema=output_schema,
                capability=capability,
            )

        return decorator
