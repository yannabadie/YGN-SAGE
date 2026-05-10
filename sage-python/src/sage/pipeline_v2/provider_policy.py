"""Runtime provider policy enforcement.

The benchmark harness may request an allow/deny policy, but the runtime
owns enforcement. The policy is checked after model assignment is emitted
for auditability and before any provider execution for safety.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext


class ProviderPolicyViolation(RuntimeError):
    """Raised when an active provider policy would be violated."""


@dataclass(frozen=True)
class ProviderPolicy:
    allowlist: frozenset[str] | None
    denylist: frozenset[str]
    source: str = "none"

    @property
    def active(self) -> bool:
        return self.allowlist is not None or bool(self.denylist)

    def violation_reason(self, provider_id: str) -> str | None:
        provider = _normalize_provider(provider_id)
        if not self.active:
            return None
        if not provider:
            return "unknown_provider"
        if provider in self.denylist:
            return "denylist"
        if self.allowlist is not None and provider not in self.allowlist:
            return "outside_allowlist"
        return None


@dataclass(frozen=True)
class ProviderPolicyViolationItem:
    node_id: str
    model_id: str
    provider_id: str
    reason: str
    hint_provider_id: str = ""

    def to_dict(self) -> dict[str, str]:
        data = {
            "node_id": self.node_id,
            "model_id": self.model_id,
            "provider_id": self.provider_id,
            "reason": self.reason,
        }
        if self.hint_provider_id:
            data["hint_provider_id"] = self.hint_provider_id
        return data


@dataclass(frozen=True)
class ProviderPolicyDecision:
    policy: ProviderPolicy
    violations: tuple[ProviderPolicyViolationItem, ...]

    @property
    def active(self) -> bool:
        return self.policy.active

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "source": self.policy.source,
            "allowlist": (
                sorted(self.policy.allowlist)
                if self.policy.allowlist is not None
                else None
            ),
            "denylist": sorted(self.policy.denylist),
            "violations": [item.to_dict() for item in self.violations],
        }


def _normalize_provider(provider: str) -> str:
    return provider.strip().casefold()


def normalize_provider_iterable(values: Iterable[str] | None) -> frozenset[str]:
    if values is None:
        return frozenset()
    return frozenset(
        provider
        for raw in values
        if isinstance(raw, str)
        for provider in (_normalize_provider(raw),)
        if provider
    )


def parse_provider_csv(raw: str | None) -> tuple[str, ...] | None:
    """Parse a comma-separated provider list.

    ``None`` means the caller did not provide that control surface. An
    explicit empty string means the caller provided an empty list.
    """
    if raw is None:
        return None
    return tuple(
        provider
        for item in raw.split(",")
        for provider in (_normalize_provider(item),)
        if provider
    )


def _policy_from_parts(
    *,
    allowlist: Iterable[str] | None,
    denylist: Iterable[str] | None,
    source: str,
) -> ProviderPolicy:
    allow = None
    if allowlist is not None:
        normalized = normalize_provider_iterable(allowlist)
        allow = normalized if normalized else None
    return ProviderPolicy(
        allowlist=allow,
        denylist=normalize_provider_iterable(denylist),
        source=source,
    )


def provider_policy_from_env() -> ProviderPolicy:
    allow_raw = os.environ.get("SAGE_PROVIDER_ALLOWLIST")
    deny_raw = os.environ.get("SAGE_PROVIDER_DENYLIST")
    if allow_raw is None and deny_raw is None:
        return ProviderPolicy(allowlist=None, denylist=frozenset(), source="none")
    return _policy_from_parts(
        allowlist=parse_provider_csv(allow_raw),
        denylist=parse_provider_csv(deny_raw),
        source="env",
    )


def effective_provider_policy(
    pipeline: "CognitiveOrchestrationPipeline",
) -> ProviderPolicy:
    source = getattr(pipeline, "_provider_policy_source", "")
    has_cli_policy = bool(source)
    allowlist = getattr(pipeline, "_provider_allowlist", None)
    denylist = getattr(pipeline, "_provider_denylist", None)
    if has_cli_policy:
        return _policy_from_parts(
            allowlist=allowlist,
            denylist=denylist,
            source=str(source),
        )
    return provider_policy_from_env()


def configure_pipeline_provider_policy(
    pipeline: "CognitiveOrchestrationPipeline",
    *,
    allowlist: Iterable[str] | None,
    denylist: Iterable[str] | None,
    source: str | None,
) -> ProviderPolicy:
    """Attach an explicit provider policy to the pipeline and pool."""
    if source is None:
        policy = effective_provider_policy(pipeline)
    else:
        policy = _policy_from_parts(
            allowlist=allowlist,
            denylist=denylist,
            source=source,
        )
        setattr(pipeline, "_provider_allowlist", tuple(sorted(policy.allowlist or ())))
        setattr(pipeline, "_provider_denylist", tuple(sorted(policy.denylist)))
        setattr(pipeline, "_provider_policy_source", source)

    pool = getattr(pipeline, "provider_pool", None)
    if pool is not None and hasattr(pool, "set_provider_policy"):
        pool.set_provider_policy(
            allowlist=policy.allowlist,
            denylist=policy.denylist,
            source=policy.source,
        )
    install_provider_call_guards(pipeline)
    return policy


def provider_id_for_model(
    pipeline: "CognitiveOrchestrationPipeline",
    model_id: str,
) -> str:
    """Resolve provider for policy enforcement.

    Provider hints are deliberately not authoritative here: a topology hint
    must not be able to relabel an OpenAI model as Google.
    """
    if not model_id:
        return ""
    pool = getattr(pipeline, "provider_pool", None)
    if pool is not None and hasattr(pool, "infer_provider"):
        try:
            provider = pool.infer_provider(model_id)
            if provider:
                return _normalize_provider(str(provider))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    return ""


def _iter_effective_assignments(
    ctx: "PipelineContext",
) -> list[tuple[int, str, str]]:
    seen: set[int] = set()
    items: list[tuple[int, str, str]] = []
    topology = getattr(ctx, "topology", None)
    if topology is not None and hasattr(topology, "node_count"):
        try:
            node_count = int(topology.node_count())
        except (TypeError, ValueError, RuntimeError):
            node_count = 0
        for idx in range(node_count):
            seen.add(idx)
            node_model = ""
            if hasattr(topology, "get_node"):
                try:
                    node = topology.get_node(idx)
                    node_model = str(getattr(node, "model_id", "") or "")
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    node_model = ""
            model_id = str(ctx.assignments.get(idx, node_model) or "")
            items.append((idx, model_id, node_model))
    for idx, model_id in sorted(ctx.assignments.items()):
        if idx not in seen:
            items.append((idx, str(model_id or ""), ""))
    return items


def evaluate_provider_policy(
    pipeline: "CognitiveOrchestrationPipeline",
    ctx: "PipelineContext",
    policy: ProviderPolicy | None = None,
) -> ProviderPolicyDecision:
    policy = policy or effective_provider_policy(pipeline)
    violations: list[ProviderPolicyViolationItem] = []
    if policy.active:
        for idx, model_id, _node_model in _iter_effective_assignments(ctx):
            if not model_id:
                continue
            provider_id = provider_id_for_model(pipeline, model_id)
            reason = policy.violation_reason(provider_id)
            if reason is None:
                continue
            hint_provider = str(getattr(ctx, "provider_hints", {}).get(idx, "") or "")
            violations.append(
                ProviderPolicyViolationItem(
                    node_id=str(idx),
                    model_id=model_id,
                    provider_id=provider_id,
                    reason=reason,
                    hint_provider_id=_normalize_provider(hint_provider),
                )
            )
    return ProviderPolicyDecision(policy=policy, violations=tuple(violations))


def _violation_message(decision: ProviderPolicyDecision) -> str:
    policy = decision.policy
    payload = {
        "source": policy.source,
        "allowlist": sorted(policy.allowlist) if policy.allowlist is not None else None,
        "denylist": sorted(policy.denylist),
        "violations": [item.to_dict() for item in decision.violations],
    }
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    message = f"provider policy violation: {rendered}"
    if len(message) <= 1024:
        return message
    return message[:1000] + "...<truncated>"


def enforce_provider_policy(
    pipeline: "CognitiveOrchestrationPipeline",
    ctx: "PipelineContext",
    event_log: Any,
) -> None:
    decision = evaluate_provider_policy(pipeline, ctx)
    setattr(ctx, "_provider_policy_decision", decision.to_dict())
    if not decision.violations:
        return
    message = _violation_message(decision)
    if event_log is not None:
        event_log.emit_failure(
            kind="provider_policy",
            error_type="provider_policy_violation",
            message=message,
        )
    raise ProviderPolicyViolation(message)


def enforce_model_provider_policy(
    pipeline: "CognitiveOrchestrationPipeline",
    *,
    model_id: str,
    provider_id: str = "",
    event_log: Any = None,
    node_id: str = "",
) -> None:
    policy = effective_provider_policy(pipeline)
    if not policy.active:
        return
    resolved_provider = _normalize_provider(provider_id) or provider_id_for_model(
        pipeline,
        model_id,
    )
    reason = policy.violation_reason(resolved_provider)
    if reason is None:
        return
    decision = ProviderPolicyDecision(
        policy=policy,
        violations=(
            ProviderPolicyViolationItem(
                node_id=node_id,
                model_id=model_id,
                provider_id=resolved_provider,
                reason=reason,
            ),
        ),
    )
    message = _violation_message(decision)
    if event_log is not None:
        event_log.emit_failure(
            kind="provider_policy",
            error_type="provider_policy_violation",
            message=message,
            node_id=node_id,
        )
    raise ProviderPolicyViolation(message)


def _current_event_log_or_none() -> Any:
    try:
        from sage.runtime.event_log import current_event_log

        return current_event_log()
    except Exception:  # noqa: BLE001 - guard must be usable during boot.
        return None


def _call_config(
    default_config: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    stream: bool = False,
) -> Any:
    config = kwargs.get("config")
    if config is not None:
        return config
    positional_index = 1 if stream else 2
    if len(args) > positional_index:
        candidate = args[positional_index]
        if candidate is not None:
            return candidate
    return default_config


def _provider_identity(provider: Any, config: Any) -> tuple[str, str]:
    provider_id = (
        getattr(config, "provider", "")
        if config is not None
        else ""
    ) or getattr(provider, "name", "") or getattr(provider, "provider_name", "")
    model_id = (
        getattr(config, "model", "")
        if config is not None
        else ""
    ) or getattr(provider, "model_id", "") or getattr(provider, "model_string", "")
    return str(provider_id or ""), str(model_id or "")


def guard_provider_call(
    pipeline: "CognitiveOrchestrationPipeline",
    *,
    provider: Any,
    config: Any,
    source: str,
) -> None:
    provider_attr_id = getattr(provider, "name", "") or getattr(
        provider,
        "provider_name",
        "",
    )
    provider_attr_model = getattr(provider, "model_id", "") or getattr(
        provider,
        "model_string",
        "",
    )
    if provider_attr_id or provider_attr_model:
        enforce_model_provider_policy(
            pipeline,
            model_id=str(provider_attr_model or ""),
            provider_id=str(provider_attr_id or ""),
            event_log=_current_event_log_or_none(),
            node_id=source,
        )
    provider_id, model_id = _provider_identity(provider, config)
    enforce_model_provider_policy(
        pipeline,
        model_id=model_id,
        provider_id=provider_id,
        event_log=_current_event_log_or_none(),
        node_id=source,
    )


def guard_provider_instance(
    pipeline: "CognitiveOrchestrationPipeline",
    provider: Any,
    *,
    default_config: Any = None,
    source: str,
) -> Any:
    """Patch a provider instance so direct generate calls cannot bypass policy."""
    if provider is None or getattr(provider, "_sage_provider_policy_guarded", False):
        return provider

    original_generate = getattr(provider, "generate", None)
    if callable(original_generate):

        async def _guarded_generate(*args: Any, **kwargs: Any) -> Any:
            config = _call_config(default_config, args, kwargs)
            guard_provider_call(
                pipeline,
                provider=provider,
                config=config,
                source=source,
            )
            return await original_generate(*args, **kwargs)

        setattr(provider, "generate", _guarded_generate)

    original_stream = getattr(provider, "generate_stream", None)
    if callable(original_stream):

        def _guarded_generate_stream(*args: Any, **kwargs: Any) -> Any:
            config = _call_config(default_config, args, kwargs, stream=True)
            guard_provider_call(
                pipeline,
                provider=provider,
                config=config,
                source=f"{source}.stream",
            )
            return original_stream(*args, **kwargs)

        setattr(provider, "generate_stream", _guarded_generate_stream)

    try:
        setattr(provider, "_sage_provider_policy_guarded", True)
    except Exception:  # noqa: BLE001 - some provider objects may be immutable.
        pass
    return provider


def install_provider_call_guards(
    pipeline: "CognitiveOrchestrationPipeline",
) -> None:
    """Install provider-call guards on known pipeline/provider-pool instances."""
    if not effective_provider_policy(pipeline).active:
        return

    default_config = getattr(pipeline, "llm_config", None)
    default_provider = getattr(pipeline, "llm_provider", None)
    if default_provider is not None:
        guarded = guard_provider_instance(
            pipeline,
            default_provider,
            default_config=default_config,
            source="pipeline.llm_provider",
        )
        setattr(pipeline, "llm_provider", guarded)

    agent_loop = getattr(pipeline, "_agent_loop", None)
    if agent_loop is not None and getattr(agent_loop, "_llm", None) is not None:
        guarded = guard_provider_instance(
            pipeline,
            agent_loop._llm,
            default_config=getattr(getattr(agent_loop, "config", None), "llm", None),
            source="agent_loop.llm_provider",
        )
        agent_loop._llm = guarded

    pool = getattr(pipeline, "provider_pool", None)
    if pool is None:
        return
    if getattr(pool, "_default", None) is not None:
        pool._default = guard_provider_instance(
            pipeline,
            pool._default,
            default_config=getattr(pool, "_default_config", None),
            source="provider_pool.default",
        )
    providers = getattr(pool, "_providers", None)
    if isinstance(providers, dict):
        for name, provider in list(providers.items()):
            providers[name] = guard_provider_instance(
                pipeline,
                provider,
                default_config=None,
                source=f"provider_pool.{name}",
            )


__all__ = [
    "ProviderPolicy",
    "ProviderPolicyDecision",
    "ProviderPolicyViolation",
    "configure_pipeline_provider_policy",
    "effective_provider_policy",
    "enforce_model_provider_policy",
    "enforce_provider_policy",
    "evaluate_provider_policy",
    "guard_provider_call",
    "guard_provider_instance",
    "install_provider_call_guards",
    "normalize_provider_iterable",
    "parse_provider_csv",
    "provider_id_for_model",
    "provider_policy_from_env",
]
