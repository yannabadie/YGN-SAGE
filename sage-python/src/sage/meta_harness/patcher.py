"""Runtime patcher: inject HarnessConfig into SAGE pipeline/runner.

Uses monkey-patching to overlay harness parameters at runtime,
so the original source code stays untouched. Each candidate runs
in an ephemeral patched state that is reverted after evaluation.

This bridges Meta-Harness search space (HarnessConfig) and SAGE's
execution engine (TopologyRunner + CognitiveOrchestrationPipeline).
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from typing import Any, Generator

from sage.llm.base import Message, Role
from sage.meta_harness.config import HarnessConfig

log = logging.getLogger(__name__)


class HarnessPatcher:
    """Applies a HarnessConfig to a live TopologyRunner instance.

    Usage:
        patcher = HarnessPatcher(config)
        with patcher.patched(runner=runner, pipeline=pipeline):
            result = await runner.run(task)
        # Original methods restored automatically
    """

    def __init__(self, config: HarnessConfig) -> None:
        self.config = config
        self._runner_originals: dict[str, Any] = {}
        self._pipeline_originals: dict[str, Any] = {}

    # ── Runner patching ─────────────────────────────────────────────────

    def patch_runner(self, runner: Any) -> None:
        """Monkey-patch a TopologyRunner with config-driven methods."""
        cfg = self.config
        ctx_cfg = cfg.context
        prompt_cfg = cfg.prompts
        exec_cfg = cfg.execution

        # Save originals for unpatching
        self._runner_originals = {
            "_gather_predecessor_context": runner._gather_predecessor_context,
            "_context_budget_per_predecessor": runner._context_budget_per_predecessor,
            "_execute_node": runner._execute_node,
            "_max_rounds": runner._max_rounds,
        }

        # ── 1. Context budget ───────────────────────────────────────────
        def _patched_budget(n_predecessors: int, node_idx: int = 0) -> int:
            context_window = 131072
            try:
                node = runner.graph.get_node(node_idx)
                model_id = getattr(node, "model_id", "")
                if model_id and runner._provider_pool:
                    _, resolved_config = runner._provider_pool.resolve(model_id)
                    cw = getattr(resolved_config, "context_window", 0)
                    if cw and cw > 0:
                        context_window = cw
            except (AttributeError, RuntimeError):
                pass

            available_chars = int(
                context_window * ctx_cfg.budget_ratio * ctx_cfg.chars_per_token
            )
            budget = available_chars // max(n_predecessors, 1)
            return max(budget, ctx_cfg.budget_floor_chars)

        runner._context_budget_per_predecessor = _patched_budget

        # ── 2. Context aggregation ──────────────────────────────────────
        def _patched_gather(node_idx: int) -> str:
            predecessor_indices: list[int] = []
            try:
                predecessor_indices = runner.graph.get_predecessors(node_idx)
            except (AttributeError, Exception):
                return runner._gather_all_context()

            budget = runner._context_budget_per_predecessor(
                len(predecessor_indices), node_idx
            )

            parts: list[str] = []
            parts_for_dedup: list[tuple[str, str]] = []
            for idx in predecessor_indices:
                output = runner._node_outputs.get(idx)
                if not output:
                    continue
                node = runner.graph.get_node(idx)
                role = getattr(node, "role", f"node-{idx}")
                model_id = getattr(node, "model_id", "")
                truncated = runner._truncate_output(output, budget)
                parts_for_dedup.append((truncated, role))

            # Dedup with configurable threshold
            deduplicated = runner.__class__._deduplicate_context(
                parts_for_dedup, ctx_cfg.similarity_threshold
            )

            # Format each predecessor output using template
            formatted_parts: list[str] = []
            for text, role in deduplicated:
                formatted = ctx_cfg.predecessor_format.format(
                    role=role, text=text, node_idx=0, model_id="",
                )
                formatted_parts.append(formatted)

            joined = ctx_cfg.predecessor_separator.join(formatted_parts)
            if not joined:
                return ""

            return ctx_cfg.injection_template.format(
                context=joined,
                n_predecessors=len(predecessor_indices),
                task_preview="",
            )

        runner._gather_predecessor_context = _patched_gather

        # ── 3. System prompt + execution ────────────────────────────────
        original_execute = self._runner_originals["_execute_node"]

        async def _patched_execute_node(
            node_idx: int, task: str, context_override: str | None = None,
        ) -> str:
            node = runner.graph.get_node(node_idx)

            # Dispatch code nodes unchanged
            node_type = getattr(node, "node_type", "llm")
            if node_type == "code":
                return await runner._execute_code_node(
                    node_idx, task, context_override
                )

            # Dispatch solver nodes unchanged (deterministic Rust)
            if node_type == "solver" or getattr(node, "role", "") == "solver":
                return await runner._execute_solver_node(
                    node_idx, task, context_override
                )

            role = getattr(node, "role", f"node-{node_idx}")
            caps = getattr(node, "required_capabilities", [])

            # ── Build system prompt ─────────────────────────────────────
            if role in prompt_cfg.role_overrides:
                system_prompt = prompt_cfg.role_overrides[role].format(
                    role=role,
                    capabilities=", ".join(caps) if caps else "",
                    task_preview=task[:200],
                    n_predecessors=len(
                        runner.graph.get_predecessors(node_idx)
                        if hasattr(runner.graph, "get_predecessors")
                        else []
                    ),
                )
            else:
                custom_prompt = getattr(node, "prompt", "")
                if custom_prompt:
                    system_prompt = custom_prompt
                else:
                    system_prompt = prompt_cfg.default_template.format(
                        role=role,
                        capabilities=", ".join(caps) if caps else "",
                        task_preview=task[:200],
                        n_predecessors=0,
                    )
                    if caps:
                        system_prompt += prompt_cfg.capability_template.format(
                            capabilities=", ".join(caps),
                        )

            # Global prefix/suffix
            if prompt_cfg.global_prefix:
                system_prompt = prompt_cfg.global_prefix + "\n" + system_prompt
            if prompt_cfg.global_suffix:
                system_prompt = system_prompt + "\n" + prompt_cfg.global_suffix

            # ── Build messages ──────────────────────────────────────────
            messages: list[Message] = [
                Message(role=Role.SYSTEM, content=system_prompt),
            ]

            context = (
                context_override
                if context_override is not None
                else runner._gather_predecessor_context(node_idx)
            )
            if context:
                messages.append(Message(role=Role.SYSTEM, content=context))

            messages.append(Message(role=Role.USER, content=task))

            # ── Resolve provider ────────────────────────────────────────
            node_model_id = getattr(node, "model_id", "")
            if node_model_id and runner._provider_pool:
                provider, config = runner._provider_pool.resolve(node_model_id)
            else:
                provider, config = runner._llm, runner._config

            # ── Context overflow ────────────────────────────────────────
            total_chars = sum(len(m.content) for m in messages)
            context_window = getattr(config, "context_window", 0) or 128000
            estimated_tokens = total_chars // ctx_cfg.chars_per_token

            if estimated_tokens > context_window * exec_cfg.overflow_threshold:
                context_msg = next(
                    (m for m in messages
                     if "previous" in m.content.lower()[:50]
                     or "context" in m.content.lower()[:30]),
                    None,
                )
                if context_msg:
                    if ctx_cfg.overflow_strategy == "truncate":
                        max_chars = int(context_window * 0.6 * ctx_cfg.chars_per_token)
                        context_msg.content = (
                            context_msg.content[:max_chars] + "\n[truncated]"
                        )
                    elif ctx_cfg.overflow_strategy == "hierarchical":
                        c = context_msg.content
                        keep = int(len(c) * 0.4)
                        context_msg.content = (
                            c[:keep] + "\n[...middle omitted...]\n" + c[-keep:]
                        )
                    else:  # summarize
                        try:
                            summary_msgs = [
                                Message(
                                    role=Role.SYSTEM,
                                    content=exec_cfg.compression_prompt,
                                ),
                                Message(
                                    role=Role.USER,
                                    content=context_msg.content[:context_window * 2],
                                ),
                            ]
                            summary_resp = await provider.generate(
                                messages=summary_msgs, config=config,
                            )
                            context_msg.content = (
                                f"Context (summarized):\n{summary_resp.content or ''}"
                            )
                        except (RuntimeError, TimeoutError, asyncio.TimeoutError):
                            max_chars = int(
                                context_window * 0.6 * ctx_cfg.chars_per_token
                            )
                            context_msg.content = (
                                context_msg.content[:max_chars] + "\n[truncated]"
                            )

            # ── Execute LLM call ────────────────────────────────────────
            # Note: DO NOT wrap in asyncio.wait_for — streaming providers
            # (OpenRouter, etc.) return HTTP 200 immediately but stream
            # tokens over 60-180s. Provider-level timeouts handle errors.
            try:
                response = await provider.generate(
                    messages=messages, config=config,
                )
                output = response.content or ""
                if runner._provider_pool and hasattr(runner._provider_pool, "record_success"):
                    provider_name = getattr(config, "provider", "unknown")
                    runner._provider_pool.record_success(provider_name)
            except (RuntimeError, TimeoutError, asyncio.TimeoutError, ConnectionError) as exc:
                log.warning("Node %d execution failed: %s", node_idx, exc)
                # Fallback: try default provider
                if provider != runner._llm and runner._llm:
                    try:
                        response = await runner._llm.generate(
                            messages=messages, config=runner._config,
                        )
                        output = response.content or ""
                        log.info("Node %d fallback to default provider succeeded", node_idx)
                    except Exception:
                        output = ""
                else:
                    output = ""

            runner._node_outputs[node_idx] = output
            return output

        runner._execute_node = _patched_execute_node

        # ── 4. Debate rounds ────────────────────────────────────────────
        runner._max_rounds = exec_cfg.max_debate_rounds

        log.info(
            "HarnessPatcher applied config '%s' (%s) to runner",
            cfg.id, cfg.description,
        )

    def unpatch_runner(self, runner: Any) -> None:
        """Restore original methods."""
        for attr, original in self._runner_originals.items():
            setattr(runner, attr, original)
        self._runner_originals.clear()

    # ── Pipeline patching ───────────────────────────────────────────────

    def patch_pipeline(self, pipeline: Any) -> None:
        """Intercept TopologyRunner creation so every runner gets patched."""
        pipeline._meta_harness_config = self.config
        patcher_self = self

        # Monkey-patch _stage_execute to auto-patch each runner it creates
        if hasattr(pipeline, "_stage_execute"):
            original_stage_execute = pipeline._stage_execute

            async def _patched_stage_execute(ctx: Any) -> Any:
                result = await original_stage_execute(ctx)
                return result

            # Instead, intercept at the TopologyRunner import level
            # by wrapping the execute method that creates runners
            pass  # handled below

        # Wrap the execute stage to patch runners after creation
        from sage.topology.runner import TopologyRunner as _TR

        _original_runner_init = _TR.__init__

        def _hooked_runner_init(self_runner: Any, *args: Any, **kwargs: Any) -> None:
            _original_runner_init(self_runner, *args, **kwargs)
            # Auto-patch this runner with the harness config
            patcher_self.patch_runner(self_runner)

        _TR.__init__ = _hooked_runner_init
        self._pipeline_originals["_TR_init"] = (_TR, _original_runner_init)

        log.info(
            "HarnessPatcher attached config '%s' to pipeline (runner auto-patch active)",
            self.config.id,
        )

    def unpatch_pipeline(self, pipeline: Any) -> None:
        if hasattr(pipeline, "_meta_harness_config"):
            del pipeline._meta_harness_config
        # Restore original TopologyRunner.__init__
        if "_TR_init" in self._pipeline_originals:
            cls, original_init = self._pipeline_originals.pop("_TR_init")
            cls.__init__ = original_init

    # ── Context manager ─────────────────────────────────────────────────

    @contextmanager
    def patched(
        self,
        runner: Any | None = None,
        pipeline: Any | None = None,
    ) -> Generator[None, None, None]:
        """Context manager for safe patching/unpatching."""
        if runner:
            self.patch_runner(runner)
        if pipeline:
            self.patch_pipeline(pipeline)
        try:
            yield
        finally:
            if runner:
                self.unpatch_runner(runner)
            if pipeline:
                self.unpatch_pipeline(pipeline)
