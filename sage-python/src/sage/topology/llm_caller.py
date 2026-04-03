"""LLM topology synthesis caller -- completes Path 3 in the Rust TopologyEngine.

Calls the LLM with structured prompts to generate:
1. Role assignments (Stage 1 JSON)
2. Structure design (Stage 2 JSON)

Then feeds both JSONs to Rust for graph construction and validation.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

_log = logging.getLogger(__name__)


def _default_fast_model() -> str:
    """Resolve the 'fast' tier model via config (no hardcoded model name)."""
    from sage.llm.config_loader import get_tier_model
    return get_tier_model("fast")


# ── Tool-call system prompt (must match training in sage_tool_schemas.py) ──
_SAGE_TOOLS_JSON = json.dumps([
    {
        "type": "function",
        "function": {
            "name": "create_topology",
            "description": "Design a multi-agent DAG topology to solve a coding task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "difficulty": {"type": "string", "enum": ["simple", "moderate", "complex"]},
                    "reasoning": {"type": "string"},
                    "nodes": {"type": "array", "items": {"type": "object", "properties": {
                        "role": {"type": "string"},
                        "model_tier": {"type": "string", "enum": ["budget", "fast", "balanced", "reasoner", "codex"]},
                        "prompt": {"type": "string"},
                    }, "required": ["role", "model_tier", "prompt"]}},
                    "edges": {"type": "array", "items": {"type": "object", "properties": {
                        "from_idx": {"type": "integer"}, "to_idx": {"type": "integer"},
                        "flow_type": {"type": "string", "enum": ["message", "control", "state"]},
                    }, "required": ["from_idx", "to_idx", "flow_type"]}},
                },
                "required": ["difficulty", "reasoning", "nodes", "edges"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adapt_topology",
            "description": "Runtime adaptation: upgrade, reroute, or continue at a checkpoint.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["continue", "upgrade", "reroute"]},
                    "node_idx": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["action"],
            },
        },
    },
], indent=2)

_TOOLCALL_SYSTEM_PROMPT = (
    "You are a multi-agent topology designer for the YGN-SAGE framework. "
    "You design optimal agent DAG topologies and make runtime adaptation decisions.\n\n"
    f"<tools>\n{_SAGE_TOOLS_JSON}\n</tools>\n\n"
    "For each task, call create_topology with a JSON topology. "
    "Use <tool_call> format."
)


def _parse_toolcall(raw: str) -> dict | None:
    """Parse <tool_call>...</tool_call> output from the local model."""
    match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', raw, re.DOTALL)
    if not match:
        # Try without tags (sometimes model omits closing tag)
        match = re.search(r'<tool_call>\s*(\{.*)', raw, re.DOTALL)
    if not match:
        return None
    try:
        call = json.loads(match.group(1))
        if call.get("name") == "create_topology" and "arguments" in call:
            return call["arguments"]
        # Direct topology dict (no wrapper)
        if "nodes" in call:
            return call
    except json.JSONDecodeError:
        pass
    return None


VALID_TEMPLATES = [
    "sequential", "parallel", "avr", "self_moa",
    "hierarchical", "hub", "debate", "brainstorming",
]


def build_role_prompt(
    task: str,
    max_agents: int = 4,
    available_models: list[str] | None = None,
) -> str:
    """Build the Stage 1 prompt: role assignment with per-agent system prompts."""
    from sage.llm.config_loader import get_tier_model
    models_str = ", ".join(available_models or [get_tier_model("fast")])
    return (
        "You are a multi-agent topology designer. Given a task, assign roles to agents.\n\n"
        f"TASK: {task}\n\n"
        f"CONSTRAINTS:\n"
        f"- Maximum {max_agents} agents\n"
        f"- Available models: {models_str}\n"
        "- Each agent needs:\n"
        "  1. name: short role name (e.g. 'coder', 'reviewer', 'planner')\n"
        "  2. model: model ID from the available list\n"
        "  3. system: cognitive tier (1=fast/reflexive, 2=deliberate/analytical, 3=formal/verification)\n"
        "  4. capabilities: list of required capabilities (e.g. 'code_generation', 'code_review')\n"
        "  5. prompt: detailed system prompt for this agent — specific instructions, expertise, "
        "constraints, and output format expectations. This prompt will be injected as the agent's "
        "system message. Make it task-specific and actionable (2-5 sentences).\n\n"
        "Respond with ONLY valid JSON (no markdown, no explanation):\n"
        '{\n  "roles": [\n'
        '    {\n'
        '      "name": "agent_name",\n'
        '      "model": "model_id",\n'
        '      "system": 2,\n'
        '      "capabilities": ["cap1"],\n'
        '      "prompt": "You are an expert ... Your task is to ..."\n'
        '    }\n'
        "  ]\n}"
    )


def build_structure_prompt(roles_json: str) -> str:
    """Build the Stage 2 prompt: structure design."""
    return (
        "Given these agent roles, design the communication structure.\n\n"
        f"ROLES:\n{roles_json}\n\n"
        "Design an adjacency matrix and edge types. Use these edge types:\n"
        '- "control" -- scheduling dependency (A must finish before B starts)\n'
        '- "message" -- data flows from A to B\n'
        '- "state" -- shared state synchronization\n\n'
        f"Choose the best topology template from: {', '.join(VALID_TEMPLATES)}\n\n"
        "Respond with ONLY valid JSON (no markdown, no explanation):\n"
        '{\n  "adjacency": [[0, 1], [0, 0]],\n'
        '  "edge_types": [["", "control"], ["", ""]],\n'
        '  "template": "sequential"\n}'
    )


def _extract_json(text: str) -> str:
    """Extract JSON from LLM response (may be wrapped in markdown fences)."""
    match = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    text = text.strip()
    if text.startswith("{"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return text[start:end + 1]
    return text


def parse_and_build_topology(
    roles_json: str,
    structure_json: str,
) -> Any:
    """Parse role + structure JSONs and build a TopologyGraph via Rust.

    Returns the TopologyGraph, or None if construction fails.
    """
    try:
        from sage_core import TopologyGraph, TopologyNode, TopologyEdge
    except ImportError:
        _log.warning("sage_core not available, cannot build topology")
        return None

    roles_data = json.loads(roles_json)
    roles = roles_data.get("roles", [])

    struct_data = json.loads(structure_json)
    adjacency = struct_data.get("adjacency", [])
    edge_types = struct_data.get("edge_types", [])
    template = struct_data.get("template", "sequential")

    n = len(roles)
    if len(adjacency) != n:
        _log.error("Dimension mismatch: %d roles but %dx adjacency", n, len(adjacency))
        return None

    graph = TopologyGraph(template)

    for role in roles:
        # TopologyNode(role, model_id, system, required_capabilities,
        #              security_label, max_cost_usd, max_wall_time_s, prompt)
        node = TopologyNode(
            role.get("name", "agent"),
            role.get("model", _default_fast_model()),
            role.get("system", 2),
            role.get("capabilities", []),
            0,     # security_label
            1.0,   # max_cost_usd
            60.0,  # max_wall_time_s
            prompt=role.get("prompt", ""),
        )
        graph.add_node(node)

    for i in range(n):
        for j in range(n):
            if i < len(adjacency) and j < len(adjacency[i]) and adjacency[i][j] == 1:
                et = ""
                if i < len(edge_types) and j < len(edge_types[i]):
                    et = edge_types[i][j]
                edge = TopologyEdge(et or "control", None, "open", None, 1.0)
                graph.add_edge(i, j, edge)

    _log.info(
        "Built topology from LLM: template=%s, nodes=%d, edges=%d",
        template, graph.node_count(), graph.edge_count(),
    )
    return graph


async def synthesize_topology(
    llm_provider: Any,
    task: str,
    max_agents: int = 4,
    available_models: list[str] | None = None,
) -> Any | None:
    """Full LLM synthesis pipeline: prompt -> LLM -> JSON -> Rust graph.

    Args:
        llm_provider: An LLMProvider instance (GoogleProvider, etc.)
        task: The task description to design a topology for.
        max_agents: Maximum number of agents.
        available_models: List of available model IDs.

    Returns:
        TopologyGraph if synthesis succeeds, None otherwise.
    """
    from sage.llm.base import Message, Role, LLMConfig

    # Stage 1: Role assignment — constrained to JSON schema
    roles_schema: dict = {
        "type": "object",
        "properties": {
            "roles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "model": {"type": "string"},
                        "system": {"type": "integer"},
                        "capabilities": {"type": "array", "items": {"type": "string"}},
                        "prompt": {"type": "string"},
                    },
                    "required": ["name", "model", "system", "capabilities", "prompt"],
                },
            }
        },
        "required": ["roles"],
    }

    # Stage 2: Structure design — constrained to JSON schema
    structure_schema: dict = {
        "type": "object",
        "properties": {
            "adjacency": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}},
            },
            "edge_types": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "string"}},
            },
            "template": {"type": "string"},
        },
        "required": ["adjacency", "edge_types", "template"],
    }

    _fast = _default_fast_model()
    config_stage1 = LLMConfig(
        provider="google", model=_fast, json_schema=roles_schema
    )
    config_stage2 = LLMConfig(
        provider="google", model=_fast, json_schema=structure_schema
    )

    role_prompt = build_role_prompt(task, max_agents, available_models)
    try:
        response1 = await llm_provider.generate(
            messages=[
                Message(role=Role.SYSTEM, content="You are a JSON-only topology designer."),
                Message(role=Role.USER, content=role_prompt),
            ],
            config=config_stage1,
        )
        roles_json = _extract_json(response1.content or "")
        json.loads(roles_json)  # validate parses
    except Exception as e:
        _log.warning("Stage 1 (role assignment) failed: %s", e)
        return None

    # Stage 2: Structure design
    structure_prompt = build_structure_prompt(roles_json)
    try:
        response2 = await llm_provider.generate(
            messages=[
                Message(role=Role.SYSTEM, content="You are a JSON-only topology designer."),
                Message(role=Role.USER, content=structure_prompt),
            ],
            config=config_stage2,
        )
        structure_json = _extract_json(response2.content or "")
        json.loads(structure_json)  # validate parses
    except Exception as e:
        _log.warning("Stage 2 (structure design) failed: %s", e)
        return None

    # Stage 3: Build + validate via Rust
    graph = parse_and_build_topology(roles_json, structure_json)
    if graph is None:
        _log.warning("Stage 3 (build) failed")
        return None

    _log.info(
        "LLM synthesis complete: task=%r, nodes=%d, edges=%d",
        task[:60], graph.node_count(), graph.edge_count(),
    )
    return graph


# ---------------------------------------------------------------------------
# Policy Model (Path 6 in TopologyEngine)
# ---------------------------------------------------------------------------
# V1: Phi-4-mini SFT (yannabadie/sage-topology-policy) — legacy
# V2: Nemotron-Orchestrator-8B GRPO (yannabadie/sage-topology-policy-v2)
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dc


@_dc
class PolicyModelConfig:
    """Configuration for a topology policy model variant."""
    repo: str
    base_model: str
    chat_template: str  # "phi4" or "qwen3"
    max_new_tokens: int
    trust_remote_code: bool = True


POLICY_LOCAL = PolicyModelConfig(
    repo="yannabadie/sage-topology-policy-local",
    base_model="Qwen/Qwen3-4B",
    chat_template="toolcall",  # <tool_call> JSON format
    max_new_tokens=1024,
    trust_remote_code=True,
)

POLICY_V2 = PolicyModelConfig(
    repo="yannabadie/sage-topology-policy-v2",
    base_model="nvidia/Nemotron-Orchestrator-8B",
    chat_template="qwen3",
    max_new_tokens=512,
    trust_remote_code=True,
)

POLICY_V1 = PolicyModelConfig(
    repo="yannabadie/sage-topology-policy",
    base_model="microsoft/Phi-4-mini-instruct",
    chat_template="phi4",
    max_new_tokens=256,
    trust_remote_code=False,
)

_POLICY_CACHE_DIR = None  # Populated by download_policy_model()
_ACTIVE_POLICY_CONFIG: PolicyModelConfig | None = None


def download_policy_model(cache_dir: str | None = None) -> tuple[str | None, PolicyModelConfig | None]:
    """Download or locate the topology policy model.

    Priority:
    1. SAGE_PATH6_ADAPTER env var → local adapter path (Qwen3-4B tool-call)
    2. V2 from HuggingFace (Nemotron-8B GRPO)
    3. V1 from HuggingFace (Phi-4-mini SFT)

    Returns (local_path, config) or (None, None) if unavailable.
    """
    global _POLICY_CACHE_DIR, _ACTIVE_POLICY_CONFIG
    if _POLICY_CACHE_DIR is not None and _ACTIVE_POLICY_CONFIG is not None:
        return _POLICY_CACHE_DIR, _ACTIVE_POLICY_CONFIG

    import os

    # Priority 1: Local adapter path (for local Qwen3-4B training)
    local_adapter = os.environ.get("SAGE_PATH6_ADAPTER")
    if local_adapter and os.path.isdir(local_adapter):
        _POLICY_CACHE_DIR = os.path.abspath(local_adapter)
        _ACTIVE_POLICY_CONFIG = POLICY_LOCAL
        _log.info("Path 6: using local adapter at %s", _POLICY_CACHE_DIR)
        return _POLICY_CACHE_DIR, _ACTIVE_POLICY_CONFIG

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        _log.debug("huggingface_hub not installed — topology policy unavailable")
        return None, None

    # Priority 2-3: HuggingFace (V2 then V1)
    for config in [POLICY_V2, POLICY_V1]:
        try:
            local_dir = snapshot_download(
                repo_id=config.repo,
                cache_dir=cache_dir,
                ignore_patterns=["*.md"],
            )
            _POLICY_CACHE_DIR = local_dir
            _ACTIVE_POLICY_CONFIG = config
            _log.info("Topology policy downloaded: %s (%s)", config.repo, local_dir)
            return local_dir, config
        except Exception as exc:
            _log.debug("Policy %s unavailable: %s", config.repo, str(exc)[:80])

    _log.info("No topology policy available — using templates")
    return None, None


# Lazy-loaded model + tokenizer (never loaded at boot)
_POLICY_MODEL = None
_POLICY_TOKENIZER = None


def _format_prompt(task: str, config: PolicyModelConfig, system: int = 2) -> str:
    """Format the generation prompt based on the model's chat template."""
    if config.chat_template == "toolcall":
        # Local Qwen3-4B: uses <tool_call> JSON format with 2 SAGE tools
        # System prompt must match training exactly (sage_tool_schemas.py)
        # Prepend kNN classification so model knows task complexity
        system_hint = {1: "simple", 2: "moderate", 3: "complex"}
        complexity = system_hint.get(system, "moderate")
        user_msg = f"[Complexity: {complexity} (S{system})]\n\n{task[:2000]}"
        return (
            f"<|im_start|>system\n{_TOOLCALL_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    system_msg = (
        "You are a multi-agent topology designer for the YGN-SAGE framework. "
        "Given a coding task, design an optimal agent topology as a YAML DAG. "
        "Include: difficulty, reasoning, nodes (role + prompt + model_tier), "
        "edges (from_idx + to_idx + flow_type). The LAST node must be a "
        "synthesizer that returns the final answer."
    )
    if config.chat_template == "qwen3":
        return (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{task[:2000]}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    else:  # phi4
        return (
            f"<|system|>{system_msg}<|end|>\n"
            f"<|user|>{task[:2000]}<|end|>\n"
            "<|assistant|>"
        )


def generate_topology_from_policy(task: str, system: int = 2) -> dict | None:
    """Generate a topology using the learned policy (Path 6).

    Tries V2 (Nemotron-Orchestrator-8B GRPO) first, falls back to V1 (Phi-4-mini SFT).
    Lazy-loads the model on first call. Returns None if:
    - Model not available (no GPU, no download)
    - YAML/JSON parsing fails (fallback to templates)
    """
    global _POLICY_MODEL, _POLICY_TOKENIZER, _ACTIVE_POLICY_CONFIG

    # Lazy load on first call only
    if _POLICY_MODEL is None:
        adapter_path, config = download_policy_model()
        if adapter_path is None or config is None:
            return None
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if not torch.cuda.is_available():
                _log.debug("Path 6: no CUDA — skipping learned policy")
                return None

            _log.info("Path 6: loading %s (first call)...", config.repo)
            # Load tokenizer from base model (adapter tokenizer may have
            # broken extra_special_tokens on different transformers versions)
            import os as _os
            base_tok_path = _os.environ.get("SAGE_PATH6_MODEL", config.base_model)
            try:
                tok = AutoTokenizer.from_pretrained(
                    base_tok_path, trust_remote_code=config.trust_remote_code,
                )
            except Exception:
                tok = AutoTokenizer.from_pretrained(
                    adapter_path, trust_remote_code=config.trust_remote_code,
                    local_files_only=True,
                )

            if config.chat_template == "toolcall":
                # Local Qwen3-4B: base model in 4-bit NF4 + LoRA adapter
                from peft import PeftModel
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                import os as _os
                base_path = _os.environ.get("SAGE_PATH6_MODEL", config.base_model)
                base = AutoModelForCausalLM.from_pretrained(
                    base_path,
                    quantization_config=bnb_config,
                    trust_remote_code=config.trust_remote_code,
                    device_map="auto",
                )
                model = PeftModel.from_pretrained(base, adapter_path)
            elif config.chat_template == "qwen3":
                # V2: merged model (GRPO output is already merged)
                model = AutoModelForCausalLM.from_pretrained(
                    adapter_path,
                    trust_remote_code=config.trust_remote_code,
                    torch_dtype=torch.bfloat16,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                )
                model = model.to("cuda:0")
            else:
                # V1: base + LoRA adapter
                from peft import PeftModel
                base = AutoModelForCausalLM.from_pretrained(
                    config.base_model,
                    trust_remote_code=config.trust_remote_code,
                    dtype=torch.float16,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                )
                model = PeftModel.from_pretrained(base, adapter_path)
                model = model.to("cuda:0")
            model.eval()
            _POLICY_MODEL = model
            _POLICY_TOKENIZER = tok
            _ACTIVE_POLICY_CONFIG = config
            _log.info("Path 6: %s loaded on %s", config.repo, next(model.parameters()).device)
        except Exception as exc:
            _log.info("Path 6: model load failed (using templates): %s", str(exc)[:100])
            return None

    # Generate
    import torch

    prompt = _format_prompt(task, _ACTIVE_POLICY_CONFIG, system)
    inputs = _POLICY_TOKENIZER(prompt, return_tensors="pt").to(_POLICY_MODEL.device)

    try:
        with torch.no_grad():
            out = _POLICY_MODEL.generate(
                **inputs,
                max_new_tokens=_ACTIVE_POLICY_CONFIG.max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=_POLICY_TOKENIZER.eos_token_id,
            )
        raw = _POLICY_TOKENIZER.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )
    except Exception as exc:
        _log.debug("Path 6: generation failed: %s", str(exc)[:100])
        return None

    # Parse output based on format
    data = None
    if _ACTIVE_POLICY_CONFIG.chat_template == "toolcall":
        data = _parse_toolcall(raw)
        if data is None:
            _log.debug("Path 6: <tool_call> parse failed — fallback to template")
    else:
        # Legacy: YAML (superset of JSON — handles both)
        try:
            import yaml
            data = yaml.safe_load(raw)
        except Exception:
            _log.debug("Path 6: YAML parse failed — fallback to template")

    if isinstance(data, dict) and "nodes" in data and len(data.get("nodes", [])) > 0:
        _log.info(
            "Path 6 (%s): %d nodes, %d edges generated",
            _ACTIVE_POLICY_CONFIG.repo,
            len(data["nodes"]),
            len(data.get("edges", [])),
        )
        return data
    _log.debug("Path 6: parsed but no valid nodes")
    return None
