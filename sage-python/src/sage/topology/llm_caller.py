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
    models_str = ", ".join(available_models or ["gemini-2.5-flash"])
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
            role.get("model", "gemini-2.5-flash"),
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

    config_stage1 = LLMConfig(
        provider="google", model="gemini-2.5-flash", json_schema=roles_schema
    )
    config_stage2 = LLMConfig(
        provider="google", model="gemini-2.5-flash", json_schema=structure_schema
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
# SFT Policy Model (Path 6 in TopologyEngine)
# ---------------------------------------------------------------------------

HF_POLICY_REPO = "yannabadie/sage-topology-policy"
_POLICY_CACHE_DIR = None  # Populated by download_policy_model()


def download_policy_model(cache_dir: str | None = None) -> str | None:
    """Download the SFT topology policy from HuggingFace Hub.

    Returns the local path to the adapter directory, or None if unavailable.
    Falls back gracefully — the system uses templates if the policy is missing.
    """
    global _POLICY_CACHE_DIR
    if _POLICY_CACHE_DIR is not None:
        return _POLICY_CACHE_DIR

    try:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id=HF_POLICY_REPO,
            cache_dir=cache_dir,
            ignore_patterns=["*.md"],
        )
        _POLICY_CACHE_DIR = local_dir
        _log.info("Topology policy downloaded: %s", local_dir)
        return local_dir
    except ImportError:
        _log.debug("huggingface_hub not installed — topology policy unavailable")
    except Exception as exc:
        _log.info(
            "Topology policy download failed (using templates): %s", str(exc)[:100]
        )
    return None


# Lazy-loaded model + tokenizer (never loaded at boot)
_POLICY_MODEL = None
_POLICY_TOKENIZER = None


def generate_topology_from_policy(task: str) -> dict | None:
    """Generate a topology using the learned SFT policy (Path 6).

    Lazy-loads the model on first call. Returns None if:
    - Model not available (no GPU, no download)
    - YAML/JSON parsing fails (fallback to templates)
    """
    global _POLICY_MODEL, _POLICY_TOKENIZER

    # Lazy load on first call only
    if _POLICY_MODEL is None:
        adapter_path = download_policy_model()
        if adapter_path is None:
            return None
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            if not torch.cuda.is_available():
                _log.debug("Path 6: no CUDA — skipping learned policy")
                return None

            _log.info("Path 6: loading topology policy (first call)...")
            tok = AutoTokenizer.from_pretrained(
                adapter_path, trust_remote_code=False, local_files_only=True,
            )
            base = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-4-mini-instruct",
                trust_remote_code=False,
                dtype=torch.float16,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
            model = PeftModel.from_pretrained(base, adapter_path)
            model = model.to("cuda:0")
            model.eval()
            _POLICY_MODEL = model
            _POLICY_TOKENIZER = tok
            _log.info("Path 6: topology policy loaded on %s", next(model.parameters()).device)
        except Exception as exc:
            _log.info("Path 6: model load failed (using templates): %s", str(exc)[:100])
            return None

    # Generate
    import torch

    prompt = (
        "<|system|>You are a multi-agent topology designer. "
        "Given a task, generate an optimal agent topology in YAML format.<|end|>\n"
        f"<|user|>{task[:2000]}<|end|>\n"
        "<|assistant|>"
    )
    inputs = _POLICY_TOKENIZER(prompt, return_tensors="pt").to(_POLICY_MODEL.device)

    try:
        with torch.no_grad():
            out = _POLICY_MODEL.generate(
                **inputs,
                max_new_tokens=256,
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

    # Parse YAML (superset of JSON — handles both)
    try:
        import yaml
        data = yaml.safe_load(raw)
        if isinstance(data, dict) and "nodes" in data and len(data.get("nodes", [])) > 0:
            _log.info("Path 6 (learned policy): %d nodes generated", len(data["nodes"]))
            return data
        _log.debug("Path 6: parsed but no valid nodes")
    except Exception:
        _log.debug("Path 6: YAML parse failed — fallback to template")

    return None
