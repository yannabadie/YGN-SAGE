//! Template catalogue: 8 factory functions that produce pre-wired TopologyGraph instances.
//!
//! Each factory builds a complete topology with nodes, control edges, message edges,
//! and (where appropriate) state edges, gates, and conditions.

use super::topology_graph::*;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Extraction prompt for terminal/sink nodes in every topology.
/// Ensures the final output is concise and directly usable by downstream
/// systems (benchmarks, A2A clients, human users).
/// Based on MALT (arXiv 2412.01928) and AgentConductor (arXiv 2602.17100):
/// the terminal node's prompt constrains output format.
// pub(crate): exposed for the sink-drift test in routing::model_assigner —
// every node tagged with this string MUST also classify as sink via
// `is_sink_role(&node.role)`. Drift between templates and SINK_ROLES would
// over-promote the node on F7 task-tier escalation. Test enforces the
// contract automatically as new templates land.
pub(crate) const SINK_NODE_PROMPT: &str = "You are the final synthesizer. Review all context from previous agents and produce the definitive answer. Output ONLY the final answer — concise, no explanation, no reasoning. If the answer is a number, output only that number.";

// ---------------------------------------------------------------------------
// 1. Sequential: A -> B -> C
// ---------------------------------------------------------------------------

/// Build a sequential pipeline: input_processor -> worker -> output_formatter.
pub fn sequential(_model_id: &str) -> TopologyGraph {
    let mut g = TopologyGraph::try_new("sequential").unwrap();

    // Each node has a different system tier so ModelAssigner picks different
    // models/providers → real multi-provider execution.
    // model_id="" forces ModelAssigner to assign based on system tier.
    //
    // F6 audit fix (2026-04-18 docs/audits/2026-04-18-astropy-14995-*):
    // Planner was system=1 (fast tier, gemini-3.1-flash-lite). On SWE-bench
    // tasks the fast model looped on execute_bash tool calls without
    // emitting the plan (4/5 tasks emitted 50-char sentinel). F2 prompt
    // tightening and F5 tool-stripping both failed. Promoting planner to
    // system=2 (reasoner tier) aligns the template with its actual job:
    // analyzing a bug report, naming files, naming root causes — this is
    // reasoner-tier work, not summarization. Coder stays S2, synthesizer
    // stays S1 (sink formatting).
    let n0 = TopologyNode::new(
        "planner".into(),
        "".into(), // ModelAssigner will assign based on system=2 (reasoner)
        2,
        // A7 (2026-04-24): "tools" declared because AgentLoop grants tools
        // to this role at runtime. Without "tools", ModelAssigner's
        // `needs_tools && !card.supports_tools` filter (model_assigner.rs:289)
        // doesn't fire and kimi-k2.6 (supports_tools=false) can get
        // assigned here → HTTP 400 on the 4th tool-call turn → fast-abort.
        // See docs/benchmarks/2026-04-24-diff-verifier-observe-smoke/findings.md.
        vec!["text_processing".into(), "reasoning".into(), "tools".into()],
        0,
        0.5,
        60.0,
    );
    // F8 audit fix (2026-04-18 docs/audits/2026-04-18-astropy-14995-*):
    // Coder node was system=2 (10-step budget, reasoner tier). Smoke v7
    // at 15 tasks showed 9/15 non-passing are coder sentinels — coder
    // runs out of budget before emitting its diff. Bump to system=3
    // (20-step budget + top-reasoner tier; D8 stall_cap=17 still catches
    // 20-for-20 thrash). SWE-bench coders routinely do grep+read+run+
    // edit+verify chains that need 12-18 tool calls. This aligns the
    // coder's tier with the task-outer tier (SWE-bench classifies as S3).
    let n1 = TopologyNode::new(
        "coder".into(),
        "".into(), // ModelAssigner will assign based on system=3 (top reasoner)
        3,
        vec!["reasoning".into(), "tools".into(), "code_generation".into()],
        0,
        1.0,
        120.0,
    );
    let mut n2 = TopologyNode::new(
        "synthesizer".into(),
        "".into(), // ModelAssigner will assign based on system=1 (fast)
        1,
        vec!["text_processing".into()],
        0,
        0.5,
        60.0,
    );
    n2.prompt = SINK_NODE_PROMPT.to_string();

    let i0 = g.add_node(n0);
    let i1 = g.add_node(n1);
    let i2 = g.add_node(n2);

    g.try_add_edge(i0, i1, TopologyEdge::control()).unwrap();
    g.try_add_edge(i1, i2, TopologyEdge::control()).unwrap();

    g
}

// ---------------------------------------------------------------------------
// 2. Parallel: Source -> [W1..WN] -> Aggregator
// ---------------------------------------------------------------------------

/// Build a parallel fan-out/fan-in topology.
///
/// `worker_count` workers run concurrently; each sends a message edge to the aggregator
/// with field_mapping `{"result" -> "input_N"}`.
pub fn parallel(model_id: &str, worker_count: usize) -> TopologyGraph {
    let mut g = TopologyGraph::try_new("parallel").unwrap();

    let source = TopologyNode::new(
        "source".into(),
        model_id.into(),
        1,
        // A7 (2026-04-24): "tools" — see planner above for rationale.
        vec!["text_processing".into(), "tools".into()],
        0,
        0.5,
        60.0,
    );
    let si = g.add_node(source);

    let mut worker_indices = Vec::with_capacity(worker_count);
    for i in 0..worker_count {
        let w = TopologyNode::new(
            format!("worker_{}", i),
            model_id.into(),
            2,
            // A7 (2026-04-24): "tools" — see planner above for rationale.
            vec!["reasoning".into(), "tools".into()],
            0,
            1.0,
            120.0,
        );
        worker_indices.push(g.add_node(w));
    }

    let mut agg = TopologyNode::new(
        "aggregator".into(),
        model_id.into(),
        1,
        vec!["aggregation".into()],
        0,
        0.5,
        60.0,
    );
    agg.prompt = SINK_NODE_PROMPT.to_string();
    let ai = g.add_node(agg);

    for (i, &wi) in worker_indices.iter().enumerate() {
        // Control: source -> worker
        g.try_add_edge(si, wi, TopologyEdge::control()).unwrap();

        // Message: worker -> aggregator with field_mapping
        let mut mapping = HashMap::new();
        mapping.insert("result".to_string(), format!("input_{}", i));
        g.try_add_edge(wi, ai, TopologyEdge::message(Some(mapping)))
            .unwrap();
    }

    g
}

// ---------------------------------------------------------------------------
// 3. AVR: Act <-> Verify (with gated back-edge for repair)
// ---------------------------------------------------------------------------

/// Build an Act-Verify-Repair topology.
///
/// Forward path: actor -> verifier -> output (control edges).
/// Back-edge: verifier -> actor (control, gate=Closed) for repair.
/// Message edge: actor -> verifier with {"code" -> "review_input"}.
pub fn avr(_actor_model: &str, _reviewer_model: &str) -> TopologyGraph {
    let mut g = TopologyGraph::try_new("avr").unwrap();

    // Actor (S3 reasoner) and verifier (S2 fast) get different models/providers
    let actor = TopologyNode::new(
        "actor".into(),
        "".into(), // ModelAssigner: S3 → reasoner model
        3,
        vec!["code_generation".into(), "tools".into()],
        0,
        1.5,
        120.0,
    );
    let verifier = TopologyNode::new(
        "verifier".into(),
        "".into(), // ModelAssigner: S2 → fast model
        2,
        // A7 (2026-04-24): "tools" — see planner above for rationale.
        // Note: AVR verifier is NOT sink-prompted (unlike robust's
        // verifier), so AgentLoop gives it tools at runtime.
        vec!["code_review".into(), "tools".into()],
        0,
        1.0,
        60.0,
    );
    let output = TopologyNode::new(
        "output".into(),
        "".into(), // ModelAssigner: S1 → budget model
        1,
        // A7 (2026-04-24): "tools" — see planner above for rationale.
        // AVR's `output` has NO SINK_NODE_PROMPT so AgentLoop gives it tools.
        vec!["text_processing".into(), "tools".into()],
        0,
        0.5,
        60.0,
    );

    let ai = g.add_node(actor);
    let vi = g.add_node(verifier);
    let oi = g.add_node(output);

    // Forward control path
    g.try_add_edge(ai, vi, TopologyEdge::control()).unwrap();
    g.try_add_edge(vi, oi, TopologyEdge::control()).unwrap();

    // Back-edge: verifier -> actor (repair path, initially closed)
    g.try_add_edge(vi, ai, TopologyEdge::control().with_gate(Gate::Closed))
        .unwrap();

    // Message: actor -> verifier
    let mut mapping = HashMap::new();
    mapping.insert("code".to_string(), "review_input".to_string());
    g.try_add_edge(ai, vi, TopologyEdge::message(Some(mapping)))
        .unwrap();

    g
}

// ---------------------------------------------------------------------------
// 4. SelfMoA: Multiple agents + mixture aggregation
// ---------------------------------------------------------------------------

/// Build a Self-Mixture-of-Agents topology.
///
/// All agents receive the same input and work in parallel.
/// An aggregator performs weighted mixture blending.
pub fn self_moa(model_id: &str, agent_count: usize) -> TopologyGraph {
    let mut g = TopologyGraph::try_new("selfmoa").unwrap();

    let dispatcher = TopologyNode::new(
        "dispatcher".into(),
        model_id.into(),
        1,
        // A7 (2026-04-24): "tools" — see planner above for rationale.
        vec!["text_processing".into(), "tools".into()],
        0,
        0.5,
        60.0,
    );
    let di = g.add_node(dispatcher);

    let mut agent_indices = Vec::with_capacity(agent_count);
    for i in 0..agent_count {
        let agent = TopologyNode::new(
            format!("agent_{}", i),
            model_id.into(),
            2,
            // A7 (2026-04-24): "tools" — see planner above for rationale.
            vec!["reasoning".into(), "tools".into()],
            0,
            1.0,
            120.0,
        );
        agent_indices.push(g.add_node(agent));
    }

    let mut mixer = TopologyNode::new(
        "mixer".into(),
        model_id.into(),
        2,
        vec!["aggregation".into()],
        0,
        0.5,
        60.0,
    );
    mixer.prompt = SINK_NODE_PROMPT.to_string();
    let mi = g.add_node(mixer);

    for (i, &ai) in agent_indices.iter().enumerate() {
        g.try_add_edge(di, ai, TopologyEdge::control()).unwrap();

        let mut mapping = HashMap::new();
        mapping.insert("response".to_string(), format!("input_{}", i));
        g.try_add_edge(ai, mi, TopologyEdge::message(Some(mapping)))
            .unwrap();
    }

    g
}

// ---------------------------------------------------------------------------
// 5. Hierarchical: Parent -> [Children] -> Parent collects
// ---------------------------------------------------------------------------

/// Build a hierarchical delegation topology.
///
/// Parent delegates via control edges, children report back via state edges.
pub fn hierarchical(parent_model: &str, child_model: &str) -> TopologyGraph {
    let mut g = TopologyGraph::try_new("hierarchical").unwrap();

    let parent = TopologyNode::new(
        "parent".into(),
        parent_model.into(),
        2,
        // A7 (2026-04-24): "tools" — see planner above for rationale.
        vec!["planning".into(), "tools".into()],
        1,
        1.5,
        120.0,
    );
    let child1 = TopologyNode::new(
        "child_0".into(),
        child_model.into(),
        1,
        // A7 (2026-04-24): "tools" — see planner above for rationale.
        vec!["reasoning".into(), "tools".into()],
        1, // same label as parent (children inherit parent's security context)
        1.0,
        60.0,
    );
    let child2 = TopologyNode::new(
        "child_1".into(),
        child_model.into(),
        1,
        // A7 (2026-04-24): "tools" — see planner above for rationale.
        vec!["reasoning".into(), "tools".into()],
        1, // same label as parent
        1.0,
        60.0,
    );

    let pi = g.add_node(parent);
    let c1i = g.add_node(child1);
    let c2i = g.add_node(child2);

    // Parent delegates
    g.try_add_edge(pi, c1i, TopologyEdge::control()).unwrap();
    g.try_add_edge(pi, c2i, TopologyEdge::control()).unwrap();

    // Children report back via state edges
    g.try_add_edge(c1i, pi, TopologyEdge::state()).unwrap();
    g.try_add_edge(c2i, pi, TopologyEdge::state()).unwrap();

    g
}

// ---------------------------------------------------------------------------
// 6. Hub: Central coordinator + spoke delegation
// ---------------------------------------------------------------------------

/// Build a hub-and-spoke topology.
///
/// Hub node is connected to N spoke nodes via control + message edges.
/// Spokes report back via state edges. Switch conditions on hub->spoke edges.
pub fn hub(coordinator_model: &str, spoke_model: &str, spoke_count: usize) -> TopologyGraph {
    let mut g = TopologyGraph::try_new("hub").unwrap();

    let coordinator = TopologyNode::new(
        "coordinator".into(),
        coordinator_model.into(),
        2,
        // A7 (2026-04-24): "tools" — see planner above for rationale.
        vec!["planning".into(), "delegation".into(), "tools".into()],
        1,
        1.0,
        120.0,
    );
    let ci = g.add_node(coordinator);

    for i in 0..spoke_count {
        let spoke = TopologyNode::new(
            format!("spoke_{}", i),
            spoke_model.into(),
            1,
            // A7 (2026-04-24): "tools" — see planner above for rationale.
            vec!["reasoning".into(), "tools".into()],
            1, // same label as coordinator (spokes inherit hub's security context)
            1.0,
            60.0,
        );
        let si = g.add_node(spoke);

        // Hub -> spoke: control with switch condition + message
        g.try_add_edge(
            ci,
            si,
            TopologyEdge::control().with_condition(format!("task_type == 'type_{}'", i)),
        )
        .unwrap();

        let mut mapping = HashMap::new();
        mapping.insert("task".to_string(), "input".to_string());
        g.try_add_edge(ci, si, TopologyEdge::message(Some(mapping)))
            .unwrap();

        // Spoke -> hub: state (reporting)
        g.try_add_edge(si, ci, TopologyEdge::state()).unwrap();
    }

    g
}

// ---------------------------------------------------------------------------
// 7. Debate: Agent A vs Agent B, Judge C
// ---------------------------------------------------------------------------

/// Build a debate topology.
///
/// topic_setter fans out to debater_a and debater_b in parallel,
/// both send their arguments to a judge node.
pub fn debate(debater_model: &str, judge_model: &str) -> TopologyGraph {
    let mut g = TopologyGraph::try_new("debate").unwrap();

    let topic = TopologyNode::new(
        "topic_setter".into(),
        debater_model.into(),
        1,
        // A7 (2026-04-24): "tools" — see planner above for rationale.
        vec!["text_processing".into(), "tools".into()],
        0,
        0.5,
        60.0,
    );
    let debater_a = TopologyNode::new(
        "debater_a".into(),
        debater_model.into(),
        2,
        // A7 (2026-04-24): "tools" — see planner above for rationale.
        vec!["reasoning".into(), "tools".into()],
        0,
        1.0,
        120.0,
    );
    let debater_b = TopologyNode::new(
        "debater_b".into(),
        debater_model.into(),
        2,
        // A7 (2026-04-24): "tools" — see planner above for rationale.
        vec!["reasoning".into(), "tools".into()],
        0,
        1.0,
        120.0,
    );
    let mut judge = TopologyNode::new(
        "judge".into(),
        judge_model.into(),
        2,
        vec!["evaluation".into()],
        0,
        1.0,
        60.0,
    );
    judge.prompt = SINK_NODE_PROMPT.to_string();

    let ti = g.add_node(topic);
    let dai = g.add_node(debater_a);
    let dbi = g.add_node(debater_b);
    let ji = g.add_node(judge);

    // topic -> A, topic -> B (parallel start)
    g.try_add_edge(ti, dai, TopologyEdge::control()).unwrap();
    g.try_add_edge(ti, dbi, TopologyEdge::control()).unwrap();

    // A -> judge, B -> judge (message with "argument" field mapping)
    let mut mapping_a = HashMap::new();
    mapping_a.insert("argument".to_string(), "argument_a".to_string());
    g.try_add_edge(dai, ji, TopologyEdge::message(Some(mapping_a)))
        .unwrap();

    let mut mapping_b = HashMap::new();
    mapping_b.insert("argument".to_string(), "argument_b".to_string());
    g.try_add_edge(dbi, ji, TopologyEdge::message(Some(mapping_b)))
        .unwrap();

    g
}

// ---------------------------------------------------------------------------
// 8. Brainstorming: N thinkers diverge, then converge
// ---------------------------------------------------------------------------

/// Build a brainstorming topology.
///
/// N thinker agents each get the full task, then a synthesizer converges all ideas.
pub fn brainstorming(model_id: &str, thinker_count: usize) -> TopologyGraph {
    let mut g = TopologyGraph::try_new("brainstorming").unwrap();

    let prompt = TopologyNode::new(
        "prompt".into(),
        model_id.into(),
        1,
        // A7 (2026-04-24): "tools" — see planner above for rationale.
        vec!["text_processing".into(), "tools".into()],
        0,
        0.5,
        60.0,
    );
    let pi = g.add_node(prompt);

    let mut thinker_indices = Vec::with_capacity(thinker_count);
    for i in 0..thinker_count {
        let thinker = TopologyNode::new(
            format!("thinker_{}", i),
            model_id.into(),
            2,
            // A7 (2026-04-24): "tools" — see planner above for rationale.
            vec!["reasoning".into(), "creativity".into(), "tools".into()],
            0,
            1.0,
            120.0,
        );
        thinker_indices.push(g.add_node(thinker));
    }

    let mut synthesizer = TopologyNode::new(
        "synthesizer".into(),
        model_id.into(),
        2,
        vec!["aggregation".into(), "synthesis".into()],
        0,
        0.5,
        60.0,
    );
    synthesizer.prompt = SINK_NODE_PROMPT.to_string();
    let si = g.add_node(synthesizer);

    for (i, &ti) in thinker_indices.iter().enumerate() {
        g.try_add_edge(pi, ti, TopologyEdge::control()).unwrap();

        let mut mapping = HashMap::new();
        mapping.insert("idea".to_string(), format!("input_{}", i));
        g.try_add_edge(ti, si, TopologyEdge::message(Some(mapping)))
            .unwrap();
    }

    g
}

// ---------------------------------------------------------------------------
// 9. Robust: Preprocessor -> [Workers] -> Verifier (majority vote)
// ---------------------------------------------------------------------------

/// Build a robust topology for noise-resilient tasks (MASBENCH robustness axis).
///
/// A preprocessor strips noise, then `worker_count` workers solve independently,
/// and a verifier performs majority voting. Based on MALT (arXiv 2412.01928)
/// and ResMAS (arXiv 2601.04694).
pub fn robust(_model_id: &str, worker_count: usize) -> TopologyGraph {
    let mut g = TopologyGraph::try_new("robust").unwrap();

    // Preprocessor (S1 fast): strips noise/distractors from input
    let mut preprocessor = TopologyNode::new(
        "preprocessor".into(),
        "".into(),
        1,
        // A7 (2026-04-24): "tools" — see planner above for rationale.
        vec![
            "text_processing".into(),
            "noise_filter".into(),
            "tools".into(),
        ],
        0,
        0.5,
        60.0,
    );
    preprocessor.fallback_tier = "fast".into();
    let pi = g.add_node(preprocessor);

    // Workers (S2 reasoner): solve cleaned problem independently
    let mut worker_indices = Vec::with_capacity(worker_count);
    for i in 0..worker_count {
        let mut worker = TopologyNode::new(
            format!("worker_{}", i),
            "".into(),
            2,
            // A7 (2026-04-24): "tools" — see planner above for rationale.
            vec!["reasoning".into(), "math".into(), "tools".into()],
            0,
            1.0,
            120.0,
        );
        worker.fallback_tier = "reasoner".into();
        worker_indices.push(g.add_node(worker));
    }

    // Verifier (S3 reasoner): majority vote across worker outputs
    let mut verifier = TopologyNode::new(
        "verifier".into(),
        "".into(),
        3,
        vec!["aggregation".into(), "verification".into()],
        0,
        2.0,
        180.0,
    );
    verifier.fallback_tier = "reasoner".into();
    verifier.is_checkpoint = true;
    verifier.prompt = SINK_NODE_PROMPT.to_string();
    let vi = g.add_node(verifier);

    for (i, &wi) in worker_indices.iter().enumerate() {
        // Control: preprocessor -> worker
        g.try_add_edge(pi, wi, TopologyEdge::control()).unwrap();

        // Message: preprocessor -> worker with cleaned input
        let mut pre_mapping = HashMap::new();
        pre_mapping.insert("cleaned_input".to_string(), "input".to_string());
        g.try_add_edge(pi, wi, TopologyEdge::message(Some(pre_mapping)))
            .unwrap();

        // Message: worker -> verifier with result
        let mut result_mapping = HashMap::new();
        result_mapping.insert("result".to_string(), format!("input_{}", i));
        g.try_add_edge(wi, vi, TopologyEdge::message(Some(result_mapping)))
            .unwrap();
    }

    g
}

// ---------------------------------------------------------------------------
// 10. HorizonPipeline: Splitter -> [Stage0 -> Stage1 -> ...] -> Aggregator
// ---------------------------------------------------------------------------

/// Build a horizon pipeline for sequential multi-step tasks (MASBENCH horizon axis).
///
/// A splitter decomposes the task, then `stage_count` stages solve sub-problems
/// sequentially (each receiving prior stage output as context), and an aggregator
/// combines all results. Based on Task-Decoupled Planning (arXiv 2601.07577).
pub fn horizon_pipeline(_model_id: &str, stage_count: usize) -> TopologyGraph {
    let mut g = TopologyGraph::try_new("horizon_pipeline").unwrap();

    // Splitter (S1 fast): parses <<horizon>> delimiters, dispatches sub-problems
    let mut splitter = TopologyNode::new(
        "splitter".into(),
        "".into(),
        1,
        // A7 (2026-04-24): "tools" — see planner above for rationale.
        vec![
            "text_processing".into(),
            "decomposition".into(),
            "tools".into(),
        ],
        0,
        0.5,
        60.0,
    );
    splitter.fallback_tier = "fast".into();
    let si = g.add_node(splitter);

    // Stages (S2 reasoner): each solves one sub-problem with prior context
    let mut stage_indices = Vec::with_capacity(stage_count);
    for i in 0..stage_count {
        let mut stage = TopologyNode::new(
            format!("stage_{}", i),
            "".into(),
            2,
            // A7 (2026-04-24): "tools" — see planner above for rationale.
            vec!["reasoning".into(), "math".into(), "tools".into()],
            0,
            1.0,
            120.0,
        );
        stage.fallback_tier = "reasoner".into();
        stage_indices.push(g.add_node(stage));
    }

    // Aggregator (S1 fast): combines all stage outputs
    let mut aggregator = TopologyNode::new(
        "aggregator".into(),
        "".into(),
        1,
        vec!["aggregation".into(), "text_processing".into()],
        0,
        0.5,
        60.0,
    );
    aggregator.fallback_tier = "fast".into();
    aggregator.prompt = SINK_NODE_PROMPT.to_string();
    let ai = g.add_node(aggregator);

    // Chain: splitter -> stage_0 -> stage_1 -> ... -> stage_{N-1} -> aggregator
    // Each hop has control + message edges
    let mut prev = si;
    for &stage_idx in stage_indices.iter() {
        g.try_add_edge(prev, stage_idx, TopologyEdge::control())
            .unwrap();

        let mut mapping = HashMap::new();
        if prev == si {
            mapping.insert("sub_problem_0".to_string(), "input".to_string());
        } else {
            mapping.insert("result".to_string(), "prior_context".to_string());
        }
        g.try_add_edge(prev, stage_idx, TopologyEdge::message(Some(mapping)))
            .unwrap();

        prev = stage_idx;
    }

    // Last stage -> aggregator
    g.try_add_edge(prev, ai, TopologyEdge::control()).unwrap();
    let mut final_mapping = HashMap::new();
    final_mapping.insert("result".to_string(), "final_input".to_string());
    g.try_add_edge(prev, ai, TopologyEdge::message(Some(final_mapping)))
        .unwrap();

    g
}

// ---------------------------------------------------------------------------
// 11. ParallelFanout: Dispatcher -> [Diverse Workers] -> Aggregator
// ---------------------------------------------------------------------------

/// Build a parallel fan-out topology with diverse worker tiers (MASBENCH parallel axis).
///
/// Workers cycle through S1/S2/S3 system tiers so ModelAssigner routes them to
/// different providers/models for output diversity. Based on SC-MAS (arXiv 2601.09434).
pub fn parallel_fanout(_model_id: &str, worker_count: usize) -> TopologyGraph {
    let mut g = TopologyGraph::try_new("parallel_fanout").unwrap();

    // Dispatcher (S1 fast): decomposes and fans out
    let mut dispatcher = TopologyNode::new(
        "dispatcher".into(),
        "".into(),
        1,
        // A7 (2026-04-24): "tools" — see planner above for rationale.
        vec![
            "text_processing".into(),
            "decomposition".into(),
            "tools".into(),
        ],
        0,
        0.5,
        60.0,
    );
    dispatcher.fallback_tier = "fast".into();
    let di = g.add_node(dispatcher);

    // Workers: cycle S1/S2/S3 for diversity
    let tier_cycle: [u8; 3] = [1, 2, 3];
    let fallback_cycle: [&str; 3] = ["fast", "reasoner", "reasoner"];

    let mut worker_indices = Vec::with_capacity(worker_count);
    for i in 0..worker_count {
        let system = tier_cycle[i % 3];
        let mut worker = TopologyNode::new(
            format!("worker_{}", i),
            "".into(),
            system,
            // A7 (2026-04-24): "tools" — see planner above for rationale.
            vec!["reasoning".into(), "tools".into()],
            0,
            if system >= 2 { 1.0 } else { 0.5 },
            if system >= 2 { 120.0 } else { 60.0 },
        );
        worker.fallback_tier = fallback_cycle[i % 3].into();
        worker_indices.push(g.add_node(worker));
    }

    // Aggregator (S2 reasoner): high-quality merge
    let mut aggregator = TopologyNode::new(
        "aggregator".into(),
        "".into(),
        2,
        vec!["aggregation".into(), "synthesis".into()],
        0,
        1.5,
        120.0,
    );
    aggregator.fallback_tier = "reasoner".into();
    aggregator.is_checkpoint = true;
    aggregator.prompt = SINK_NODE_PROMPT.to_string();
    let ai = g.add_node(aggregator);

    for (i, &wi) in worker_indices.iter().enumerate() {
        // Control: dispatcher -> worker
        g.try_add_edge(di, wi, TopologyEdge::control()).unwrap();

        // Message: dispatcher -> worker with sub-problem
        let mut dispatch_mapping = HashMap::new();
        dispatch_mapping.insert("sub_problem".to_string(), "input".to_string());
        g.try_add_edge(di, wi, TopologyEdge::message(Some(dispatch_mapping)))
            .unwrap();

        // Message: worker -> aggregator with result
        let mut result_mapping = HashMap::new();
        result_mapping.insert("result".to_string(), format!("input_{}", i));
        g.try_add_edge(wi, ai, TopologyEdge::message(Some(result_mapping)))
            .unwrap();
    }

    g
}

// ---------------------------------------------------------------------------
// TemplateStore
// ---------------------------------------------------------------------------
// 12. FormalSolver: Formalizer (LLM) → Solver (Rust deterministic)
// ---------------------------------------------------------------------------

/// 2-node formal solver topology: LLM formalizes, Rust solves.
///
/// The formalizer node translates natural language into a list of equations
/// (variable = expression, one per line). The solver node parses these
/// equations and evaluates them deterministically via `solve_equation_system()`.
///
/// Based on SatLM (NeurIPS 2023): separating formalization (LLM's strength)
/// from solving (deterministic computation) gives +23% on hard math.
/// Generic — works for any math problem, not just iGSM.
pub fn formal_solver(_model_id: &str) -> TopologyGraph {
    let mut g = TopologyGraph::try_new("formal_solver").unwrap();

    // Formalizer (LLM): translates NL to equations.
    // Pinned to deepseek-chat: fast, cheap, reliable for math formalization.
    // OpenRouter is unreliable (circuit breaker opens frequently).
    let mut formalizer = TopologyNode::new(
        "formalizer".into(),
        "deepseek-chat".into(),
        2, // S2 reasoner — needs to understand the problem
        // A7 (2026-04-24): "tools" — see planner above for rationale.
        vec!["reasoning".into(), "math".into(), "tools".into()],
        0,
        1.0,
        120.0,
    );
    formalizer.prompt = concat!(
        "You are a math formalizer. Convert this word problem into a system of equations.\n",
        "Rules:\n",
        "- Write ONE equation per line: variable_name = expression\n",
        "- Use lowercase_with_underscores for names (entity_attribute)\n",
        "- Use only +, -, * operators and integer constants\n",
        "- Resolve each variable to a concrete value when possible\n",
        "- The LAST line MUST be: ANSWER = variable_name\n",
        "- Output ONLY equations, no explanation or markdown\n",
        "\n",
        "Example 1:\n",
        "zoo_a_pelican = 12\n",
        "zoo_a_eagle = zoo_a_pelican + 3\n",
        "zoo_b_pelican = zoo_a_pelican\n",
        "zoo_b_eagle = zoo_b_pelican * 2\n",
        "ANSWER = zoo_b_eagle\n",
        "\n",
        "Example 2:\n",
        "factory_a_widget = 100\n",
        "factory_b_widget = factory_a_widget - 20\n",
        "factory_b_gear = factory_b_widget * 3\n",
        "total = factory_b_gear + factory_b_widget\n",
        "ANSWER = total",
    )
    .to_string();

    // Solver (deterministic): evaluates equations via Rust
    let mut solver = TopologyNode::new(
        "solver".into(),
        "".into(),
        1, // S1 fast — no LLM call, pure computation
        vec!["math".into()],
        0,
        0.1, // Very cheap
        5.0, // Very fast
    );
    solver.prompt = SINK_NODE_PROMPT.to_string();

    let fi = g.add_node(formalizer);
    let si = g.add_node(solver);

    g.try_add_edge(fi, si, TopologyEdge::control()).unwrap();
    let mut mapping = HashMap::new();
    mapping.insert("equations".to_string(), "input".to_string());
    g.try_add_edge(fi, si, TopologyEdge::message(Some(mapping)))
        .unwrap();

    g
}

// ---------------------------------------------------------------------------
// TemplateStore
// ---------------------------------------------------------------------------

/// Registry that creates topologies from template names.
pub struct TemplateStore;

impl TemplateStore {
    /// Create a topology from a template name and default model.
    pub fn create(template_name: &str, model_id: &str) -> Result<TopologyGraph, String> {
        match template_name.to_lowercase().as_str() {
            "sequential" => Ok(sequential(model_id)),
            "parallel" => Ok(parallel(model_id, 3)),
            "avr" => Ok(avr(model_id, model_id)),
            "selfmoa" | "self_moa" | "self-moa" => Ok(self_moa(model_id, 3)),
            "hierarchical" => Ok(hierarchical(model_id, model_id)),
            "hub" => Ok(hub(model_id, model_id, 3)),
            "debate" => Ok(debate(model_id, model_id)),
            "brainstorming" => Ok(brainstorming(model_id, 3)),
            "robust" => Ok(robust(model_id, 3)),
            "horizon_pipeline" | "horizon-pipeline" => Ok(horizon_pipeline(model_id, 3)),
            "parallel_fanout" | "parallel-fanout" => Ok(parallel_fanout(model_id, 5)),
            "formal_solver" | "formal-solver" => Ok(formal_solver(model_id)),
            _ => Err(format!("Unknown template: {}", template_name)),
        }
    }

    /// List all available template names.
    pub fn available() -> Vec<&'static str> {
        vec![
            "sequential",
            "parallel",
            "avr",
            "selfmoa",
            "hierarchical",
            "hub",
            "debate",
            "brainstorming",
            "robust",
            "horizon_pipeline",
            "parallel_fanout",
            "formal_solver",
        ]
    }
}

// ---------------------------------------------------------------------------
// PyO3 wrapper
// ---------------------------------------------------------------------------

/// PyO3-exposed template store for creating topologies from template names.
#[pyclass]
pub struct PyTemplateStore;

#[pymethods]
impl PyTemplateStore {
    #[new]
    pub fn new() -> Self {
        Self
    }

    /// Create a topology from a template name and default model ID.
    pub fn create(&self, template_name: &str, model_id: &str) -> PyResult<TopologyGraph> {
        TemplateStore::create(template_name, model_id)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// List all available template names.
    pub fn available(&self) -> Vec<String> {
        TemplateStore::available()
            .into_iter()
            .map(String::from)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "TemplateStore(templates={})",
            TemplateStore::available().len()
        )
    }
}

impl Default for PyTemplateStore {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_structure() {
        let g = sequential("model-a");
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 2);
        assert!(g.is_acyclic());
        assert_eq!(g.entry_nodes().len(), 1);
        assert_eq!(g.exit_nodes().len(), 1);
    }

    #[test]
    fn test_sequential_planner_tier_is_s2() {
        // F6 audit fix (2026-04-18 docs/audits/2026-04-18-astropy-14995-*):
        // planner must be system=2 (reasoner-tier), not system=1. S1 fast
        // models loop on tool calls without emitting the plan; reasoner-
        // tier models complete the plan and hand off to the coder.
        // Empirical proof: smoke v6 jumped 1/5 → 3/5 real patches after
        // this single-line tier bump. If someone drops the planner back
        // to S1 "for cost reasons," this test catches it.
        let g = sequential("model-a");
        let planner = g.try_get_node(0).expect("sequential has 3 nodes");
        assert_eq!(planner.role, "planner");
        assert_eq!(
            planner.system, 2,
            "Sequential planner must be system=2 (reasoner). F6 fix; \
             dropping to system=1 regressed astropy-14995 from real \
             patch to 52-char sentinel in all earlier smokes."
        );
        // F8 audit fix (2026-04-18): coder bumped S2→S3 after v7 showed
        // 9/15 non-passing were coder sentinels. S3 gives 20-step budget
        // (vs S2=10) for grep+read+run+edit+verify chains.
        let coder = g.try_get_node(1).expect("sequential has 3 nodes");
        assert_eq!(coder.role, "coder");
        assert_eq!(
            coder.system, 3,
            "Sequential coder must be system=3 (top reasoner, 20-step \
             budget). F8 fix; reverting to system=2 re-introduces the \
             10-step limit that blocked 9/15 tasks in smoke v7."
        );
        // Synthesizer remains S1 (sink formatting, cheap is fine)
        let synth = g.try_get_node(2).expect("sequential has 3 nodes");
        assert_eq!(synth.role, "synthesizer");
        assert_eq!(synth.system, 1);
    }

    #[test]
    fn test_parallel_structure() {
        let g = parallel("model-a", 4);
        // 1 source + 4 workers + 1 aggregator = 6 nodes
        assert_eq!(g.node_count(), 6);
        // 4 control (source->worker) + 4 message (worker->agg) = 8
        assert_eq!(g.edge_count(), 8);
        assert!(g.is_acyclic());
    }

    #[test]
    fn test_avr_structure() {
        let g = avr("actor-model", "reviewer-model");
        assert_eq!(g.node_count(), 3);
        // 2 control forward + 1 control back-edge + 1 message = 4
        assert_eq!(g.edge_count(), 4);
        // Has cycle due to back-edge (even though gated)
        assert!(!g.is_acyclic());
    }

    #[test]
    fn test_self_moa_structure() {
        let g = self_moa("model-a", 5);
        // 1 dispatcher + 5 agents + 1 mixer = 7
        assert_eq!(g.node_count(), 7);
        // 5 control + 5 message = 10
        assert_eq!(g.edge_count(), 10);
        assert!(g.is_acyclic());
    }

    #[test]
    fn test_hierarchical_structure() {
        let g = hierarchical("parent-model", "child-model");
        assert_eq!(g.node_count(), 3);
        // 2 control (parent->child) + 2 state (child->parent) = 4
        assert_eq!(g.edge_count(), 4);
        // Has cycle (state edges form cycle)
        assert!(!g.is_acyclic());
    }

    #[test]
    fn test_hub_structure() {
        let g = hub("coord-model", "spoke-model", 3);
        // 1 coordinator + 3 spokes = 4
        assert_eq!(g.node_count(), 4);
        // Per spoke: 1 control + 1 message + 1 state = 3 edges x 3 spokes = 9
        assert_eq!(g.edge_count(), 9);
    }

    #[test]
    fn test_debate_structure() {
        let g = debate("debater-model", "judge-model");
        assert_eq!(g.node_count(), 4);
        // 2 control (topic->debaters) + 2 message (debaters->judge) = 4
        assert_eq!(g.edge_count(), 4);
        assert!(g.is_acyclic());
    }

    #[test]
    fn test_brainstorming_structure() {
        let g = brainstorming("model-a", 3);
        // 1 prompt + 3 thinkers + 1 synthesizer = 5
        assert_eq!(g.node_count(), 5);
        // 3 control + 3 message = 6
        assert_eq!(g.edge_count(), 6);
        assert!(g.is_acyclic());
    }

    #[test]
    fn test_template_store_create_all() {
        for name in TemplateStore::available() {
            let result = TemplateStore::create(name, "test-model");
            assert!(result.is_ok(), "Failed to create template '{}'", name);
            assert!(result.unwrap().node_count() > 0);
        }
    }

    #[test]
    fn test_template_store_unknown() {
        let result = TemplateStore::create("nonexistent", "model");
        assert!(result.is_err());
        assert!(result.err().unwrap().contains("Unknown template"));
    }

    #[test]
    fn test_template_store_available() {
        let names = TemplateStore::available();
        assert_eq!(names.len(), 12);
    }

    // ── New MASBENCH-axis templates ──────────────────────────────────────

    #[test]
    fn test_robust_structure() {
        let g = robust("", 3);
        // preprocessor + 3 workers + verifier = 5 nodes
        assert_eq!(g.node_count(), 5);
        // 3 control (pre→workers) + 3 message (pre→workers) + 3 message (workers→verifier) = 9
        assert_eq!(g.edge_count(), 9);
        assert!(g.is_acyclic());
        assert_eq!(g.entry_nodes().len(), 1); // preprocessor
        assert_eq!(g.exit_nodes().len(), 1); // verifier
    }

    #[test]
    fn test_robust_diverse_workers() {
        let g = robust("", 5);
        // preprocessor + 5 workers + verifier = 7 nodes
        assert_eq!(g.node_count(), 7);
        // 5 control + 5 message (pre→workers) + 5 message (workers→verifier) = 15
        assert_eq!(g.edge_count(), 15);
    }

    #[test]
    fn test_horizon_pipeline_structure() {
        let g = horizon_pipeline("", 3);
        // splitter + 3 stages + aggregator = 5 nodes
        assert_eq!(g.node_count(), 5);
        // Chain: splitter→s0→s1→s2→agg = 4 hops × 2 edges (control+message) = 8
        assert_eq!(g.edge_count(), 8);
        assert!(g.is_acyclic());
        assert_eq!(g.entry_nodes().len(), 1); // splitter
        assert_eq!(g.exit_nodes().len(), 1); // aggregator
    }

    #[test]
    fn test_horizon_pipeline_single_stage() {
        let g = horizon_pipeline("", 1);
        // splitter + 1 stage + aggregator = 3 nodes
        assert_eq!(g.node_count(), 3);
        // 2 hops × 2 = 4 edges
        assert_eq!(g.edge_count(), 4);
        assert!(g.is_acyclic());
    }

    #[test]
    fn test_parallel_fanout_structure() {
        let g = parallel_fanout("", 5);
        // dispatcher + 5 workers + aggregator = 7 nodes
        assert_eq!(g.node_count(), 7);
        // 5 control + 5 message (dispatch→workers) + 5 message (workers→agg) = 15
        assert_eq!(g.edge_count(), 15);
        assert!(g.is_acyclic());
        assert_eq!(g.entry_nodes().len(), 1); // dispatcher
        assert_eq!(g.exit_nodes().len(), 1); // aggregator
    }

    #[test]
    fn test_parallel_fanout_tier_diversity() {
        let g = parallel_fanout("", 6);
        // 6 workers should cycle S1, S2, S3, S1, S2, S3
        assert_eq!(g.node_count(), 8); // dispatcher + 6 workers + aggregator
    }

    #[test]
    fn test_template_store_create_new_templates() {
        assert!(TemplateStore::create("robust", "").is_ok());
        assert!(TemplateStore::create("horizon_pipeline", "").is_ok());
        assert!(TemplateStore::create("horizon-pipeline", "").is_ok());
        assert!(TemplateStore::create("parallel_fanout", "").is_ok());
        assert!(TemplateStore::create("parallel-fanout", "").is_ok());
    }

    // ── A7 (2026-04-24): tool-capability hygiene ─────────────────────────
    //
    // Every template node whose role is NOT sink-prompted (i.e. doesn't
    // carry ``SINK_NODE_PROMPT``) is executed by Python's ``AgentLoop``
    // which unconditionally grants tools. ``ModelAssigner`` only
    // filters ``kimi-k2.6`` (``supports_tools=false``) when the node
    // declares ``"tools"`` in ``required_capabilities`` — without it
    // kimi gets assigned and HTTP-400s on the 4th tool-call turn
    // (F9 `cards.toml:598`, reproduced 2026-04-23 + 2026-04-24 smokes
    // on astropy-14182 and astropy-7746 — 20% deterministic fast-abort).
    //
    // This test locks the contract: if someone adds a new template or
    // adds a new role to an existing template without "tools", the test
    // fires with a pointer to A7's rationale.

    /// Roles whose prompt is ``SINK_NODE_PROMPT`` in our templates.
    /// Kept narrow — these are tool-free by construction, and adding
    /// "tools" to them would exclude kimi from a perfectly legitimate
    /// text-only synthesis role.
    ///
    /// Residual risk (advisor review, 2026-04-24): AgentLoop still
    /// provides the tool registry to sink nodes at runtime.
    /// SINK_NODE_PROMPT asks for "Output ONLY the final answer — concise,
    /// no explanation, no reasoning", but the LLM ultimately decides
    /// whether to invoke a tool. If a sink bandit-assigned to kimi-k2.6
    /// nevertheless emits 4+ tool-call turns, the Moonshot HTTP 400
    /// ("reasoning_content missing") relapse is possible. Empirically
    /// this hasn't triggered on current sinks (they are terminal
    /// formatters the model treats as 1-turn), but the attack surface
    /// is NARROWED, not closed. Proper close-out is B9 (AgentLoop
    /// honours ``required_capabilities`` and serves a tool-free agent
    /// variant for sinks). Until then, observe-mode logs will flag any
    /// kimi-on-sink 4th-turn-400 relapse for triage.
    const TOOL_FREE_SINK_ROLES: &[&str] = &[
        "synthesizer", // sequential, brainstorming
        "aggregator",  // parallel, horizon_pipeline, parallel_fanout
        "mixer",       // selfmoa
        "judge",       // debate
        "verifier",    // robust (ONLY — avr's verifier is not sink-prompted)
        "solver",      // formal_solver (deterministic Rust, not LLM)
    ];

    fn is_tool_free_sink(template: &str, role: &str) -> bool {
        // Special case: "verifier" is sink-prompted in `robust` but
        // NOT in `avr`. Distinguish by template.
        if role == "verifier" {
            return template == "robust";
        }
        TOOL_FREE_SINK_ROLES.iter().any(|&r| r == role)
    }

    fn assert_tools_policy(template_name: &str, g: &TopologyGraph) {
        for idx in 0..g.node_count() {
            let node = g.try_get_node(idx).expect("node index in range");
            let has_tools = node.required_capabilities.iter().any(|c| c == "tools");
            if is_tool_free_sink(template_name, &node.role) {
                // Sink-prompted roles stay tool-free — kimi-k2.6 is
                // explicitly allowed here (F9 intent).
                assert!(
                    !has_tools,
                    "Template `{}` node `{}` is a sink (SINK_NODE_PROMPT) \
                     but declares \"tools\" in required_capabilities. Sinks \
                     are 1-turn text-only roles; they don't need tool \
                     capability and keeping them tool-free preserves \
                     kimi-k2.6 as a routing option for text-only synthesis.",
                    template_name, node.role
                );
            } else {
                assert!(
                    has_tools,
                    "A7 (2026-04-24) violation: template `{}` node `{}` \
                     (role does NOT have SINK_NODE_PROMPT, so AgentLoop \
                     grants it tools at runtime) is missing \"tools\" in \
                     required_capabilities. Without it, \
                     ModelAssigner.supports_tools filter doesn't fire and \
                     kimi-k2.6 (supports_tools=false) can be assigned here \
                     → HTTP 400 on the 4th tool-call turn → fast-abort. \
                     Add \"tools\" to this node's required_capabilities. \
                     See docs/benchmarks/2026-04-24-diff-verifier-observe-\
                     smoke/findings.md for the empirical trace.",
                    template_name, node.role
                );
            }
        }
    }

    #[test]
    fn test_a7_tool_capability_hygiene_all_templates() {
        // Every template must either declare "tools" on non-sink nodes
        // or be explicitly excluded via TOOL_FREE_SINK_ROLES.
        assert_tools_policy("sequential", &sequential("m"));
        assert_tools_policy("parallel", &parallel("m", 3));
        assert_tools_policy("avr", &avr("a", "b"));
        assert_tools_policy("selfmoa", &self_moa("m", 3));
        assert_tools_policy("hierarchical", &hierarchical("p", "c"));
        assert_tools_policy("hub", &hub("c", "s", 3));
        assert_tools_policy("debate", &debate("d", "j"));
        assert_tools_policy("brainstorming", &brainstorming("m", 3));
        assert_tools_policy("robust", &robust("m", 3));
        assert_tools_policy("horizon_pipeline", &horizon_pipeline("m", 3));
        assert_tools_policy("parallel_fanout", &parallel_fanout("m", 5));
        assert_tools_policy("formal_solver", &formal_solver("m"));
    }
}
