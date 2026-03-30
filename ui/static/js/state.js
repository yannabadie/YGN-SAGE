/**
 * state.js — YGN-SAGE v2 reactive shared state store
 *
 * Central data hub: WebSocket events flow in, all UI modules subscribe.
 * Zero dependencies — plain ES-module, works in modern browsers.
 *
 * Usage:
 *   import { getState, setState, subscribe, pushEvent, resetState } from './state.js';
 */

'use strict';

// ----------------------------------------------------------------
// Default (blank) state shape
// ----------------------------------------------------------------
function makeDefault() {
    return {
        // Connection
        wsConnected: false,

        // Agent
        isRunning: false,
        currentTaskId: null,

        // Stats
        stepCount: 0,
        llmCalls: 0,
        totalCost: 0.0,
        lastSystem: 0,        // 1 | 2 | 3 (S-type cognitive system)
        lastModel: null,
        lastPhase: null,      // 'perceive' | 'think' | 'act' | 'learn' | null
        lastLatency: 0,
        totalEvents: 0,

        // Memory tiers
        memSTM: 0,
        memEpisodic: 0,
        memSemantic: 0,
        memExoCortex: 0,

        // Guardrails
        z3Pass: 0,
        z3Fail: 0,
        guards: {
            cot:      false,
            sandbox:  false,
            z3:       false,
            avr:      false,
            cgrs:     false,
            episodic: false,
        },

        // Events buffer (newest first, max 500)
        events: [],

        // Chat
        chatMessages: [],     // { role: 'user'|'assistant', content, ts }

        // Topology
        topologyGraph: {
            nodes: [],        // [{ id, label, role, system }]
            edges: [],        // [{ source, target, label }]
        },
        topologySource: null, // 'template' | 'llm' | 'mcts' | 'cma_me' | null
        topologyConfidence: 0.0,

        // Providers
        providers: [],        // [{ name, status, latency_ms, model_count, last_check }]

        // Evolution (MAP-Elites / CMA-ME)
        evolution: {
            generation:      0,
            population_size: 0,
            best_score:      0.0,
            cells:           [],
        },

        // Routing pipeline
        routingPipeline: {
            stage:      null,   // 'knn' | 'onnx' | 'bandit' | 'heuristic' | null
            confidence: 0.0,
            method:     null,
        },

        // UI
        activeTab: 'dashboard',
    };
}

// ----------------------------------------------------------------
// Internal state and subscriber list
// ----------------------------------------------------------------
let _state = makeDefault();
const _subscribers = new Set();
const MAX_EVENTS = 500;

// ----------------------------------------------------------------
// Public API
// ----------------------------------------------------------------

/**
 * Return a shallow copy of the current state.
 * Deep-clone individual fields as needed (e.g. state.events.slice()).
 */
export function getState() {
    return Object.assign({}, _state);
}

/**
 * Merge patch into state and notify all subscribers.
 * @param {Partial<typeof _state>} patch
 * @param {object} [hint]  Optional extra hint forwarded to subscribers.
 */
export function setState(patch, hint = {}) {
    _state = Object.assign({}, _state, patch);
    _notify(hint);
}

/**
 * Register a subscriber callback.
 * The callback receives a notification object { newEvent?, patch? }.
 * Returns an unsubscribe function.
 * @param {(notification: object) => void} cb
 * @returns {() => void}
 */
export function subscribe(cb) {
    _subscribers.add(cb);
    return () => _subscribers.delete(cb);
}

/**
 * Prepend an event to the events buffer (cap at MAX_EVENTS),
 * update derived stats, and notify subscribers.
 * @param {object} evt  Raw WebSocket event object.
 */
export function pushEvent(evt) {
    // 1. Prepend and cap
    const events = [evt, ..._state.events];
    if (events.length > MAX_EVENTS) events.length = MAX_EVENTS;

    // 2. Derive state updates from event fields
    const patch = _deriveFromEvent(evt, events);

    // 3. Apply atomically
    _state = Object.assign({}, _state, patch);

    // 4. Notify with event hint so subscribers can react specifically
    _notify({ newEvent: evt });
}

/**
 * Reset state to defaults (e.g. after /api/reset).
 */
export function resetState() {
    _state = makeDefault();
    _notify({ reset: true });
}

// ----------------------------------------------------------------
// Internal helpers
// ----------------------------------------------------------------

function _notify(hint) {
    for (const cb of _subscribers) {
        try { cb(hint); } catch (e) { console.error('[state] subscriber error:', e); }
    }
}

/**
 * Derive a state patch from a single incoming event.
 * Does NOT mutate _state — returns a plain object.
 */
function _deriveFromEvent(evt, eventsArray) {
    const type  = (evt.type  || '').toUpperCase();
    const meta  = evt.meta   || {};
    const patch = {
        events:      eventsArray,
        totalEvents: _state.totalEvents + 1,
    };

    // ---- Step / cost / model / latency ----
    if (evt.step != null) {
        patch.stepCount = Math.max(_state.stepCount, evt.step);
    }
    if (evt.cost_usd != null) {
        patch.totalCost = Math.max(_state.totalCost, evt.cost_usd);
    }
    if (meta.cost_usd != null) {
        patch.totalCost = Math.max(patch.totalCost ?? _state.totalCost, meta.cost_usd);
    }
    if (evt.system != null && evt.system > 0) {
        patch.lastSystem = evt.system;
    }
    if (evt.model) {
        patch.lastModel = evt.model;
    }
    if (evt.latency_ms != null && evt.latency_ms > 0) {
        patch.lastLatency = evt.latency_ms;
    }

    // ---- Phase tracking ----
    if (['PERCEIVE', 'THINK', 'ACT', 'LEARN'].includes(type)) {
        patch.lastPhase = type.toLowerCase();
    }

    // ---- PERCEIVE — routing info ----
    if (type === 'PERCEIVE') {
        if (meta.system) patch.lastSystem = meta.system;

        const rp = Object.assign({}, _state.routingPipeline);
        if (meta.routing_stage)      rp.stage      = meta.routing_stage;
        if (meta.routing_confidence) rp.confidence = meta.routing_confidence;
        if (meta.routing_method)     rp.method     = meta.routing_method;
        patch.routingPipeline = rp;

        // Guardrail flags from validation_level
        if (meta.validation_level >= 2) {
            patch.guards = Object.assign({}, _state.guards, { sandbox: true });
        }
        if (meta.validation_level >= 3) {
            patch.guards = Object.assign({}, patch.guards ?? _state.guards, { z3: true });
        }
    }

    // ---- THINK — LLM calls, guardrails, Z3 ----
    if (type === 'THINK') {
        if (evt.model) patch.llmCalls = _state.llmCalls + 1;

        const guards = Object.assign({}, _state.guards);
        if (meta.content)              guards.cot = true;
        if (meta.entropy !== undefined) guards.cgrs = (meta.brake === true);
        if (evt.validation === 's2_avr_pass' || evt.validation === 's2_avr_fail') {
            guards.avr     = true;
            guards.sandbox = true;
        }
        if (meta.r_path !== undefined) {
            guards.z3 = true;
            if (meta.r_path >= 0) {
                patch.z3Pass = _state.z3Pass + 1;
            } else {
                patch.z3Fail = _state.z3Fail + 1;
            }
        }
        patch.guards = guards;
    }

    // ---- LEARN — memory tiers ----
    if (type === 'LEARN') {
        if (meta.events !== undefined)   patch.memSTM = meta.events;
        if (meta.memory_tiers) {
            const t = meta.memory_tiers;
            if (t.stm       != null) patch.memSTM       = t.stm;
            if (t.episodic  != null) patch.memEpisodic  = t.episodic;
            if (t.semantic  != null) patch.memSemantic  = t.semantic;
            if (t.exocortex != null) patch.memExoCortex = t.exocortex;
        }
        if ((_state.memEpisodic || patch.memEpisodic || 0) > 0) {
            patch.guards = Object.assign({}, _state.guards, patch.guards, { episodic: true });
        }
    }

    // ---- ACT — tool execution (ExoCortex / Episodic hits) ----
    if (type === 'ACT' && meta.tool) {
        if (meta.tool.includes('exocortex') || meta.tool.includes('search_exocortex')) {
            patch.memExoCortex = (_state.memExoCortex || 0) + 1;
        }
        if (
            meta.tool.includes('episodic')     ||
            meta.tool.includes('recall')       ||
            meta.tool.includes('store_episode')
        ) {
            patch.memEpisodic = (_state.memEpisodic || 0) + 1;
            patch.guards = Object.assign({}, _state.guards, patch.guards, { episodic: true });
        }
    }

    // ---- TOPOLOGY_UPDATE — graph data ----
    if (type === 'TOPOLOGY_UPDATE') {
        if (meta.nodes || meta.edges) {
            patch.topologyGraph = {
                nodes: meta.nodes || _state.topologyGraph.nodes,
                edges: meta.edges || _state.topologyGraph.edges,
            };
        }
        if (meta.source)     patch.topologySource     = meta.source;
        if (meta.confidence) patch.topologyConfidence = meta.confidence;
    }

    return patch;
}
