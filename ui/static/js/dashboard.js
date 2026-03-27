/**
 * dashboard.js — YGN-SAGE v2 Control Panel View
 *
 * Refactored from monolithic index.html.
 * Exports mount(el) and unmount().
 *
 * Dependencies:
 *   - state.js  (getState, setState, subscribe, resetState)
 *   - Chart.js  (global Chart)
 */

'use strict';

import { getState, setState, subscribe, resetState } from './state.js';

// ---------------------------------------------------------------
// Module-level references (populated by mount, cleared by unmount)
// ---------------------------------------------------------------
let _root        = null;   // mounted container element
let _chart       = null;   // Chart.js instance
let _unsub       = null;   // unsubscribe from state
let _latency     = [];     // 30-point latency history
let _lastResponse = '';    // last response text for the response pane

// ---------------------------------------------------------------
// HTML template
// ---------------------------------------------------------------
function _buildHTML() {
    return `
<div class="grid grid-cols-[1fr_300px] gap-3 h-full min-h-0" id="dash-grid">

    <!-- ================================================ -->
    <!-- LEFT COLUMN                                       -->
    <!-- ================================================ -->
    <div class="flex flex-col gap-3 min-h-0">

        <!-- Task Input -->
        <div class="bg-sage-800 border border-sage-700 rounded-lg p-4 flex-shrink-0">
            <label class="text-xs font-medium text-sage-400 uppercase tracking-wider mb-2 block">Task Input</label>
            <textarea
                id="dash-task-input"
                class="w-full bg-sage-900 border border-sage-700 rounded-md px-3 py-2 text-sm font-mono text-white placeholder-sage-600 resize-none focus:outline-none focus:border-sky-500 transition-colors"
                rows="3"
                placeholder="Enter a task for YGN-SAGE… (Ctrl+Enter to run)"
            ></textarea>
            <div class="flex items-center gap-2 mt-2">
                <button id="dash-btn-run"
                    class="px-4 py-1.5 bg-sky-600 hover:bg-sky-500 text-white text-sm font-medium rounded-md transition-colors disabled:opacity-40 disabled:cursor-not-allowed">
                    Run
                </button>
                <button id="dash-btn-stop"
                    class="px-4 py-1.5 bg-sage-700 hover:bg-sage-600 text-sage-300 text-sm font-medium rounded-md transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                    disabled>
                    Stop
                </button>
                <button id="dash-btn-reset"
                    class="px-4 py-1.5 bg-sage-700 hover:bg-sage-600 text-sage-300 text-sm font-medium rounded-md transition-colors">
                    Reset
                </button>
                <div id="dash-status-badge" class="ml-auto text-xs px-2 py-0.5 rounded-full bg-sage-700 text-sage-400">
                    idle
                </div>
            </div>
        </div>

        <!-- Response Pane -->
        <div class="bg-sage-800 border border-sage-700 rounded-lg p-4 flex-1 min-h-0 flex flex-col overflow-hidden">
            <div class="flex items-center justify-between mb-2 flex-shrink-0">
                <label class="text-xs font-medium text-sage-400 uppercase tracking-wider">Response</label>
                <span id="dash-response-system" class="text-xs px-2 py-0.5 rounded-full bg-sage-700 text-sage-400">--</span>
            </div>
            <div id="dash-response-pane"
                class="flex-1 overflow-y-auto bg-sage-900 border border-sage-700 rounded-md p-3 text-sm font-mono leading-relaxed whitespace-pre-wrap text-sage-400">
                Waiting for task…
            </div>
        </div>
    </div>

    <!-- ================================================ -->
    <!-- RIGHT COLUMN (300px)                             -->
    <!-- ================================================ -->
    <div class="flex flex-col gap-3 min-h-0 overflow-y-auto">

        <!-- Memory Tiers -->
        <div class="bg-sage-800 border border-sage-700 rounded-lg p-4 flex-shrink-0">
            <label class="text-xs font-medium text-sage-400 uppercase tracking-wider mb-3 block">Memory Tiers</label>
            <div class="space-y-2.5">
                <div>
                    <div class="flex justify-between text-xs mb-1">
                        <span class="text-sage-400">STM / Working</span>
                        <span id="dash-mem-stm-val" class="font-mono text-white">0</span>
                    </div>
                    <div class="h-2 bg-sage-900 rounded-full overflow-hidden">
                        <div id="dash-mem-stm-bar" class="h-full bg-sky-500 rounded-full transition-all duration-500" style="width:0%"></div>
                    </div>
                </div>
                <div>
                    <div class="flex justify-between text-xs mb-1">
                        <span class="text-sage-400">Episodic</span>
                        <span id="dash-mem-ep-val" class="font-mono text-white">0</span>
                    </div>
                    <div class="h-2 bg-sage-900 rounded-full overflow-hidden">
                        <div id="dash-mem-ep-bar" class="h-full bg-purple-500 rounded-full transition-all duration-500" style="width:0%"></div>
                    </div>
                </div>
                <div>
                    <div class="flex justify-between text-xs mb-1">
                        <span class="text-sage-400">Semantic</span>
                        <span id="dash-mem-sem-val" class="font-mono text-white">0</span>
                    </div>
                    <div class="h-2 bg-sage-900 rounded-full overflow-hidden">
                        <div id="dash-mem-sem-bar" class="h-full bg-amber-500 rounded-full transition-all duration-500" style="width:0%"></div>
                    </div>
                </div>
                <div>
                    <div class="flex justify-between text-xs mb-1">
                        <span class="text-sage-400">ExoCortex</span>
                        <span id="dash-mem-exo-val" class="font-mono text-white">0</span>
                    </div>
                    <div class="h-2 bg-sage-900 rounded-full overflow-hidden">
                        <div id="dash-mem-exo-bar" class="h-full bg-emerald-500 rounded-full transition-all duration-500" style="width:0%"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Guardrails -->
        <div class="bg-sage-800 border border-sage-700 rounded-lg p-4 flex-shrink-0">
            <label class="text-xs font-medium text-sage-400 uppercase tracking-wider mb-3 block">Guardrails</label>
            <div class="space-y-1.5 text-xs">
                <div id="dash-guard-cot" class="flex items-center gap-2 text-sage-500">
                    <span class="w-4 text-center font-mono">--</span>
                    <span>Chain-of-Thought</span>
                </div>
                <div id="dash-guard-sandbox" class="flex items-center gap-2 text-sage-500">
                    <span class="w-4 text-center font-mono">--</span>
                    <span>Sandbox Execution</span>
                </div>
                <div id="dash-guard-z3" class="flex items-center gap-2 text-sage-500">
                    <span class="w-4 text-center font-mono">--</span>
                    <span>Z3 Formal Proof</span>
                </div>
                <div id="dash-guard-avr" class="flex items-center gap-2 text-sage-500">
                    <span class="w-4 text-center font-mono">--</span>
                    <span>AVR Loop</span>
                </div>
                <div id="dash-guard-cgrs" class="flex items-center gap-2 text-sage-500">
                    <span class="w-4 text-center font-mono">--</span>
                    <span>CGRS Self-Brake</span>
                </div>
                <div id="dash-guard-episodic" class="flex items-center gap-2 text-sage-500">
                    <span class="w-4 text-center font-mono">--</span>
                    <span>Episodic Memory</span>
                </div>
            </div>
        </div>

        <!-- Routing Pipeline -->
        <div class="bg-sage-800 border border-sage-700 rounded-lg p-4 flex-shrink-0">
            <label class="text-xs font-medium text-sage-400 uppercase tracking-wider mb-3 block">Routing Pipeline</label>
            <div class="space-y-1.5" id="dash-routing-stages">
                <!-- stages injected by _renderRoutingPipeline -->
            </div>
        </div>

        <!-- Statistics -->
        <div class="bg-sage-800 border border-sage-700 rounded-lg p-4 flex-shrink-0">
            <label class="text-xs font-medium text-sage-400 uppercase tracking-wider mb-3 block">Statistics</label>
            <div class="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
                <div class="text-sage-500">Steps</div>
                <div id="dash-stat-steps" class="text-right font-mono">0</div>
                <div class="text-sage-500">LLM Calls</div>
                <div id="dash-stat-llm" class="text-right font-mono">0</div>
                <div class="text-sage-500">Cost (USD)</div>
                <div id="dash-stat-cost" class="text-right font-mono">$0.0000</div>
                <div class="text-sage-500">Latency</div>
                <div id="dash-stat-latency" class="text-right font-mono">--</div>
                <div class="text-sage-500">Z3 Pass / Fail</div>
                <div id="dash-stat-z3" class="text-right font-mono">0 / 0</div>
                <div class="text-sage-500">Total Events</div>
                <div id="dash-stat-events" class="text-right font-mono">0</div>
            </div>
        </div>

        <!-- Latency Chart -->
        <div class="bg-sage-800 border border-sage-700 rounded-lg p-4 flex-shrink-0">
            <label class="text-xs font-medium text-sage-400 uppercase tracking-wider mb-2 block">Latency (ms)</label>
            <div style="height:120px; position:relative;">
                <canvas id="dash-latency-chart"></canvas>
            </div>
        </div>

    </div><!-- end right column -->
</div><!-- end dash-grid -->
`;
}

// ---------------------------------------------------------------
// Routing pipeline stage descriptors (order = stage index)
// ---------------------------------------------------------------
const ROUTING_STAGES = [
    { key: 'structural', label: 'Structural',   subtitle: 'Hard constraints' },
    { key: 'knn',        label: 'kNN 92%',      subtitle: 'Embedding similarity' },
    { key: 'onnx',       label: 'ONNX BERT',    subtitle: 'Intent classifier' },
    { key: 'entropy',    label: 'Entropy Probe', subtitle: 'Uncertainty gate' },
];

// Map state.routingPipeline.stage values → stage index
const STAGE_INDEX = {
    structural: 0,
    knn:        1,
    onnx:       2,
    entropy:    3,
    bandit:     3,  // bandit fires at entropy-probe level
    heuristic:  0,  // fallback treated as structural
};

// ---------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------
function _q(id) {
    return _root ? _root.querySelector('#' + id) : null;
}

function _setBar(prefix, value, max) {
    const pct = Math.min(100, (value / Math.max(1, max)) * 100);
    const val = _q(`dash-mem-${prefix}-val`);
    const bar = _q(`dash-mem-${prefix}-bar`);
    if (val) val.textContent = value;
    if (bar) bar.style.width = pct + '%';
}

function _setGuard(key, active) {
    const el = _q(`dash-guard-${key}`);
    if (!el) return;
    const icon = el.querySelector('span:first-child');
    if (active) {
        if (icon) icon.textContent = '\u2713';
        el.className = 'flex items-center gap-2 text-emerald-400';
    } else {
        if (icon) icon.textContent = '--';
        el.className = 'flex items-center gap-2 text-sage-500';
    }
}

// ---------------------------------------------------------------
// Chart initialisation
// ---------------------------------------------------------------
function _initChart() {
    const canvas = _q('dash-latency-chart');
    if (!canvas || typeof Chart === 'undefined') return;
    _chart = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                data: [],
                borderColor: '#38bdf8',
                backgroundColor: 'rgba(56, 189, 248, 0.10)',
                borderWidth: 1.5,
                pointRadius: 2,
                pointBackgroundColor: '#38bdf8',
                fill: true,
                tension: 0.3,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: {
                    beginAtZero: true,
                    ticks: { color: '#64748b', font: { size: 10 } },
                    grid:  { color: '#1e293b' },
                },
            },
            animation: { duration: 300 },
        },
    });
}

function _pushLatency(ms) {
    if (!_chart || ms <= 0) return;
    _latency.push(ms);
    if (_latency.length > 30) _latency.shift();
    _chart.data.labels   = _latency.map((_, i) => i + 1);
    _chart.data.datasets[0].data = _latency;
    _chart.update('none');
}

function _clearChart() {
    if (!_chart) return;
    _latency = [];
    _chart.data.labels = [];
    _chart.data.datasets[0].data = [];
    _chart.update('none');
}

// ---------------------------------------------------------------
// Rendering: full state refresh
// ---------------------------------------------------------------
function _render(hint) {
    if (!_root) return;
    const s = getState();

    // --- Status badge ---
    const badge = _q('dash-status-badge');
    if (badge) {
        if (s.isRunning) {
            badge.textContent = s.lastPhase || 'running';
            badge.className = 'ml-auto text-xs px-2 py-0.5 rounded-full bg-sky-500/20 text-sky-400 phase-active';
        } else {
            badge.textContent = 'idle';
            badge.className = 'ml-auto text-xs px-2 py-0.5 rounded-full bg-sage-700 text-sage-400';
        }
    }

    // --- Button states ---
    const btnRun  = _q('dash-btn-run');
    const btnStop = _q('dash-btn-stop');
    if (btnRun)  btnRun.disabled  = s.isRunning;
    if (btnStop) btnStop.disabled = !s.isRunning;

    // --- Response pane ---
    const responsePane = _q('dash-response-pane');
    if (responsePane) {
        // Update text only when there is new content (avoid DOM thrash on every event)
        const latestContent = _extractLastResponse(s);
        if (latestContent && latestContent !== _lastResponse) {
            _lastResponse = latestContent;
            responsePane.textContent = latestContent;
            responsePane.className = 'flex-1 overflow-y-auto bg-sage-900 border border-sage-700 rounded-md p-3 text-sm font-mono leading-relaxed whitespace-pre-wrap text-white';
        }
    }

    // --- System label ---
    const sysLabel = _q('dash-response-system');
    if (sysLabel) {
        if (s.lastSystem === 1) {
            sysLabel.textContent = 'S1 Fast';
            sysLabel.className = 'text-xs px-2 py-0.5 rounded-full bg-green-500/20 text-green-400';
        } else if (s.lastSystem === 2) {
            sysLabel.textContent = 'S2 Empirical';
            sysLabel.className = 'text-xs px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-400';
        } else if (s.lastSystem === 3) {
            sysLabel.textContent = 'S3 Formal';
            sysLabel.className = 'text-xs px-2 py-0.5 rounded-full bg-red-500/20 text-red-400';
        } else {
            sysLabel.textContent = '--';
            sysLabel.className = 'text-xs px-2 py-0.5 rounded-full bg-sage-700 text-sage-400';
        }
    }

    // --- Memory bars ---
    _setBar('stm', s.memSTM,       100);
    _setBar('ep',  s.memEpisodic,  50);
    _setBar('sem', s.memSemantic,  50);
    _setBar('exo', s.memExoCortex, 50);

    // --- Guardrails ---
    const g = s.guards || {};
    _setGuard('cot',      g.cot);
    _setGuard('sandbox',  g.sandbox);
    _setGuard('z3',       g.z3);
    _setGuard('avr',      g.avr);
    _setGuard('cgrs',     g.cgrs);
    _setGuard('episodic', g.episodic);

    // --- Routing pipeline ---
    _renderRoutingPipeline(s.routingPipeline);

    // --- Statistics ---
    const setText = (id, v) => { const el = _q(id); if (el) el.textContent = v; };
    setText('dash-stat-steps',   s.stepCount);
    setText('dash-stat-llm',     s.llmCalls);
    setText('dash-stat-cost',    '$' + (s.totalCost || 0).toFixed(4));
    setText('dash-stat-latency', s.lastLatency > 0 ? s.lastLatency.toFixed(0) + 'ms' : '--');
    setText('dash-stat-z3',      `${s.z3Pass} / ${s.z3Fail}`);
    setText('dash-stat-events',  s.totalEvents);

    // --- Latency chart: push on newEvent with latency ---
    if (hint && hint.newEvent) {
        const ms = hint.newEvent.latency_ms;
        if (ms && ms > 0) _pushLatency(ms);
    }

    // --- Clear chart on reset ---
    if (hint && hint.reset) {
        _clearChart();
        _lastResponse = '';
        if (responsePane) {
            responsePane.textContent = 'Waiting for task\u2026';
            responsePane.className = 'flex-1 overflow-y-auto bg-sage-900 border border-sage-700 rounded-md p-3 text-sm font-mono leading-relaxed whitespace-pre-wrap text-sage-400';
        }
    }
}

/**
 * Extract the most recent response text from the event buffer.
 * We look at the THINK events that contain meta.content.
 */
function _extractLastResponse(s) {
    const events = s.events || [];
    for (const evt of events) {
        const type = (evt.type || '').toUpperCase();
        if (type === 'THINK' && evt.meta && evt.meta.content) {
            return evt.meta.content;
        }
        if (type === 'LEARN' && evt.meta && evt.meta.response_text) {
            return evt.meta.response_text;
        }
    }
    return null;
}

function _renderRoutingPipeline(rp) {
    const container = _q('dash-routing-stages');
    if (!container) return;

    const activeStage = rp ? (STAGE_INDEX[rp.stage] ?? -1) : -1;

    container.innerHTML = ROUTING_STAGES.map((stage, idx) => {
        const isActive = idx === activeStage;
        const isDone   = idx < activeStage;

        let dotColor  = 'bg-sage-600';
        let textColor = 'text-sage-500';
        let borderClass = 'border-sage-700';
        let badge = '';

        if (isActive) {
            dotColor    = 'bg-sky-400';
            textColor   = 'text-sky-300';
            borderClass = 'border-sky-500/50';
            if (rp && rp.confidence > 0) {
                badge = `<span class="ml-auto text-sky-400 font-mono text-xs">${(rp.confidence * 100).toFixed(0)}%</span>`;
            }
        } else if (isDone) {
            dotColor  = 'bg-emerald-500';
            textColor = 'text-sage-400';
        }

        return `
<div class="flex items-center gap-2 px-2 py-1.5 rounded border ${borderClass} ${isActive ? 'bg-sky-500/5' : ''}">
    <span class="text-sage-600 font-mono text-xs w-4 text-right flex-shrink-0">${idx}</span>
    <span class="w-2 h-2 rounded-full flex-shrink-0 ${dotColor}"></span>
    <span class="${textColor} text-xs font-medium flex-1">${stage.label}</span>
    ${badge}
</div>`;
    }).join('');
}

// ---------------------------------------------------------------
// API actions
// ---------------------------------------------------------------
async function _runTask() {
    const input = _q('dash-task-input');
    if (!input) return;
    const task = input.value.trim();
    if (!task) return;

    try {
        const resp = await fetch('/api/task', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ task }),
        });
        const data = await resp.json();
        if (resp.ok) {
            setState({ isRunning: true });
        } else {
            alert(data.error || 'Failed to submit task');
        }
    } catch (e) {
        alert('Network error: ' + e.message);
    }
}

async function _stopTask() {
    try {
        await fetch('/api/stop', { method: 'POST' });
        setState({ isRunning: false });
    } catch (e) {
        console.error('[dashboard] Stop failed:', e);
    }
}

async function _resetAll() {
    try {
        await fetch('/api/reset', { method: 'POST' });
        resetState();   // notifies all subscribers, including _render with hint.reset=true
    } catch (e) {
        console.error('[dashboard] Reset failed:', e);
    }
}

// ---------------------------------------------------------------
// Public API
// ---------------------------------------------------------------

/**
 * Mount the dashboard into `el`.
 * @param {HTMLElement} el
 */
export function mount(el) {
    if (_root) unmount();

    _root = el;
    _root.innerHTML = _buildHTML();

    // Wire buttons
    const btnRun   = _q('dash-btn-run');
    const btnStop  = _q('dash-btn-stop');
    const btnReset = _q('dash-btn-reset');
    const textarea = _q('dash-task-input');

    if (btnRun)   btnRun.addEventListener('click',  _runTask);
    if (btnStop)  btnStop.addEventListener('click', _stopTask);
    if (btnReset) btnReset.addEventListener('click', _resetAll);

    // Ctrl+Enter shortcut
    if (textarea) {
        textarea.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                _runTask();
            }
        });
    }

    // Init chart
    _initChart();

    // Subscribe to state changes
    _unsub = subscribe(_render);

    // Initial render from current state
    _render({});
}

/**
 * Unmount the dashboard: destroy chart, clear DOM, unsubscribe.
 */
export function unmount() {
    if (_unsub) { _unsub(); _unsub = null; }
    if (_chart)  { _chart.destroy(); _chart = null; }
    if (_root)   { _root.innerHTML = ''; _root = null; }
    _latency      = [];
    _lastResponse = '';
}
