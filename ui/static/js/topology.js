/**
 * topology.js — YGN-SAGE v2 Topology Graph Module
 *
 * Interactive DAG visualization implementing the 3-flow edge model from
 * MASFactory (arXiv 2603.06007): Control + Message + State flows.
 *
 * Uses Cytoscape.js (global `cytoscape`) and ELK.js (global `ELK`) loaded
 * via CDN. Falls back to grid layout when ELK is unavailable.
 *
 * Exports:
 *   mount(el)  — render toolbar, graph container, detail panel, legend, init Cytoscape
 *   unmount()  — clear poll timer, destroy Cytoscape instance
 */

'use strict';

import { getState, subscribe } from './state.js';

// ----------------------------------------------------------------
// Constants
// ----------------------------------------------------------------

/** px — node size per model tier */
const TIER_SIZES = {
    budget:   40,
    fast:     45,
    balanced: 50,
    reasoner: 55,
    strong:   60,
};

const DEFAULT_SIZE = 50;

/** Cytoscape-compatible hex colors */
const COLORS = {
    // Node fills
    llm:  '#0ea5e9',  // sky-500
    code: '#22c55e',  // green-500

    // Edge stroke colors — 3-flow model
    control: '#6366f1',  // indigo-500
    message: '#38bdf8',  // sky-400
    state:   '#f59e0b',  // amber-500

    // UI / text
    nodeLabel:    '#e2e8f0',   // slate-200
    nodeBorder:   '#475569',   // sage-600
    selectedBorder: '#ffffff',
    bg:           '#020617',   // sage-950
};

/** Source badge colors (bg / text via inline style) */
const SOURCE_COLORS = {
    smmu_hit:         { bg: '#059669', text: '#d1fae5' },  // emerald
    archive_hit:      { bg: '#7c3aed', text: '#ede9fe' },  // purple
    llm_synthesis:    { bg: '#0284c7', text: '#e0f2fe' },  // sky
    mutation:         { bg: '#d97706', text: '#fef3c7' },  // amber
    mcts_search:      { bg: '#0891b2', text: '#cffafe' },  // cyan
    template_fallback:{ bg: '#16a34a', text: '#dcfce7' },  // sage/green (closest)
};

const DEFAULT_SOURCE_COLOR = { bg: '#334155', text: '#94a3b8' };  // sage-700 / sage-400

const POLL_INTERVAL = 5000; // ms

// ----------------------------------------------------------------
// Module-level state
// ----------------------------------------------------------------
let _cy       = null;   // Cytoscape instance
let _el       = null;   // mount root element
let _pollTimer = null;  // setInterval handle
let _unsub    = null;   // state unsubscribe function

// ----------------------------------------------------------------
// Public API
// ----------------------------------------------------------------

/**
 * Mount the topology module into `el`.
 * @param {HTMLElement} el
 */
export function mount(el) {
    _el = el;
    _el.innerHTML = '';

    _renderShell();
    _initCytoscape();
    _fetchAndRender();

    _pollTimer = setInterval(() => _fetchAndRender(), POLL_INTERVAL);

    _unsub = subscribe((hint) => {
        if (hint && hint.reset) {
            if (_cy) _cy.elements().remove();
            return;
        }
        const evt = hint && hint.newEvent;
        if (evt && (evt.type || '').toUpperCase() === 'TOPOLOGY_UPDATE') {
            _fetchAndRender();
        }
    });
}

/**
 * Tear down: clear poll, destroy Cytoscape instance, unsubscribe.
 */
export function unmount() {
    if (_pollTimer)  { clearInterval(_pollTimer); _pollTimer = null; }
    if (_unsub)      { _unsub(); _unsub = null; }
    if (_cy)         { _cy.destroy(); _cy = null; }
    _el = null;
}

// ----------------------------------------------------------------
// Shell rendering (toolbar + graph container + detail panel + legend)
// ----------------------------------------------------------------

function _renderShell() {
    // Wrapper fills the host element
    _el.style.cssText = 'position:relative; display:flex; flex-direction:column; width:100%; height:100%; overflow:hidden; background:#020617; border-radius:8px;';

    // ---- Toolbar ----
    const toolbar = document.createElement('div');
    toolbar.id = 'topo-toolbar';
    toolbar.style.cssText = [
        'display:flex; align-items:center; gap:8px;',
        'padding:8px 12px;',
        'background:#1e293b;',
        'border-bottom:1px solid #334155;',
        'flex-shrink:0;',
        'font-size:12px;',
    ].join('');

    // Source badge
    const sourceBadge = document.createElement('span');
    sourceBadge.id = 'topo-source-badge';
    sourceBadge.style.cssText = 'padding:2px 8px; border-radius:999px; font-weight:600; background:#334155; color:#94a3b8; letter-spacing:0.04em;';
    sourceBadge.textContent = '—';

    // Confidence badge
    const confBadge = document.createElement('span');
    confBadge.id = 'topo-conf-badge';
    confBadge.style.cssText = 'padding:2px 8px; border-radius:999px; background:#1e293b; border:1px solid #334155; color:#94a3b8;';
    confBadge.textContent = 'conf: —';

    // Spacer
    const spacer = document.createElement('span');
    spacer.style.flex = '1';

    // Fit button
    const fitBtn = document.createElement('button');
    fitBtn.textContent = 'Fit';
    fitBtn.style.cssText = 'padding:3px 10px; background:#334155; color:#e2e8f0; border:none; border-radius:4px; cursor:pointer; font-size:11px; font-weight:500;';
    fitBtn.addEventListener('mouseenter', () => { fitBtn.style.background = '#475569'; });
    fitBtn.addEventListener('mouseleave', () => { fitBtn.style.background = '#334155'; });
    fitBtn.addEventListener('click', () => { if (_cy) _cy.fit(undefined, 30); });

    // Refresh button
    const refreshBtn = document.createElement('button');
    refreshBtn.textContent = 'Refresh';
    refreshBtn.style.cssText = 'padding:3px 10px; background:#334155; color:#e2e8f0; border:none; border-radius:4px; cursor:pointer; font-size:11px; font-weight:500;';
    refreshBtn.addEventListener('mouseenter', () => { refreshBtn.style.background = '#475569'; });
    refreshBtn.addEventListener('mouseleave', () => { refreshBtn.style.background = '#334155'; });
    refreshBtn.addEventListener('click', () => _fetchAndRender());

    toolbar.appendChild(sourceBadge);
    toolbar.appendChild(confBadge);
    toolbar.appendChild(spacer);
    toolbar.appendChild(fitBtn);
    toolbar.appendChild(refreshBtn);
    _el.appendChild(toolbar);

    // ---- Graph container ----
    const cyContainer = document.createElement('div');
    cyContainer.id = 'cy-container';
    cyContainer.style.cssText = 'flex:1; width:100%; min-height:0; position:relative; overflow:hidden; background:#020617;';
    _el.appendChild(cyContainer);

    // ---- Legend ----
    const legend = document.createElement('div');
    legend.id = 'topo-legend';
    legend.style.cssText = [
        'display:flex; align-items:center; flex-wrap:wrap; gap:12px;',
        'padding:6px 12px;',
        'background:#1e293b;',
        'border-top:1px solid #334155;',
        'flex-shrink:0;',
        'font-size:11px; color:#94a3b8;',
    ].join('');

    const legendItems = [
        // Edge types — 3-flow model (MASFactory arXiv:2603.06007)
        { label: 'Control',  color: COLORS.control, dash: 'none',   shape: 'line' },
        { label: 'Message',  color: COLORS.message, dash: 'dashed', shape: 'line' },
        { label: 'State',    color: COLORS.state,   dash: 'dotted', shape: 'line' },
        // Node types
        { label: 'LLM',   color: COLORS.llm,  dash: 'none', shape: 'circle' },
        { label: 'Code',  color: COLORS.code, dash: 'none', shape: 'square' },
    ];

    for (const item of legendItems) {
        const wrap = document.createElement('div');
        wrap.style.cssText = 'display:flex; align-items:center; gap:5px;';

        if (item.shape === 'line') {
            // Inline SVG for edge legend
            const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            svg.setAttribute('width', '28');
            svg.setAttribute('height', '10');
            svg.style.display = 'block';
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', '2');
            line.setAttribute('y1', '5');
            line.setAttribute('x2', '26');
            line.setAttribute('y2', '5');
            line.setAttribute('stroke', item.color);
            line.setAttribute('stroke-width', '2');
            if (item.dash === 'dashed') {
                line.setAttribute('stroke-dasharray', '5,3');
            } else if (item.dash === 'dotted') {
                line.setAttribute('stroke-dasharray', '2,3');
            }
            svg.appendChild(line);
            wrap.appendChild(svg);
        } else if (item.shape === 'circle') {
            const dot = document.createElement('span');
            dot.style.cssText = `width:12px; height:12px; border-radius:50%; background:${item.color}; flex-shrink:0; display:inline-block;`;
            wrap.appendChild(dot);
        } else if (item.shape === 'square') {
            const sq = document.createElement('span');
            sq.style.cssText = `width:12px; height:12px; border-radius:2px; background:${item.color}; flex-shrink:0; display:inline-block;`;
            wrap.appendChild(sq);
        }

        const lbl = document.createElement('span');
        lbl.textContent = item.label;
        wrap.appendChild(lbl);
        legend.appendChild(wrap);
    }

    _el.appendChild(legend);

    // ---- Node detail panel (hidden by default, absolute) ----
    const detailPanel = document.createElement('div');
    detailPanel.id = 'topo-detail';
    detailPanel.style.cssText = [
        'display:none;',
        'position:absolute;',
        'top:52px; right:8px;',
        'width:220px;',
        'background:#1e293b;',
        'border:1px solid #475569;',
        'border-radius:8px;',
        'padding:12px;',
        'z-index:20;',
        'font-size:12px;',
        'color:#e2e8f0;',
        'box-shadow:0 4px 20px rgba(0,0,0,0.5);',
    ].join('');
    _el.appendChild(detailPanel);
}

// ----------------------------------------------------------------
// Cytoscape initialization
// ----------------------------------------------------------------

function _initCytoscape() {
    if (typeof cytoscape === 'undefined') {
        console.warn('[topology] Cytoscape.js not loaded — graph unavailable');
        return;
    }

    _cy = cytoscape({
        container: document.getElementById('cy-container'),

        elements: [],

        style: [
            // ---- Node base ----
            {
                selector: 'node',
                style: {
                    'label':            'data(role)',
                    'font-size':        '10px',
                    'font-family':      'JetBrains Mono, Consolas, monospace',
                    'text-valign':      'center',
                    'text-halign':      'center',
                    'color':            COLORS.nodeLabel,
                    'text-outline-color': '#020617',
                    'text-outline-width': '2px',
                    'border-width':     '2px',
                    'border-color':     COLORS.nodeBorder,
                    'width':            'data(size)',
                    'height':           'data(size)',
                    'background-color': '#334155',
                },
            },

            // ---- LLM node — ellipse, sky-blue ----
            {
                selector: 'node[node_type = "llm"]',
                style: {
                    'shape':            'ellipse',
                    'background-color': COLORS.llm,
                    'border-color':     '#0369a1',  // sky-700
                },
            },

            // ---- Code node — round-rectangle, green ----
            {
                selector: 'node[node_type = "code"]',
                style: {
                    'shape':            'round-rectangle',
                    'background-color': COLORS.code,
                    'border-color':     '#15803d',  // green-700
                },
            },

            // ---- Selected node — white border ----
            {
                selector: 'node:selected',
                style: {
                    'border-width':  '4px',
                    'border-color':  COLORS.selectedBorder,
                },
            },

            // ---- Edge base ----
            {
                selector: 'edge',
                style: {
                    'width':              '2px',
                    'line-color':         '#475569',
                    'target-arrow-color': '#475569',
                    'target-arrow-shape': 'triangle',
                    'curve-style':        'bezier',
                    'opacity':            0.8,
                },
            },

            // ---- Control edges — solid indigo ----
            {
                selector: 'edge[flow_type = "control"]',
                style: {
                    'line-color':         COLORS.control,
                    'target-arrow-color': COLORS.control,
                    'line-style':         'solid',
                },
            },

            // ---- Message edges — dashed sky ----
            {
                selector: 'edge[flow_type = "message"]',
                style: {
                    'line-color':         COLORS.message,
                    'target-arrow-color': COLORS.message,
                    'line-style':         'dashed',
                    'line-dash-pattern':  [6, 4],
                },
            },

            // ---- State edges — dotted amber ----
            {
                selector: 'edge[flow_type = "state"]',
                style: {
                    'line-color':         COLORS.state,
                    'target-arrow-color': COLORS.state,
                    'line-style':         'dashed',
                    'line-dash-pattern':  [2, 4],
                },
            },
        ],

        minZoom: 0.3,
        maxZoom: 3,
        wheelSensitivity: 0.3,
        userZoomingEnabled: true,
        userPanningEnabled: true,
        boxSelectionEnabled: false,
        autoungrabify: false,
    });

    // ---- Node tap: show detail panel ----
    _cy.on('tap', 'node', (evt) => {
        const node   = evt.target;
        const data   = node.data();
        _showDetail(data, node.renderedPosition());
    });

    // ---- Background tap: hide detail panel ----
    _cy.on('tap', (evt) => {
        if (evt.target === _cy) {
            _hideDetail();
        }
    });
}

// ----------------------------------------------------------------
// Detail panel
// ----------------------------------------------------------------

function _showDetail(data, renderedPos) {
    const panel = document.getElementById('topo-detail');
    if (!panel) return;

    const tier   = data.model_tier || '—';
    const type   = data.node_type  || '—';
    const role   = data.role       || data.id || '—';
    const prompt = data.prompt     ? data.prompt.slice(0, 160) + (data.prompt.length > 160 ? '…' : '') : '—';

    panel.innerHTML = `
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
            <span style="font-weight:600;color:#f8fafc;font-size:13px;">${_esc(role)}</span>
            <button id="topo-detail-close" style="background:none;border:none;color:#64748b;cursor:pointer;font-size:16px;line-height:1;padding:0;">&times;</button>
        </div>
        <div style="display:flex;flex-direction:column;gap:6px;">
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#64748b;">Type</span>
                <span style="color:#38bdf8;font-family:monospace;">${_esc(type)}</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#64748b;">Tier</span>
                <span style="color:#a78bfa;font-family:monospace;">${_esc(tier)}</span>
            </div>
            <div style="border-top:1px solid #334155;padding-top:6px;margin-top:2px;">
                <div style="color:#64748b;margin-bottom:4px;">Prompt</div>
                <div style="color:#cbd5e1;font-family:monospace;font-size:11px;word-break:break-word;white-space:pre-wrap;max-height:80px;overflow-y:auto;">${_esc(prompt)}</div>
            </div>
        </div>
    `;

    document.getElementById('topo-detail-close').addEventListener('click', _hideDetail);
    panel.style.display = 'block';
}

function _hideDetail() {
    const panel = document.getElementById('topo-detail');
    if (panel) panel.style.display = 'none';
}

// ----------------------------------------------------------------
// Fetch and render
// ----------------------------------------------------------------

async function _fetchAndRender() {
    try {
        const resp = await fetch('/api/topology/graph');
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        // Update badges
        _updateSourceBadge(data.source);
        _updateConfBadge(data.confidence);

        if (!_cy) return;

        const nodes = data.nodes || [];
        const edges = data.edges || [];

        const positions = await _computeLayout(nodes, edges);
        _renderGraph(nodes, edges, positions);

    } catch (err) {
        // Silently ignore — polling will retry
        console.debug('[topology] fetch error:', err.message);
    }
}

function _updateSourceBadge(source) {
    const badge = document.getElementById('topo-source-badge');
    if (!badge) return;
    const c = SOURCE_COLORS[source] || DEFAULT_SOURCE_COLOR;
    badge.style.background = c.bg;
    badge.style.color       = c.text;
    badge.textContent       = source ? source.replace(/_/g, ' ') : '—';
}

function _updateConfBadge(confidence) {
    const badge = document.getElementById('topo-conf-badge');
    if (!badge) return;
    const pct   = confidence != null ? (confidence * 100).toFixed(0) : '—';
    badge.textContent = `conf: ${pct}%`;
}

// ----------------------------------------------------------------
// Layout computation (ELK layered or grid fallback)
// ----------------------------------------------------------------

async function _computeLayout(nodes, edges) {
    if (typeof ELK === 'undefined') {
        return _gridFallback(nodes);
    }

    try {
        const elk = new ELK();

        const elkNodes = nodes.map((n) => {
            const sz = TIER_SIZES[n.model_tier] || DEFAULT_SIZE;
            return { id: n.id, width: sz, height: sz };
        });

        const elkEdges = edges.map((e) => ({
            id:      e.id || `${e.source}-${e.target}`,
            sources: [e.source],
            targets: [e.target],
        }));

        const graph = {
            id: 'root',
            layoutOptions: {
                'elk.algorithm':             'layered',
                'elk.direction':             'DOWN',
                'elk.spacing.nodeNode':      '50',
                'elk.layered.spacing.nodeNodeBetweenLayers': '60',
            },
            children: elkNodes,
            edges:    elkEdges,
        };

        const result  = await elk.layout(graph);
        const posDict = {};

        for (const child of result.children || []) {
            posDict[child.id] = {
                x: child.x + child.width  / 2,
                y: child.y + child.height / 2,
            };
        }

        return posDict;

    } catch (err) {
        console.debug('[topology] ELK layout error, falling back to grid:', err.message);
        return _gridFallback(nodes);
    }
}

function _gridFallback(nodes) {
    const cols = Math.max(1, Math.ceil(Math.sqrt(nodes.length)));
    const STEP = 120;
    const positions = {};
    nodes.forEach((n, i) => {
        positions[n.id] = {
            x: (i % cols)           * STEP + 60,
            y: Math.floor(i / cols) * STEP + 60,
        };
    });
    return positions;
}

// ----------------------------------------------------------------
// Graph rendering
// ----------------------------------------------------------------

function _renderGraph(nodes, edges, positions) {
    if (!_cy) return;

    _cy.elements().remove();

    const cyNodes = nodes.map((n) => {
        const sz  = TIER_SIZES[n.model_tier] || DEFAULT_SIZE;
        const pos = positions[n.id] || { x: 0, y: 0 };
        return {
            group: 'nodes',
            data: {
                id:         n.id,
                role:       n.role       || n.id,
                node_type:  n.node_type  || 'llm',
                model_tier: n.model_tier || 'balanced',
                prompt:     n.prompt     || '',
                size:       sz,
            },
            position: { x: pos.x, y: pos.y },
        };
    });

    const cyEdges = edges.map((e) => ({
        group: 'edges',
        data: {
            id:        e.id || `${e.source}->${e.target}`,
            source:    e.source,
            target:    e.target,
            flow_type: e.flow_type || 'control',
        },
    }));

    _cy.add([...cyNodes, ...cyEdges]);
    _cy.fit(undefined, 30);
}

// ----------------------------------------------------------------
// Utility
// ----------------------------------------------------------------

function _esc(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}
