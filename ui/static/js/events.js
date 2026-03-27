/**
 * events.js — Filterable real-time event stream table
 *
 * Subscribes to state.js and renders a live event table with per-type
 * color coding, filter dropdown, and event count badge.
 *
 * Exports: mount(container), unmount()
 */

'use strict';

import { getState, subscribe } from './state.js';

// ----------------------------------------------------------------
// Constants
// ----------------------------------------------------------------

const MAX_ROWS = 200;

const EVENT_TYPES = ['ALL', 'PERCEIVE', 'THINK', 'ACT', 'LEARN', 'ERROR', 'TOPOLOGY_UPDATE'];

/** Tailwind color classes per event type (bg for dot, text for badge) */
const TYPE_STYLES = {
    PERCEIVE:        { dot: 'bg-purple-400',  badge: 'text-purple-300  bg-purple-900/40  border-purple-700',  row: 'hover:bg-purple-900/10'  },
    THINK:           { dot: 'bg-sky-400',      badge: 'text-sky-300     bg-sky-900/40     border-sky-700',      row: 'hover:bg-sky-900/10'      },
    ACT:             { dot: 'bg-amber-400',    badge: 'text-amber-300   bg-amber-900/40   border-amber-700',    row: 'hover:bg-amber-900/10'    },
    LEARN:           { dot: 'bg-emerald-400',  badge: 'text-emerald-300 bg-emerald-900/40 border-emerald-700', row: 'hover:bg-emerald-900/10'  },
    ERROR:           { dot: 'bg-red-400',      badge: 'text-red-300     bg-red-900/40     border-red-700',      row: 'hover:bg-red-900/10'      },
    TOPOLOGY_UPDATE: { dot: 'bg-cyan-400',     badge: 'text-cyan-300    bg-cyan-900/40    border-cyan-700',     row: 'hover:bg-cyan-900/10'     },
};

const DEFAULT_STYLE = {
    dot:   'bg-sage-500',
    badge: 'text-sage-300 bg-sage-800 border-sage-600',
    row:   'hover:bg-sage-700/20',
};

// ----------------------------------------------------------------
// Module-level state
// ----------------------------------------------------------------

let _container   = null;
let _unsub       = null;
let _filter      = 'ALL';

// DOM refs (set in mount, cleared in unmount)
let _tbody       = null;
let _countBadge  = null;
let _filterSel   = null;

// ----------------------------------------------------------------
// Public API
// ----------------------------------------------------------------

/**
 * Mount the event stream table into the given DOM container.
 * @param {HTMLElement} container
 */
export function mount(container) {
    _container = container;
    _filter    = 'ALL';

    container.innerHTML = _buildShell();

    // Grab live DOM refs
    _tbody      = container.querySelector('#evt-tbody');
    _countBadge = container.querySelector('#evt-count');
    _filterSel  = container.querySelector('#evt-filter');

    // Wire filter dropdown
    _filterSel.addEventListener('change', () => {
        _filter = _filterSel.value;
        _rebuildTable();
    });

    // Populate from existing state
    _rebuildTable();

    // Subscribe to future updates
    _unsub = subscribe((patch) => {
        if (patch && patch.reset) {
            _rebuildTable();
        } else if (patch && patch.newEvent) {
            _addRow(patch.newEvent);
            _updateCount();
        }
    });
}

/**
 * Unmount and clean up subscriptions + DOM refs.
 */
export function unmount() {
    if (_unsub) { _unsub(); _unsub = null; }
    _tbody      = null;
    _countBadge = null;
    _filterSel  = null;
    _container  = null;
}

// ----------------------------------------------------------------
// Private: DOM construction
// ----------------------------------------------------------------

/**
 * Build the outer shell HTML (filter bar + table skeleton).
 * @returns {string}
 */
function _buildShell() {
    const options = EVENT_TYPES.map(t =>
        `<option value="${t}">${t}</option>`
    ).join('');

    return `
<div class="flex flex-col h-full gap-0">
    <!-- Toolbar -->
    <div class="flex items-center justify-between px-4 py-2.5 border-b border-sage-700 flex-shrink-0">
        <div class="flex items-center gap-3">
            <span class="text-xs font-semibold text-sage-400 uppercase tracking-wider">Event Stream</span>
            <span id="evt-count"
                  class="text-xs font-mono px-1.5 py-0.5 rounded-full bg-sage-700 text-sage-300 min-w-[1.75rem] text-center">
                0
            </span>
        </div>
        <div class="flex items-center gap-2">
            <label for="evt-filter" class="text-xs text-sage-500">Filter:</label>
            <select id="evt-filter"
                    class="text-xs bg-sage-900 border border-sage-700 text-sage-300 rounded px-2 py-1
                           focus:outline-none focus:border-sky-500 transition-colors cursor-pointer">
                ${options}
            </select>
        </div>
    </div>

    <!-- Table -->
    <div class="flex-1 overflow-y-auto min-h-0">
        <table class="w-full text-xs border-collapse">
            <thead class="sticky top-0 bg-sage-800 z-10">
                <tr class="text-sage-500 uppercase tracking-wider">
                    <th class="text-left px-3 py-2 font-medium w-[72px]">Time</th>
                    <th class="text-left px-3 py-2 font-medium w-[44px]">Step</th>
                    <th class="text-left px-3 py-2 font-medium w-[130px]">Type</th>
                    <th class="text-left px-3 py-2 font-medium w-[110px]">Model</th>
                    <th class="text-right px-3 py-2 font-medium w-[64px]">Latency</th>
                    <th class="text-left px-3 py-2 font-medium">Details</th>
                </tr>
            </thead>
            <tbody id="evt-tbody" class="divide-y divide-sage-800">
                <tr>
                    <td colspan="6" class="px-4 py-6 text-center text-sage-600 italic">
                        No events yet.
                    </td>
                </tr>
            </tbody>
        </table>
    </div>
</div>`;
}

// ----------------------------------------------------------------
// Private: row management
// ----------------------------------------------------------------

/**
 * Prepend a single new event row (respects current filter).
 * Trims the table to MAX_ROWS visible rows.
 * @param {object} evt
 */
function _addRow(evt) {
    if (!_tbody) return;

    const type = (evt.type || '').toUpperCase();
    if (_filter !== 'ALL' && type !== _filter) return;

    // Remove the placeholder "No events yet" row if present
    const placeholder = _tbody.querySelector('tr[data-placeholder]');
    if (placeholder) placeholder.remove();

    const tr = _createRow(evt, type);
    tr.classList.add('event-row-new');

    _tbody.insertBefore(tr, _tbody.firstChild);

    // Cap visible rows
    while (_tbody.rows.length > MAX_ROWS) {
        _tbody.deleteRow(_tbody.rows.length - 1);
    }
}

/**
 * Full rebuild of the table from state.events using the current filter.
 * Called on initial mount, filter change, or reset.
 */
function _rebuildTable() {
    if (!_tbody) return;

    const { events, totalEvents } = getState();

    // Clear existing rows
    _tbody.innerHTML = '';

    const filtered = _filter === 'ALL'
        ? events
        : events.filter(e => (e.type || '').toUpperCase() === _filter);

    const visible = filtered.slice(0, MAX_ROWS);

    if (visible.length === 0) {
        _tbody.innerHTML = `
            <tr data-placeholder>
                <td colspan="6" class="px-4 py-6 text-center text-sage-600 italic">
                    No events yet.
                </td>
            </tr>`;
    } else {
        for (const evt of visible) {
            _tbody.appendChild(_createRow(evt, (evt.type || '').toUpperCase()));
        }
    }

    _updateCount();
}

/**
 * Create a single <tr> element for an event.
 * @param {object} evt
 * @param {string} type  Normalised uppercase type string.
 * @returns {HTMLTableRowElement}
 */
function _createRow(evt, type) {
    const style   = TYPE_STYLES[type] || DEFAULT_STYLE;
    const meta    = evt.meta   || {};
    const ts      = evt.ts     != null ? evt.ts : (evt.timestamp ?? null);
    const step    = evt.step   != null ? evt.step : '--';
    const model   = evt.model  || meta.model || '--';
    const latency = evt.latency_ms != null
        ? `${evt.latency_ms.toFixed(0)} ms`
        : (meta.latency_ms != null ? `${Number(meta.latency_ms).toFixed(0)} ms` : '--');

    const details = _buildDetails(evt, meta);

    const tr = document.createElement('tr');
    tr.className = `border-b border-sage-800/60 transition-colors ${style.row}`;

    tr.innerHTML = `
        <td class="px-3 py-1.5 font-mono text-sage-400 whitespace-nowrap">${_fmtTime(ts)}</td>
        <td class="px-3 py-1.5 font-mono text-sage-400 text-center">${step}</td>
        <td class="px-3 py-1.5">
            <span class="inline-flex items-center gap-1.5 px-1.5 py-0.5 rounded border text-[10px] font-semibold tracking-wide ${style.badge}">
                <span class="w-1.5 h-1.5 rounded-full flex-shrink-0 ${style.dot}"></span>
                ${_trunc(type || '?', 18)}
            </span>
        </td>
        <td class="px-3 py-1.5 font-mono text-sage-300 truncate max-w-[110px]" title="${_esc(model)}">${_trunc(model, 14)}</td>
        <td class="px-3 py-1.5 font-mono text-sage-400 text-right whitespace-nowrap">${latency}</td>
        <td class="px-3 py-1.5 text-sage-400 truncate max-w-[260px]" title="${_esc(details)}">${_trunc(details, 80)}</td>
    `;

    return tr;
}

// ----------------------------------------------------------------
// Private: helpers
// ----------------------------------------------------------------

/**
 * Build a human-readable details string from an event.
 * Pulls validation, task, tool, content, result, error fields.
 * @param {object} evt
 * @param {object} meta
 * @returns {string}
 */
function _buildDetails(evt, meta) {
    const parts = [];

    if (evt.validation)   parts.push(evt.validation);
    if (evt.task)         parts.push(_trunc(String(evt.task), 40));
    if (meta.task)        parts.push(_trunc(String(meta.task), 40));
    if (meta.tool)        parts.push(`tool:${meta.tool}`);
    if (evt.tool)         parts.push(`tool:${evt.tool}`);
    if (meta.content)     parts.push(_trunc(String(meta.content), 60));
    if (evt.content)      parts.push(_trunc(String(evt.content), 60));
    if (meta.result)      parts.push(_trunc(String(meta.result), 60));
    if (evt.result)       parts.push(_trunc(String(evt.result), 60));
    if (evt.error)        parts.push(`error:${_trunc(String(evt.error), 60)}`);
    if (meta.error)       parts.push(`error:${_trunc(String(meta.error), 60)}`);

    return parts.join(' | ') || '--';
}

/**
 * Format a unix timestamp (seconds or ms) to HH:MM:SS.
 * Falls back to '--' for null/undefined.
 * @param {number|null} ts
 * @returns {string}
 */
function _fmtTime(ts) {
    if (ts == null) return '--:--:--';
    // Treat values < 1e10 as seconds, otherwise milliseconds
    const ms  = ts < 1e10 ? ts * 1000 : ts;
    const d   = new Date(ms);
    const pad = n => String(n).padStart(2, '0');
    return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

/**
 * Truncate a string to at most n characters, appending '…' if truncated.
 * @param {string} s
 * @param {number} n
 * @returns {string}
 */
function _trunc(s, n) {
    if (!s) return '';
    const str = String(s);
    return str.length <= n ? str : str.slice(0, n - 1) + '\u2026';
}

/**
 * HTML-escape a string for use in attributes.
 * @param {string} s
 * @returns {string}
 */
function _esc(s) {
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/"/g, '&quot;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

/**
 * Sync the count badge with the current totalEvents from state.
 */
function _updateCount() {
    if (!_countBadge) return;
    const { totalEvents } = getState();
    _countBadge.textContent = totalEvents > 999 ? '999+' : String(totalEvents);
}
