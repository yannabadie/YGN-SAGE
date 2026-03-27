/**
 * providers.js — Provider health monitor module
 *
 * Renders a grid of provider cards with circuit breaker status,
 * latency metrics, and error rates. Subscribes to state reset events
 * and polls health data every 10 seconds.
 *
 * Exports: mount(el), unmount()
 */

import { subscribe } from './state.js';

'use strict';

// ----------------------------------------------------------------
// Module state
// ----------------------------------------------------------------
let _container = null;
let _pollTimer = null;
let _unsubscribe = null;
const POLL_INTERVAL = 10000; // 10 seconds

// ----------------------------------------------------------------
// Public API
// ----------------------------------------------------------------

/**
 * Mount the provider health monitor into the given DOM element.
 * @param {HTMLElement} el The container element
 */
export function mount(el) {
    _container = el;

    // Render initial header + grid
    _render({});

    // Fetch immediately
    _fetch();

    // Set up polling
    _pollTimer = setInterval(_fetch, POLL_INTERVAL);

    // Subscribe to state reset events
    _unsubscribe = subscribe((notification) => {
        if (notification.reset) {
            _fetch();
        }
    });
}

/**
 * Unmount the provider health monitor.
 * Clears polling timer and state subscriptions.
 */
export function unmount() {
    if (_pollTimer) {
        clearInterval(_pollTimer);
        _pollTimer = null;
    }
    if (_unsubscribe) {
        _unsubscribe();
        _unsubscribe = null;
    }
    _container = null;
}

// ----------------------------------------------------------------
// Internal helpers
// ----------------------------------------------------------------

/**
 * Fetch provider health data from backend.
 * GET /api/providers/health
 * Expected response: Array of provider objects
 */
async function _fetch() {
    if (!_container) return;

    try {
        const response = await fetch('/api/providers/health');
        if (!response.ok) {
            console.warn('[providers] health endpoint returned', response.status);
            return;
        }

        const providers = await response.json();
        _renderProviders(providers);
    } catch (err) {
        console.error('[providers] fetch error:', err);
    }
}

/**
 * Render the full provider health UI.
 */
function _render(data) {
    if (!_container) return;

    // Clear existing content
    _container.innerHTML = '';

    // Header
    const header = document.createElement('div');
    header.style.cssText = `
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
        padding: 0 4px;
    `;

    const title = document.createElement('h2');
    title.style.cssText = `
        margin: 0;
        font-size: 16px;
        font-weight: 600;
        color: #e2e8f0;
    `;
    title.textContent = 'Provider Health';

    const refreshBtn = document.createElement('button');
    refreshBtn.style.cssText = `
        padding: 6px 12px;
        font-size: 12px;
        font-weight: 500;
        color: #38bdf8;
        background: transparent;
        border: 1px solid #38bdf8;
        border-radius: 4px;
        cursor: pointer;
        transition: background 0.15s ease;
    `;
    refreshBtn.textContent = '↻ Refresh';
    refreshBtn.addEventListener('click', _fetch);
    refreshBtn.addEventListener('mouseover', () => {
        refreshBtn.style.background = 'rgba(56, 189, 248, 0.1)';
    });
    refreshBtn.addEventListener('mouseout', () => {
        refreshBtn.style.background = 'transparent';
    });

    header.appendChild(title);
    header.appendChild(refreshBtn);
    _container.appendChild(header);

    // Grid container (2-3 columns responsive)
    const grid = document.createElement('div');
    grid.className = 'provider-grid';
    grid.style.cssText = `
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
        gap: 12px;
        padding: 0 4px;
    `;
    grid.id = 'provider-grid';

    _container.appendChild(grid);
}

/**
 * Render individual provider cards into the grid.
 * @param {Array} providers Array of provider health objects
 */
function _renderProviders(providers) {
    if (!_container) return;

    const grid = _container.querySelector('#provider-grid');
    if (!grid) {
        _render({});
        return _renderProviders(providers);
    }

    // Clear existing cards
    grid.innerHTML = '';

    // If no providers, show placeholder
    if (!providers || providers.length === 0) {
        const empty = document.createElement('div');
        empty.style.cssText = `
            grid-column: 1 / -1;
            padding: 24px;
            text-align: center;
            color: #64748b;
            font-size: 13px;
        `;
        empty.textContent = 'No providers available';
        grid.appendChild(empty);
        return;
    }

    // Render each provider
    for (const provider of providers) {
        const card = _makeProviderCard(provider);
        grid.appendChild(card);
    }
}

/**
 * Create a single provider card element.
 * @param {object} provider Provider health object
 * @returns {HTMLElement}
 */
function _makeProviderCard(provider) {
    const card = document.createElement('div');
    card.className = 'provider-card';

    // Determine health status for border/shadow styling
    const circuitState = provider.circuit_state || 'unknown';
    if (circuitState === 'closed') {
        card.classList.add('healthy');
    } else if (circuitState === 'half-open') {
        card.classList.add('degraded');
    } else if (circuitState === 'open') {
        card.classList.add('down');
    }

    card.style.cssText = `
        display: flex;
        flex-direction: column;
        gap: 8px;
    `;

    // ---- Header: Status dot + Model ID + Provider ----
    const header = document.createElement('div');
    header.style.cssText = `
        display: flex;
        align-items: center;
        gap: 8px;
    `;

    const statusDot = document.createElement('div');
    statusDot.className = 'status-dot';
    statusDot.style.cssText = `
        width: 8px;
        height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
    `;

    if (provider.available !== false) {
        statusDot.style.background = '#22c55e'; // green
    } else {
        statusDot.style.background = '#ef4444'; // red
    }

    const modelId = document.createElement('div');
    modelId.style.cssText = `
        flex: 1;
        font-size: 12px;
        font-weight: 600;
        color: #f8fafc;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    `;
    modelId.textContent = provider.id || 'unknown';
    modelId.title = provider.id || ''; // Tooltip for full text

    const providerName = document.createElement('div');
    providerName.style.cssText = `
        font-size: 11px;
        color: #94a3b8;
        white-space: nowrap;
    `;
    providerName.textContent = provider.provider || '—';

    header.appendChild(statusDot);
    header.appendChild(modelId);
    header.appendChild(providerName);
    card.appendChild(header);

    // ---- Metrics grid: Circuit | P50 | Error Rate | Code Score ----
    const metricsGrid = document.createElement('div');
    metricsGrid.style.cssText = `
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        font-size: 11px;
    `;

    // Circuit state
    const circuitRow = document.createElement('div');
    circuitRow.style.cssText = `
        display: flex;
        flex-direction: column;
        gap: 2px;
    `;
    const circuitLabel = document.createElement('div');
    circuitLabel.style.color = '#64748b';
    circuitLabel.textContent = 'Circuit';
    const circuitValue = document.createElement('div');
    circuitValue.style.cssText = `
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
        width: fit-content;
    `;
    const circuitText = _formatCircuitState(circuitState);
    circuitValue.textContent = circuitText;

    // Color code circuit state
    if (circuitState === 'closed') {
        circuitValue.style.cssText += 'color: #22c55e; background: rgba(34, 197, 94, 0.1);';
    } else if (circuitState === 'half-open') {
        circuitValue.style.cssText += 'color: #f59e0b; background: rgba(245, 158, 11, 0.1);';
    } else if (circuitState === 'open') {
        circuitValue.style.cssText += 'color: #ef4444; background: rgba(239, 68, 68, 0.1);';
    }

    circuitRow.appendChild(circuitLabel);
    circuitRow.appendChild(circuitValue);

    // P50 latency
    const p50Row = document.createElement('div');
    p50Row.style.cssText = `
        display: flex;
        flex-direction: column;
        gap: 2px;
    `;
    const p50Label = document.createElement('div');
    p50Label.style.color = '#64748b';
    p50Label.textContent = 'P50 Latency';
    const p50Value = document.createElement('div');
    p50Value.style.color = '#e2e8f0';
    p50Value.textContent = provider.latency_p50_ms != null
        ? `${provider.latency_p50_ms.toFixed(1)}ms`
        : '—';
    p50Row.appendChild(p50Label);
    p50Row.appendChild(p50Value);

    // P99 latency
    const p99Row = document.createElement('div');
    p99Row.style.cssText = `
        display: flex;
        flex-direction: column;
        gap: 2px;
    `;
    const p99Label = document.createElement('div');
    p99Label.style.color = '#64748b';
    p99Label.textContent = 'P99 Latency';
    const p99Value = document.createElement('div');
    p99Value.style.color = '#e2e8f0';
    p99Value.textContent = provider.latency_p99_ms != null
        ? `${provider.latency_p99_ms.toFixed(1)}ms`
        : '—';
    p99Row.appendChild(p99Label);
    p99Row.appendChild(p99Value);

    // Error rate
    const errorRow = document.createElement('div');
    errorRow.style.cssText = `
        display: flex;
        flex-direction: column;
        gap: 2px;
    `;
    const errorLabel = document.createElement('div');
    errorLabel.style.color = '#64748b';
    errorLabel.textContent = 'Error Rate';
    const errorValue = document.createElement('div');
    const errorRate = provider.error_rate_1h != null ? provider.error_rate_1h * 100 : null;
    errorValue.textContent = errorRate != null ? `${errorRate.toFixed(1)}%` : '—';
    errorValue.style.color = errorRate != null && errorRate > 5 ? '#ef4444' : '#e2e8f0';
    errorRow.appendChild(errorLabel);
    errorRow.appendChild(errorValue);

    // Code score
    const scoreRow = document.createElement('div');
    scoreRow.style.cssText = `
        display: flex;
        flex-direction: column;
        gap: 2px;
    `;
    const scoreLabel = document.createElement('div');
    scoreLabel.style.color = '#64748b';
    scoreLabel.textContent = 'Code Score';
    const scoreValue = document.createElement('div');
    scoreValue.style.color = '#e2e8f0';
    scoreValue.textContent = provider.code_score != null
        ? `${(provider.code_score * 100).toFixed(0)}%`
        : '—';
    scoreRow.appendChild(scoreLabel);
    scoreRow.appendChild(scoreValue);

    metricsGrid.appendChild(circuitRow);
    metricsGrid.appendChild(p50Row);
    metricsGrid.appendChild(p99Row);
    metricsGrid.appendChild(errorRow);
    metricsGrid.appendChild(scoreRow);

    card.appendChild(metricsGrid);

    return card;
}

/**
 * Format circuit state for display.
 * @param {string} state Circuit breaker state
 * @returns {string}
 */
function _formatCircuitState(state) {
    const map = {
        'closed': 'Closed',
        'half-open': 'Half-Open',
        'open': 'Open',
    };
    return map[state] || state || 'Unknown';
}
