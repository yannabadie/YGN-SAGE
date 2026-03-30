/**
 * chat.js — YGN-SAGE v2 conversational chat with SSE streaming
 *
 * Exports: mount(el), unmount()
 *
 * Features:
 *  - SSE streaming from POST /api/chat/stream
 *  - Markdown rendering via marked.js (CDN, optional fallback)
 *  - S1/S2/S3 cognitive system badges on completed assistant messages
 *  - AbortController-backed Stop button
 *  - Syncs with shared state (chatMessages array)
 */

'use strict';

import { getState, setState, subscribe } from './state.js';

// ----------------------------------------------------------------
// Module-level state
// ----------------------------------------------------------------
let _container = null;       // root element passed to mount()
let _messagesEl = null;      // scrollable messages div
let _textareaEl = null;      // input textarea
let _sendBtn = null;         // Send button element
let _stopBtn = null;         // Stop button element
let _statusBar = null;       // status text element
let _abortCtrl = null;       // AbortController for active stream
let _streaming = false;      // true while SSE stream is active
let _unsubscribe = null;     // state subscription cleanup
let _activeAssistantEl = null; // current streaming assistant bubble content el

// ----------------------------------------------------------------
// Public API
// ----------------------------------------------------------------

/**
 * Mount the chat UI into el.
 * @param {HTMLElement} el
 */
export function mount(el) {
    if (_container) unmount();
    _container = el;
    _render();

    // Render existing messages from state
    const { chatMessages } = getState();
    for (const msg of chatMessages) {
        _appendMessageEl(_renderMessage(msg));
    }
    _scrollToBottom();

    // Subscribe to state resets
    _unsubscribe = subscribe((hint) => {
        if (hint && hint.reset) {
            _clearMessages();
        }
    });
}

/**
 * Unmount and clean up.
 */
export function unmount() {
    if (_abortCtrl) {
        _abortCtrl.abort();
        _abortCtrl = null;
    }
    if (_unsubscribe) {
        _unsubscribe();
        _unsubscribe = null;
    }
    if (_container) {
        _container.innerHTML = '';
    }
    _container = null;
    _messagesEl = null;
    _textareaEl = null;
    _sendBtn = null;
    _stopBtn = null;
    _statusBar = null;
    _streaming = false;
    _activeAssistantEl = null;
}

// ----------------------------------------------------------------
// Rendering
// ----------------------------------------------------------------

function _render() {
    _container.innerHTML = '';
    _container.style.cssText = 'display:flex;flex-direction:column;height:100%;min-height:0;';

    // Messages area
    _messagesEl = document.createElement('div');
    _messagesEl.style.cssText = [
        'flex:1',
        'overflow-y:auto',
        'padding:16px',
        'display:flex',
        'flex-direction:column',
        'gap:12px',
        'min-height:0',
    ].join(';');
    _container.appendChild(_messagesEl);

    // Status bar
    _statusBar = document.createElement('div');
    _statusBar.style.cssText = [
        'padding:4px 16px',
        'font-size:11px',
        'color:#64748b',
        'min-height:20px',
        'line-height:20px',
        'border-top:1px solid #1e293b',
        'transition:color 0.2s',
    ].join(';');
    _container.appendChild(_statusBar);

    // Input area
    const inputArea = document.createElement('div');
    inputArea.style.cssText = [
        'display:flex',
        'gap:8px',
        'padding:12px 16px',
        'border-top:1px solid #1e293b',
        'background:#0f172a',
        'align-items:flex-end',
    ].join(';');

    _textareaEl = document.createElement('textarea');
    _textareaEl.placeholder = 'Ask SAGE anything…';
    _textareaEl.rows = 2;
    _textareaEl.style.cssText = [
        'flex:1',
        'resize:none',
        'background:#1e293b',
        'color:#f1f5f9',
        'border:1px solid #334155',
        'border-radius:8px',
        'padding:8px 12px',
        'font-size:13px',
        'font-family:inherit',
        'line-height:1.5',
        'outline:none',
        'transition:border-color 0.15s',
    ].join(';');
    _textareaEl.addEventListener('focus', () => {
        _textareaEl.style.borderColor = '#38bdf8';
    });
    _textareaEl.addEventListener('blur', () => {
        _textareaEl.style.borderColor = '#334155';
    });
    _textareaEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!_streaming) _send();
        }
    });

    // Button container
    const btnBox = document.createElement('div');
    btnBox.style.cssText = 'display:flex;flex-direction:column;gap:6px;';

    _sendBtn = document.createElement('button');
    _sendBtn.textContent = 'Send';
    _sendBtn.style.cssText = _btnStyle('#0ea5e9', '#fff');
    _sendBtn.addEventListener('click', () => { if (!_streaming) _send(); });

    _stopBtn = document.createElement('button');
    _stopBtn.textContent = 'Stop';
    _stopBtn.style.cssText = _btnStyle('#475569', '#94a3b8');
    _stopBtn.disabled = true;
    _stopBtn.addEventListener('click', () => _stop());

    btnBox.appendChild(_sendBtn);
    btnBox.appendChild(_stopBtn);
    inputArea.appendChild(_textareaEl);
    inputArea.appendChild(btnBox);
    _container.appendChild(inputArea);
}

function _btnStyle(bg, color) {
    return [
        `background:${bg}`,
        `color:${color}`,
        'border:none',
        'border-radius:6px',
        'padding:7px 14px',
        'font-size:12px',
        'font-weight:600',
        'cursor:pointer',
        'white-space:nowrap',
        'transition:opacity 0.15s',
        'min-width:58px',
    ].join(';');
}

// ----------------------------------------------------------------
// Send / Stop
// ----------------------------------------------------------------

async function _send() {
    if (_streaming) return;
    const text = (_textareaEl.value || '').trim();
    if (!text) return;

    // 1. Push user message to state
    const userMsg = {
        role: 'user',
        content: text,
        timestamp: new Date().toISOString(),
    };
    const prev = getState().chatMessages;
    setState({ chatMessages: [...prev, userMsg] });

    // 2. Render user bubble
    _appendMessageEl(_renderMessage(userMsg));

    // 3. Clear textarea and set streaming state
    _textareaEl.value = '';
    _setStreaming(true);

    // 4. Create assistant placeholder
    const assistantMsg = {
        role: 'assistant',
        content: '',
        timestamp: new Date().toISOString(),
        streaming: true,
    };

    const { outerEl, contentEl, headerEl } = _renderMessage(assistantMsg);
    _activeAssistantEl = contentEl;
    _appendMessageEl({ outerEl, contentEl, headerEl });

    // 5. POST to SSE stream
    _abortCtrl = new AbortController();
    let accContent = '';
    let finalSystem = null;
    let finalModel = null;

    try {
        const token = localStorage.getItem('sage_token') || '';
        const headers = { 'Content-Type': 'application/json' };
        if (token) headers['Authorization'] = `Bearer ${token}`;

        const resp = await fetch('/api/chat/stream', {
            method: 'POST',
            headers,
            body: JSON.stringify({ message: text }),
            signal: _abortCtrl.signal,
        });

        if (!resp.ok) {
            throw new Error(`HTTP ${resp.status}: ${resp.statusText}`);
        }

        // 6. Read SSE stream
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buf += decoder.decode(value, { stream: true });
            const lines = buf.split('\n');
            buf = lines.pop(); // keep incomplete line

            for (const line of lines) {
                if (!line.startsWith('data:')) continue;
                const raw = line.slice(5).trim();
                if (!raw) continue;

                let evt;
                try { evt = JSON.parse(raw); } catch { continue; }

                // 7. Handle event types
                if (evt.type === 'start') {
                    // Show model name in status bar
                    finalModel = evt.model || null;
                    if (finalModel) {
                        _setStatus(`Streaming · ${_escapeHtml(finalModel)}`);
                    }
                    // Apply streaming cursor
                    _activeAssistantEl.classList.add('chat-streaming-cursor');

                } else if (evt.type === 'delta') {
                    // Append content and render markdown
                    accContent += (evt.content || '');
                    _activeAssistantEl.innerHTML = _renderMarkdown(accContent);
                    _activeAssistantEl.classList.add('chat-streaming-cursor');
                    _scrollToBottom();

                } else if (evt.type === 'done') {
                    // Remove cursor, add S-badge
                    _activeAssistantEl.classList.remove('chat-streaming-cursor');
                    finalSystem = evt.system != null ? evt.system : null;

                    if (finalSystem && headerEl) {
                        const badge = _makeSBadge(finalSystem);
                        headerEl.appendChild(badge);
                    }

                    // Persist completed message to state
                    const completedMsg = {
                        role: 'assistant',
                        content: accContent,
                        timestamp: assistantMsg.timestamp,
                        model: finalModel,
                        system: finalSystem,
                    };
                    const current = getState().chatMessages;
                    setState({ chatMessages: [...current, completedMsg] });

                    _setStatus(finalModel ? `Done · ${_escapeHtml(finalModel)}` : 'Done');

                } else if (evt.type === 'error') {
                    _activeAssistantEl.classList.remove('chat-streaming-cursor');
                    const errText = evt.content || 'Unknown error';
                    _activeAssistantEl.innerHTML =
                        `<span style="color:#f87171">${_escapeHtml(errText)}</span>`;
                    _setStatus('Error', true);
                }
            }
        }

    } catch (err) {
        if (err.name === 'AbortError') {
            // User stopped — finalize gracefully
            if (_activeAssistantEl) {
                _activeAssistantEl.classList.remove('chat-streaming-cursor');
                if (accContent) {
                    _activeAssistantEl.innerHTML = _renderMarkdown(accContent);
                } else {
                    _activeAssistantEl.innerHTML =
                        '<span style="color:#94a3b8;font-style:italic">Stopped.</span>';
                }
            }
            _setStatus('Stopped');
        } else {
            if (_activeAssistantEl) {
                _activeAssistantEl.classList.remove('chat-streaming-cursor');
                _activeAssistantEl.innerHTML =
                    `<span style="color:#f87171">${_escapeHtml(String(err))}</span>`;
            }
            _setStatus('Error', true);
        }
    } finally {
        _setStreaming(false);
        _abortCtrl = null;
        _activeAssistantEl = null;
        _scrollToBottom();
    }
}

function _stop() {
    if (_abortCtrl) {
        _abortCtrl.abort();
    }
}

// ----------------------------------------------------------------
// Message rendering
// ----------------------------------------------------------------

/**
 * Build a message DOM element.
 * Returns { outerEl, contentEl, headerEl } so the caller can mutate contentEl.
 * @param {{ role: string, content: string, timestamp?: string, system?: number, model?: string, streaming?: boolean }} msg
 */
function _renderMessage(msg) {
    const isUser = msg.role === 'user';

    // Outer wrapper (chat-msg + alignment)
    const outerEl = document.createElement('div');
    outerEl.className = 'chat-msg ' + (isUser ? 'chat-user' : 'chat-assistant');

    if (isUser) {
        // Right-align with left margin
        outerEl.style.marginLeft = '3rem';
        outerEl.style.alignSelf = 'flex-end';
    } else {
        // Left-align with right margin
        outerEl.style.marginRight = '1rem';
        outerEl.style.alignSelf = 'flex-start';
    }

    // Header row: role label + timestamp + optional S-badge
    const headerEl = document.createElement('div');
    headerEl.style.cssText = [
        'display:flex',
        'align-items:center',
        'gap:8px',
        'margin-bottom:4px',
        'font-size:11px',
        'font-weight:600',
        'letter-spacing:0.03em',
    ].join(';');

    const roleLabel = document.createElement('span');
    roleLabel.textContent = isUser ? 'You' : 'SAGE';
    roleLabel.style.color = isUser ? '#93c5fd' : '#7dd3fc'; // blue-300 / sky-300

    const tsLabel = document.createElement('span');
    tsLabel.style.color = '#475569'; // muted
    tsLabel.style.fontWeight = '400';
    if (msg.timestamp) {
        try {
            tsLabel.textContent = new Date(msg.timestamp).toLocaleTimeString([], {
                hour: '2-digit', minute: '2-digit', second: '2-digit',
            });
        } catch {
            tsLabel.textContent = '';
        }
    }

    headerEl.appendChild(roleLabel);
    headerEl.appendChild(tsLabel);

    // S-badge if system is already known (replayed messages)
    if (!msg.streaming && msg.system != null) {
        headerEl.appendChild(_makeSBadge(msg.system));
    }

    // Content area
    const contentEl = document.createElement('div');
    contentEl.style.cssText = [
        'word-break:break-word',
        'overflow-wrap:anywhere',
        'line-height:1.6',
    ].join(';');

    if (isUser) {
        contentEl.textContent = msg.content;  // safe — no HTML for user input
    } else {
        contentEl.innerHTML = msg.content ? _renderMarkdown(msg.content) : '';
        if (msg.streaming) {
            contentEl.classList.add('chat-streaming-cursor');
        }
    }

    outerEl.appendChild(headerEl);
    outerEl.appendChild(contentEl);

    return { outerEl, contentEl, headerEl };
}

/**
 * Build an S1/S2/S3 badge element.
 * @param {number} system  1 | 2 | 3
 */
function _makeSBadge(system) {
    const colors = {
        1: { bg: '#1e3a5f', border: '#1d4ed8', text: '#93c5fd' },  // blue — fast/reactive
        2: { bg: '#1c3a2a', border: '#15803d', text: '#86efac' },  // green — deliberate
        3: { bg: '#2d1b4e', border: '#7c3aed', text: '#c4b5fd' },  // purple — executive
    };
    const c = colors[system] || { bg: '#1e293b', border: '#475569', text: '#94a3b8' };

    const badge = document.createElement('span');
    badge.textContent = `S${system}`;
    badge.style.cssText = [
        `background:${c.bg}`,
        `border:1px solid ${c.border}`,
        `color:${c.text}`,
        'border-radius:4px',
        'padding:1px 5px',
        'font-size:10px',
        'font-weight:700',
        'letter-spacing:0.05em',
        'font-family:monospace',
        'flex-shrink:0',
    ].join(';');
    badge.title = `Cognitive system S${system} (${
        system === 1 ? 'Fast/Reactive' :
        system === 2 ? 'Deliberate/Analytical' :
        'Executive/Strategic'
    })`;
    return badge;
}

// ----------------------------------------------------------------
// Markdown + HTML escaping
// ----------------------------------------------------------------

/**
 * Render markdown to HTML.
 * Uses marked.js if available (loaded via CDN), else falls back to escaped HTML.
 * @param {string} text
 * @returns {string}  HTML string
 */
function _renderMarkdown(text) {
    if (!text) return '';

    if (typeof marked !== 'undefined' && marked.parse) {
        try {
            // marked v4+ API
            return marked.parse(text, { breaks: true });
        } catch {
            // fall through to plain fallback
        }
    }

    // Fallback: escape HTML, convert newlines to <br>
    return _escapeHtml(text).replace(/\n/g, '<br>');
}

/**
 * Escape special HTML characters to prevent XSS.
 * @param {string} s
 * @returns {string}
 */
function _escapeHtml(s) {
    const div = document.createElement('div');
    div.textContent = s;
    return div.innerHTML;
}

// ----------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------

function _appendMessageEl({ outerEl }) {
    _messagesEl.appendChild(outerEl);
}

function _clearMessages() {
    if (_messagesEl) _messagesEl.innerHTML = '';
    // Also clear any in-progress stream
    if (_abortCtrl) { _abortCtrl.abort(); _abortCtrl = null; }
    _streaming = false;
    _activeAssistantEl = null;
    _setStreaming(false);
    _setStatus('');
}

function _scrollToBottom() {
    if (_messagesEl) {
        _messagesEl.scrollTop = _messagesEl.scrollHeight;
    }
}

function _setStreaming(active) {
    _streaming = active;
    if (_sendBtn) {
        _sendBtn.disabled = active;
        _sendBtn.style.opacity = active ? '0.45' : '1';
        _sendBtn.style.cursor = active ? 'not-allowed' : 'pointer';
    }
    if (_stopBtn) {
        _stopBtn.disabled = !active;
        _stopBtn.style.opacity = active ? '1' : '0.45';
        _stopBtn.style.cursor = active ? 'pointer' : 'not-allowed';
        _stopBtn.style.background = active ? '#dc2626' : '#475569';
        _stopBtn.style.color = active ? '#fff' : '#94a3b8';
    }
    if (_textareaEl) {
        _textareaEl.disabled = active;
        _textareaEl.style.opacity = active ? '0.6' : '1';
    }
}

function _setStatus(text, isError = false) {
    if (_statusBar) {
        _statusBar.textContent = text;
        _statusBar.style.color = isError ? '#f87171' : '#64748b';
    }
}
