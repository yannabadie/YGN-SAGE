import { pushEvent, setState } from './state.js';

let ws = null;
let reconnectTimer = null;
const RECONNECT_DELAY = 2000;

export function connectWebSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${location.host}/ws`;
    ws = new WebSocket(url);
    ws.onopen = () => {
        const token = localStorage.getItem('sage_token') || '';
        if (token) ws.send(JSON.stringify({ action: 'auth', token }));
        setState({ wsConnected: true });
        if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
    };
    ws.onmessage = (msg) => {
        try { pushEvent(JSON.parse(msg.data)); } catch (e) { console.warn('Failed to parse event:', e); }
    };
    ws.onclose = () => {
        setState({ wsConnected: false });
        _scheduleReconnect();
    };
    ws.onerror = () => ws.close();
}

function _scheduleReconnect() {
    if (reconnectTimer) return;
    reconnectTimer = setTimeout(() => { reconnectTimer = null; connectWebSocket(); }, RECONNECT_DELAY);
}

export function disconnectWebSocket() {
    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
    if (ws) { ws.close(); ws = null; }
}
