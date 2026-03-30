"""Browser integration tests for YGN-SAGE Agent UI.

Requires:
  - Server running on http://127.0.0.1:8765
  - pip install playwright && python -m playwright install chromium

Run: python tests/test_ui_browser.py
"""
import json
import os
import sys
import time

os.environ["PYTHONIOENCODING"] = "utf-8"
# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Playwright launch ────────────────────────────────────────────
from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8765"
RESULTS = []


def log(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    RESULTS.append((name, ok, detail))
    mark = "\u2713" if ok else "\u2717"
    print(f"  {mark} {name}" + (f"  ({detail})" if detail else ""))


def run_tests():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))

        # ── Test 1: Page loads ────────────────────────────────────
        print("\n== Page Load ==")
        resp = page.goto(BASE, wait_until="networkidle")
        log("HTTP 200", resp.status == 200, f"status={resp.status}")
        log("Title contains YGN-SAGE", "YGN-SAGE" in page.title())

        # ── Test 2: No JS errors on load ──────────────────────────
        print("\n== Console Errors ==")
        # Give modules time to initialize
        page.wait_for_timeout(2000)
        js_errors = [e for e in errors if "net::ERR" not in e]  # ignore WebSocket connection errors to missing WS
        log("No JS errors on load", len(js_errors) == 0,
            f"{len(js_errors)} errors: {js_errors[:3]}" if js_errors else "clean")

        # ── Test 3: ES Modules loaded ─────────────────────────────
        print("\n== ES Modules ==")
        # state.js should have set up the global state
        has_modules = page.evaluate("""() => {
            // Check that the module script executed by verifying DOM elements exist
            return {
                chatPanel: !!document.getElementById('panel-chat'),
                dashPanel: !!document.getElementById('panel-dashboard'),
                topoPanel: !!document.getElementById('panel-topology'),
                provPanel: !!document.getElementById('panel-providers'),
                eventsBar: !!document.getElementById('events-container'),
                wsStatus:  !!document.getElementById('ws-dot'),
            }
        }""")
        for key, val in has_modules.items():
            log(f"DOM element: {key}", val)

        # ── Test 4: Tab panels structure ──────────────────────────
        print("\n== Tab System ==")
        # Chat should be the active tab by default
        chat_visible = page.evaluate("""() => {
            const panel = document.getElementById('panel-chat');
            const style = window.getComputedStyle(panel);
            return {
                display: style.display,
                hasActive: panel.classList.contains('active'),
            }
        }""")
        log("Chat panel is active", chat_visible["hasActive"])
        log("Chat panel display=flex", chat_visible["display"] == "flex",
            f"display={chat_visible['display']}")

        # Dashboard should be hidden
        dash_visible = page.evaluate("""() => {
            const panel = document.getElementById('panel-dashboard');
            const style = window.getComputedStyle(panel);
            return style.display;
        }""")
        log("Dashboard panel hidden", dash_visible == "none", f"display={dash_visible}")

        # ── Test 5: Tab switching ─────────────────────────────────
        print("\n== Tab Switching ==")
        # Click Dashboard tab
        page.click('button[data-tab="dashboard"]')
        page.wait_for_timeout(500)

        dash_after = page.evaluate("""() => {
            const panel = document.getElementById('panel-dashboard');
            const style = window.getComputedStyle(panel);
            return {
                display: style.display,
                hasActive: panel.classList.contains('active'),
                hasContent: panel.innerHTML.length > 50,
            }
        }""")
        log("Dashboard tab activates", dash_after["hasActive"])
        log("Dashboard display=flex", dash_after["display"] == "flex",
            f"display={dash_after['display']}")
        log("Dashboard has content", dash_after["hasContent"],
            f"innerHTML length={page.evaluate('document.getElementById(\"panel-dashboard\").innerHTML.length')}")

        chat_after = page.evaluate("""() => {
            const panel = document.getElementById('panel-chat');
            return window.getComputedStyle(panel).display;
        }""")
        log("Chat panel hidden after switch", chat_after == "none", f"display={chat_after}")

        # Switch to Topology
        page.click('button[data-tab="topology"]')
        page.wait_for_timeout(500)

        topo_after = page.evaluate("""() => {
            const panel = document.getElementById('panel-topology');
            const style = window.getComputedStyle(panel);
            return {
                display: style.display,
                hasActive: panel.classList.contains('active'),
                hasCyContainer: !!panel.querySelector('#cy-container'),
            }
        }""")
        log("Topology tab activates", topo_after["hasActive"])
        log("Topology has #cy-container", topo_after["hasCyContainer"])

        # Switch to Providers
        page.click('button[data-tab="providers"]')
        page.wait_for_timeout(500)
        prov_after = page.evaluate("""() => {
            const panel = document.getElementById('panel-providers');
            return {
                active: panel.classList.contains('active'),
                hasContent: panel.innerHTML.length > 50,
            }
        }""")
        log("Providers tab activates", prov_after["active"])
        log("Providers has content", prov_after["hasContent"])

        # Switch back to Chat
        page.click('button[data-tab="chat"]')
        page.wait_for_timeout(300)

        # ── Test 6: Chat module renders ───────────────────────────
        print("\n== Chat Module ==")
        chat_dom = page.evaluate("""() => {
            const panel = document.getElementById('panel-chat');
            return {
                hasTextarea: !!panel.querySelector('textarea'),
                hasSendBtn: !!panel.querySelector('button'),
                hasMessagesArea: panel.innerHTML.includes('chat') || panel.innerHTML.includes('message'),
            }
        }""")
        log("Chat has textarea", chat_dom["hasTextarea"])
        log("Chat has send button", chat_dom["hasSendBtn"])

        # ── Test 7: Events bar always visible ─────────────────────
        print("\n== Events Bar ==")
        events_dom = page.evaluate("""() => {
            const bar = document.getElementById('events-bar');
            const container = document.getElementById('events-container');
            const style = window.getComputedStyle(bar);
            return {
                barVisible: style.display !== 'none',
                hasContent: container && container.innerHTML.length > 50,
                hasTable: container && !!container.querySelector('table'),
            }
        }""")
        log("Events bar visible", events_dom["barVisible"])
        log("Events has content", events_dom["hasContent"])
        log("Events has table", events_dom["hasTable"])

        # ── Test 8: CDN libraries loaded ──────────────────────────
        print("\n== CDN Libraries ==")
        libs = page.evaluate("""() => ({
            tailwind: typeof tailwind !== 'undefined',
            Chart: typeof Chart !== 'undefined',
            cytoscape: typeof cytoscape !== 'undefined',
            ELK: typeof ELK !== 'undefined',
            marked: typeof marked !== 'undefined',
        })""")
        for lib, loaded in libs.items():
            log(f"CDN: {lib}", loaded)

        # ── Test 9: API endpoints from browser ────────────────────
        print("\n== API Endpoints ==")
        endpoints = [
            ("/api/state", "step_count"),
            ("/api/topology/graph", "nodes"),
            ("/api/providers/health", None),  # returns array
            ("/api/routing/pipeline", "stages"),
        ]
        for path, key in endpoints:
            result = page.evaluate(f"""async () => {{
                try {{
                    const r = await fetch('{BASE}{path}');
                    const d = await r.json();
                    return {{ status: r.status, hasKey: {f'"{key}" in d' if key else 'Array.isArray(d)'} }};
                }} catch(e) {{
                    return {{ status: 0, hasKey: false, error: e.message }};
                }}
            }}""")
            log(f"GET {path}", result["status"] == 200 and result["hasKey"],
                f"status={result['status']}")

        # ── Test 10: WebSocket connection ─────────────────────────
        print("\n== WebSocket ==")
        page.wait_for_timeout(3000)  # Give WS time to connect
        ws_state = page.evaluate("""() => {
            const dot = document.getElementById('ws-dot');
            const label = document.getElementById('ws-label');
            return {
                dotClass: dot ? dot.className : 'not found',
                labelText: label ? label.textContent : 'not found',
            }
        }""")
        ws_connected = "green" in ws_state["dotClass"]
        log("WebSocket connected", ws_connected,
            f"dot={ws_state['dotClass']}, label={ws_state['labelText']}")

        # ── Test 11: Header stats render ──────────────────────────
        print("\n== Header Stats ==")
        header = page.evaluate("""() => ({
            step: document.getElementById('hdr-step')?.textContent,
            cost: document.getElementById('hdr-cost')?.textContent,
            model: document.getElementById('hdr-model')?.textContent,
        })""")
        log("Header step renders", header["step"] is not None, f"value='{header['step']}'")
        log("Header cost renders", header["cost"] is not None and header["cost"] != "",
            f"value='{header['cost']}'")
        log("Header model renders", header["model"] is not None, f"value='{header['model']}'")

        # ── Test 12: Auth banner visible (no token set) ───────────
        print("\n== Auth Banner ==")
        page.wait_for_timeout(500)
        banner = page.evaluate("""() => {
            const b = document.getElementById('auth-banner');
            return b ? window.getComputedStyle(b).display : 'not found';
        }""")
        log("Auth banner shown (no token)", banner != "none",
            f"display={banner}")

        # ── Test 13: Dashboard module content check ───────────────
        print("\n== Dashboard Content ==")
        page.click('button[data-tab="dashboard"]')
        page.wait_for_timeout(800)
        dash_content = page.evaluate("""() => {
            const panel = document.getElementById('panel-dashboard');
            const html = panel.innerHTML;
            return {
                hasMemory: html.includes('Memory') || html.includes('memory') || html.includes('STM'),
                hasGuardrails: html.includes('Guardrail') || html.includes('guardrail') || html.includes('Z3') || html.includes('CoT'),
                hasRouting: html.includes('Routing') || html.includes('routing') || html.includes('kNN'),
                hasStats: html.includes('Statistic') || html.includes('Steps') || html.includes('LLM'),
                hasLatencyChart: !!panel.querySelector('canvas'),
                hasTaskInput: !!panel.querySelector('textarea'),
            }
        }""")
        for key, val in dash_content.items():
            log(f"Dashboard: {key}", val)

        # ── Test 14: Topology module Cytoscape init ───────────────
        print("\n== Topology Cytoscape ==")
        page.click('button[data-tab="topology"]')
        page.wait_for_timeout(1000)
        topo_content = page.evaluate("""() => {
            const panel = document.getElementById('panel-topology');
            const cyEl = panel.querySelector('#cy-container');
            return {
                hasCyContainer: !!cyEl,
                cyHasCanvas: cyEl ? !!cyEl.querySelector('canvas') : false,
                hasLegend: panel.innerHTML.includes('Control') && panel.innerHTML.includes('Message') && panel.innerHTML.includes('State'),
                hasSourceBadge: panel.innerHTML.includes('source') || panel.innerHTML.includes('topo-source') || panel.innerHTML.includes('conf'),
            }
        }""")
        log("Topology: #cy-container exists", topo_content["hasCyContainer"])
        log("Topology: Cytoscape canvas rendered", topo_content["cyHasCanvas"])
        log("Topology: 3-flow legend present", topo_content["hasLegend"])

        # ── Final JS errors check ─────────────────────────────────
        print("\n== Final Error Check ==")
        all_errors = [e for e in errors if "WebSocket" not in e and "net::ERR" not in e]
        log("No JS errors during test session", len(all_errors) == 0,
            f"{len(all_errors)} errors" if all_errors else "clean")
        if all_errors:
            for e in all_errors[:5]:
                print(f"    ERROR: {e[:200]}")

        browser.close()

    # ── Summary ───────────────────────────────────────────────────
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = sum(1 for _, ok, _ in RESULTS if not ok)
    total = len(RESULTS)
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    if failed:
        print("\nFAILURES:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  \u2717 {name}  ({detail})")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
