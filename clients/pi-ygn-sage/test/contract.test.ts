import assert from "node:assert/strict";
import test from "node:test";

import {
  SAGE_CLI_ENVELOPE_EVENT_TYPES,
  SAGE_CLI_PROTOCOL_VERSION,
  SAGE_INBOUND_COMMAND_TYPES,
  SAGE_OUTBOUND_EVENT_TYPES,
  SAGE_RUNTIME_EVENT_TYPES,
  createSageBridge,
} from "../src/index.js";

test("runtime event catalog is the 15-event v0 catalog", () => {
  assert.equal(SAGE_RUNTIME_EVENT_TYPES.length, 15);
  assert.ok(SAGE_RUNTIME_EVENT_TYPES.includes("prompt_injection_detected"));
  assert.equal(
    (SAGE_RUNTIME_EVENT_TYPES as readonly string[]).includes("task_ended"),
    false,
  );
});

test("outbound catalog is 15 inherited runtime events plus 4 CLI envelope events", () => {
  assert.equal(SAGE_CLI_ENVELOPE_EVENT_TYPES.length, 4);
  assert.equal(SAGE_OUTBOUND_EVENT_TYPES.length, 19);
});

test("inbound command catalog remains 5 verbs, types-only for now", () => {
  assert.deepEqual([...SAGE_INBOUND_COMMAND_TYPES].sort(), [
    "approve_tool_call",
    "cancel",
    "deny_tool_call",
    "prompt",
    "set_budget",
  ]);
});

test("protocol version is pinned to v0", () => {
  assert.equal(SAGE_CLI_PROTOCOL_VERSION, "v0");
});

test("bridge remains explicitly unimplemented", () => {
  assert.throws(() => createSageBridge({}), /implementation NYI/);
});
