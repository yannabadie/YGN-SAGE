import assert from "node:assert/strict";
import path from "node:path";
import test from "node:test";

import {
  SAGE_CLI_ENVELOPE_EVENT_TYPES,
  SAGE_CLI_PROTOCOL_VERSION,
  SAGE_INBOUND_COMMAND_TYPES,
  SAGE_OUTBOUND_EVENT_TYPES,
  SAGE_RUNTIME_EVENT_TYPES,
  SageBridgeProtocolError,
  SageCliJsonlParser,
  createSageBridge,
  toSageDisplayEvent,
  type SageOutboundEvent,
} from "../src/index.js";

const RUN_ID = "01H00000000000000000000000";

function frame(
  eventType: string,
  seq: number,
  payload: Record<string, unknown> | null = {},
  extra: Record<string, unknown> = {},
): string {
  return (
    JSON.stringify({
      protocol_version: "v0",
      event_type: eventType,
      seq,
      run_id: RUN_ID,
      ts_ms: 1,
      payload_schema_version: eventType.startsWith("cli_") ? "cli_v1" : "v1",
      payload,
      ...extra,
    }) + "\n"
  );
}

function parseAll(text: string): SageOutboundEvent[] {
  const parser = new SageCliJsonlParser();
  const events = parser.feed(Buffer.from(text, "utf8"));
  parser.finish();
  return events;
}

function payloadRecord(event: SageOutboundEvent): Record<string, unknown> {
  assert.equal(typeof event.payload, "object");
  assert.notEqual(event.payload, null);
  assert.equal(Array.isArray(event.payload), false);
  return event.payload as Record<string, unknown>;
}

function hasErrorCode(error: unknown, code: string): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    (error as { code?: unknown }).code === code
  );
}

function validSuccessStream(): string {
  return [
    frame("cli_started", 0, { protocol_version: "v0", task: "hello" }),
    frame("task_started", 1, null),
    frame("routing_decision", 2, null),
    frame("final_result", 3, null),
    frame("oracle_verdict", 4, {}),
    frame("run_frame_summary", 5, {}),
    frame("cli_complete", 6, {
      exit_code: 0,
      outcome: "success",
      final_seq: 5,
    }),
  ].join("");
}

async function collect(bridge: ReturnType<typeof createSageBridge>) {
  const events: SageOutboundEvent[] = [];
  for await (const event of bridge.events) {
    events.push(event);
  }
  const completion = await bridge.completed;
  return { events, completion };
}

function fixturePath(): string {
  return path.join(process.cwd(), "test", "fixtures", "backend-fixture.mjs");
}

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

test("inbound command catalog remains 5 verbs", () => {
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

test("display mapping covers every outbound event type", () => {
  for (const eventType of SAGE_OUTBOUND_EVENT_TYPES) {
    const display = toSageDisplayEvent({
      protocol_version: "v0",
      event_type: eventType,
      seq: 0,
      run_id: RUN_ID,
      ts_ms: 1,
      payload_schema_version: "v1",
      payload: {},
    });
    assert.equal(display.eventType, eventType);
    assert.ok(display.label.length > 0);
    assert.equal(display.raw.event_type, eventType);
  }
});

test("parser accepts a valid success stream", () => {
  const events = parseAll(validSuccessStream());
  assert.deepEqual(
    events.map((event) => event.event_type),
    [
      "cli_started",
      "task_started",
      "routing_decision",
      "final_result",
      "oracle_verdict",
      "run_frame_summary",
      "cli_complete",
    ],
  );
});

test("parser accepts multiple frames in one chunk and split frames", () => {
  const parser = new SageCliJsonlParser();
  const stream = validSuccessStream();
  const splitAt = stream.indexOf("routing_decision") + 6;
  const first = parser.feed(Buffer.from(stream.slice(0, splitAt), "utf8"));
  const second = parser.feed(Buffer.from(stream.slice(splitAt), "utf8"));
  const final = parser.finish();

  assert.equal([...first, ...second].length, 7);
  assert.equal(final.event_type, "cli_complete");
});

test("parser does not split on unicode line separator inside JSON strings", () => {
  const stream = [
    frame("cli_started", 0, { protocol_version: "v0", task: "a\u2028b" }),
    frame("cli_complete", 1, {
      exit_code: 0,
      outcome: "success",
      final_seq: 0,
    }),
  ].join("");
  const events = parseAll(stream);
  assert.equal(payloadRecord(events[0])["task"], "a\u2028b");
});

test("parser accepts the documented cancel sequence", () => {
  const stream = [
    frame("cli_started", 0),
    frame("task_started", 1),
    frame(
      "failure",
      2,
      null,
      { kind: "cli_cancel", error_type: "cancelled" },
    ),
    frame("cli_complete", 3, {
      exit_code: 130,
      outcome: "cancelled",
      final_seq: 2,
    }),
  ].join("");
  const events = parseAll(stream);
  assert.equal(payloadRecord(events.at(-1) as SageOutboundEvent)["outcome"], "cancelled");
});

test("parser rejects fail-closed stream violations", () => {
  const cases: Array<[string, Buffer | string, string]> = [
    [
      "bom",
      Buffer.concat([Buffer.from([0xef, 0xbb, 0xbf]), Buffer.from(frame("cli_started", 0))]),
      "bom_forbidden",
    ],
    ["crlf", frame("cli_started", 0).replace("\n", "\r\n"), "cr_forbidden"],
    ["blank", "\n", "blank_line"],
    ["json", "{not-json}\n", "invalid_json"],
    ["non-object", "[]\n", "non_object_frame"],
    ["utf8", Buffer.from([0xff, 0x0a]), "invalid_utf8"],
    [
      "version",
      frame("cli_started", 0).replace('"protocol_version":"v0"', '"protocol_version":"v1"'),
      "protocol_version_mismatch",
    ],
    [
      "event",
      frame("cli_started", 0).replace('"event_type":"cli_started"', '"event_type":"new_event"'),
      "unknown_event_type",
    ],
    ["first", frame("task_started", 0), "first_frame_not_cli_started"],
    [
      "gap",
      frame("cli_started", 0) + frame("cli_complete", 2, {
        exit_code: 0,
        outcome: "success",
        final_seq: 1,
      }),
      "seq_not_contiguous",
    ],
    [
      "final_seq",
      frame("cli_started", 0) + frame("cli_complete", 1, {
        exit_code: 0,
        outcome: "success",
        final_seq: 99,
      }),
      "final_seq_mismatch",
    ],
    [
      "cancel",
      frame("cli_started", 0) + frame("cli_complete", 1, {
        exit_code: 130,
        outcome: "cancelled",
        final_seq: 0,
      }),
      "invalid_cancel_sequence",
    ],
    [
      "cancel-exit",
      frame("cli_started", 0) +
        frame(
          "failure",
          1,
          null,
          { kind: "cli_cancel", error_type: "cancelled" },
        ) +
        frame("cli_complete", 2, {
          exit_code: 0,
          outcome: "cancelled",
          final_seq: 1,
        }),
      "invalid_cancel_exit_code",
    ],
    [
      "double-cancel",
      frame("cli_started", 0) +
        frame(
          "failure",
          1,
          null,
          { kind: "cli_cancel", error_type: "cancelled" },
        ) +
        frame(
          "failure",
          2,
          null,
          { kind: "cli_cancel", error_type: "cancelled" },
        ) +
        frame("cli_complete", 3, {
          exit_code: 130,
          outcome: "cancelled",
          final_seq: 2,
        }),
      "invalid_cancel_sequence",
    ],
    [
      "ts_ms",
      frame("cli_started", 0).replace('"ts_ms":1', '"ts_ms":-1'),
      "invalid_ts_ms",
    ],
    [
      "payload-schema",
      frame("cli_started", 0).replace('"payload_schema_version":"cli_v1",', ""),
      "invalid_envelope",
    ],
    [
      "cli-payload",
      JSON.stringify({
        protocol_version: "v0",
        event_type: "cli_started",
        seq: 0,
        run_id: RUN_ID,
        ts_ms: 1,
        payload_schema_version: "cli_v1",
      }) + "\n",
      "invalid_payload",
    ],
    [
      "tool-correlation",
      frame("cli_started", 0) +
        frame("cli_tool_request", 1, { tool_name: "apply_patch" }),
      "invalid_tool_correlation",
    ],
    [
      "node-order",
      frame("cli_started", 0) + frame("node_started", 1, null),
      "node_started_before_topology_or_model",
    ],
  ];

  for (const [name, input, code] of cases) {
    const parser = new SageCliJsonlParser();
    assert.throws(
      () => parser.feed(typeof input === "string" ? Buffer.from(input, "utf8") : input),
      (error: unknown) =>
        error instanceof SageBridgeProtocolError &&
        error.code === code,
      name,
    );
  }
});

test("parser rejects BOM on any line including chunk boundary", () => {
  const nonFirst = new SageCliJsonlParser();
  nonFirst.feed(Buffer.from(frame("cli_started", 0), "utf8"));
  assert.throws(
    () =>
      nonFirst.feed(
        Buffer.concat([
          Buffer.from([0xef, 0xbb, 0xbf]),
          Buffer.from(frame("cli_complete", 1, {
            exit_code: 0,
            outcome: "success",
            final_seq: 0,
          })),
        ]),
      ),
    (error: unknown) =>
      error instanceof SageBridgeProtocolError &&
      error.code === "bom_forbidden",
  );

  const split = new SageCliJsonlParser();
  split.feed(Buffer.from([0xef]));
  assert.throws(
    () =>
      split.feed(
        Buffer.concat([
          Buffer.from([0xbb, 0xbf]),
          Buffer.from(frame("cli_started", 0), "utf8"),
        ]),
      ),
    (error: unknown) =>
      error instanceof SageBridgeProtocolError &&
      error.code === "bom_forbidden",
  );
});

test("parser enforces stdout line size bound", () => {
  const parser = new SageCliJsonlParser({ maxLineBytes: 4 });
  assert.throws(
    () => parser.feed(Buffer.from("12345", "utf8")),
    (error: unknown) =>
      error instanceof SageBridgeProtocolError &&
      error.code === "line_too_large",
  );
});

test("parser rejects EOF before cli_complete and unterminated partial lines", () => {
  const missingComplete = new SageCliJsonlParser();
  missingComplete.feed(Buffer.from(frame("cli_started", 0), "utf8"));
  assert.throws(
    () => missingComplete.finish(),
    (error: unknown) =>
      error instanceof SageBridgeProtocolError &&
      error.code === "missing_cli_complete",
  );

  const partial = new SageCliJsonlParser();
  partial.feed(Buffer.from(frame("cli_started", 0).trimEnd(), "utf8"));
  assert.throws(
    () => partial.finish(),
    (error: unknown) =>
      error instanceof SageBridgeProtocolError &&
      error.code === "partial_line",
  );
});

test("bridge completion is eager and does not require event iteration", async () => {
  const ok = createSageBridge({
    command: process.execPath,
    args: [fixturePath(), "success-after-prompt"],
  });
  await ok.send({ command: "prompt", args: { task: "no iteration" } });
  const completion = await ok.completed;
  assert.equal(completion.finalEvent.event_type, "cli_complete");

  const bad = createSageBridge({
    command: process.execPath,
    args: [fixturePath(), "no-complete"],
  });
  await assert.rejects(
    bad.completed,
    (error: unknown) =>
      error instanceof SageBridgeProtocolError &&
      error.code === "missing_cli_complete",
  );
});

test("createSageBridge spawns a fixture backend and writes prompt LF-only", async () => {
  const bridge = createSageBridge({
    command: process.execPath,
    args: [fixturePath(), "success-after-prompt"],
  });

  await bridge.send({
    command: "prompt",
    args: { task: "fixture task", budget_usd: 2, system_hint: 2 },
  });
  const { events, completion } = await collect(bridge);

  assert.equal(completion.exitCode, 0);
  assert.equal(events[0].event_type, "cli_started");
  const observed = String(payloadRecord(events[0])["observedLine"]);
  assert.ok(observed.endsWith("\n"));
  assert.equal(observed.includes("\r"), false);
  assert.equal(JSON.parse(observed).command, "prompt");
});

test("bridge validates tool approval correlations before writing", async () => {
  const bridge = createSageBridge({
    command: process.execPath,
    args: [fixturePath(), "tool-approval"],
  });

  await bridge.send({ command: "prompt", args: { task: "tool task" } });
  const iterator = bridge.events[Symbol.asyncIterator]();
  const started = await iterator.next();
  const toolRequest = await iterator.next();
  assert.equal(started.value.event_type, "cli_started");
  assert.equal(toolRequest.value.event_type, "cli_tool_request");

  await assert.rejects(
    bridge.send({ command: "approve_tool_call", id: "missing" }),
    (error: unknown) =>
      error instanceof SageBridgeProtocolError &&
      error.code === "unknown_tool_correlation",
  );
  await bridge.send({ command: "approve_tool_call", id: "approval-1" });
  await assert.rejects(
    bridge.send({ command: "approve_tool_call", id: "approval-1" }),
    (error: unknown) =>
      error instanceof SageBridgeProtocolError &&
      error.code === "unknown_tool_correlation",
  );

  const remaining: SageOutboundEvent[] = [];
  for (;;) {
    const next = await iterator.next();
    if (next.done) {
      break;
    }
    remaining.push(next.value);
  }
  const completion = await bridge.completed;
  assert.equal(completion.finalEvent.event_type, "cli_complete");
  assert.equal(
    JSON.parse(String(payloadRecord(completion.finalEvent)["observedLine"])).command,
    "approve_tool_call",
  );
  assert.equal(remaining.at(-1)?.event_type, "cli_complete");
});

test("bridge rejects approve/deny duplicate resolution variants", async () => {
  for (const [first, second] of [
    ["approve_tool_call", "approve_tool_call"],
    ["approve_tool_call", "deny_tool_call"],
    ["deny_tool_call", "approve_tool_call"],
  ] as const) {
    const bridge = createSageBridge({
      command: process.execPath,
      args: [fixturePath(), "tool-approval"],
    });
    await bridge.send({ command: "prompt", args: { task: `${first}-${second}` } });
    const iterator = bridge.events[Symbol.asyncIterator]();
    await iterator.next();
    await iterator.next();

    await bridge.send({ command: first, id: "approval-1" });
    await assert.rejects(
      bridge.send({ command: second, id: "approval-1" }),
      (error: unknown) =>
        error instanceof SageBridgeProtocolError &&
        error.code === "unknown_tool_correlation",
    );
    await bridge.completed;
  }
});

test("bridge writes cancel once and accepts the documented cancel completion", async () => {
  const bridge = createSageBridge({
    command: process.execPath,
    args: [fixturePath(), "cancel-after-prompt"],
  });

  await bridge.send({ command: "prompt", args: { task: "cancel task" } });
  await bridge.cancel("user requested");
  await bridge.cancel("duplicate ignored");
  const { events, completion } = await collect(bridge);

  assert.equal(completion.exitCode, 130);
  assert.equal(payloadRecord(events.at(-1) as SageOutboundEvent)["outcome"], "cancelled");
  assert.equal(
    JSON.parse(String(payloadRecord(completion.finalEvent)["observedLine"])).command,
    "cancel",
  );
});

test("bridge rejects invalid or loosening set_budget before stdin write", async () => {
  const bridge = createSageBridge({
    command: process.execPath,
    args: [fixturePath(), "budget-after-prompt"],
  });

  await bridge.send({
    command: "prompt",
    args: { task: "budget task", budget_usd: 5 },
  });
  await assert.rejects(
    bridge.send({ command: "set_budget", args: { budget_usd: 0 } }),
    (error: unknown) =>
      error instanceof SageBridgeProtocolError &&
      error.code === "invalid_budget",
  );
  await assert.rejects(
    bridge.send({ command: "set_budget", args: { budget_usd: 6 } }),
    (error: unknown) =>
      error instanceof SageBridgeProtocolError &&
      error.code === "budget_loosen_rejected",
  );
  await bridge.send({ command: "set_budget", args: { budget_usd: 4 } });
  const { completion } = await collect(bridge);
  assert.equal(
    JSON.parse(String(payloadRecord(completion.finalEvent)["observedLine"])).command,
    "set_budget",
  );
});

test("bridge rejects subprocess lifecycle violations", async () => {
  for (const [mode, code] of [
    ["post-terminal", "frame_after_complete"],
    ["no-complete", "missing_cli_complete"],
    ["exit-mismatch", "exit_code_mismatch"],
  ] as const) {
    const bridge = createSageBridge({
      command: process.execPath,
      args: [fixturePath(), mode],
    });
    await assert.rejects(
      bridge.completed,
      (error: unknown) =>
        (error instanceof SageBridgeProtocolError ||
          (typeof error === "object" &&
            error !== null &&
            error.constructor?.name === "SageBridgeProcessError")) &&
        hasErrorCode(error, code),
      mode,
    );
  }
});

test("bridge rejects future sends after parser failure", async () => {
  const bridge = createSageBridge({
    command: process.execPath,
    args: [fixturePath(), "post-terminal"],
  });

  await assert.rejects(
    bridge.completed,
    (error: unknown) =>
      error instanceof SageBridgeProtocolError &&
      error.code === "frame_after_complete",
  );
  await assert.rejects(
    bridge.send({ command: "prompt", args: { task: "too late" } }),
    (error: unknown) => hasErrorCode(error, "stdin_closed"),
  );
});

test("bridge surfaces spawn failure", async () => {
  const bridge = createSageBridge({
    command: "__definitely_missing_sage_bridge_fixture__",
    args: [],
  });

  await assert.rejects(
    bridge.completed,
    (error: unknown) =>
      typeof error === "object" &&
      error !== null &&
      error.constructor?.name === "SageBridgeProcessError" &&
      hasErrorCode(error, "spawn_error"),
  );
});
