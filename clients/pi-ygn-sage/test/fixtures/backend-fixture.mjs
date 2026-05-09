const RUN_ID = "01H00000000000000000000000";

let buffer = Buffer.alloc(0);
let ended = false;
const waiters = [];

process.stdin.on("data", (chunk) => {
  buffer = Buffer.concat([buffer, chunk]);
  pump();
});

process.stdin.on("end", () => {
  ended = true;
  pump();
});

function pump() {
  while (waiters.length > 0) {
    const lf = buffer.indexOf(0x0a);
    if (lf === -1) {
      if (ended) {
        waiters.shift()(null);
        continue;
      }
      return;
    }
    const line = buffer.subarray(0, lf + 1);
    buffer = buffer.subarray(lf + 1);
    waiters.shift()(line);
  }
}

function readLine() {
  const lf = buffer.indexOf(0x0a);
  if (lf !== -1) {
    const line = buffer.subarray(0, lf + 1);
    buffer = buffer.subarray(lf + 1);
    return Promise.resolve(line);
  }
  if (ended) {
    return Promise.resolve(null);
  }
  return new Promise((resolve) => {
    waiters.push(resolve);
  });
}

function frame(eventType, seq, payload = {}, extra = {}) {
  return JSON.stringify({
    protocol_version: "v0",
    event_type: eventType,
    seq,
    run_id: RUN_ID,
    ts_ms: 1,
    payload_schema_version: eventType.startsWith("cli_") ? "cli_v1" : "v1",
    payload,
    ...extra,
  }) + "\n";
}

function writeFrame(eventType, seq, payload = {}, extra = {}) {
  process.stdout.write(frame(eventType, seq, payload, extra));
}

const mode = process.argv[2] ?? "success-after-prompt";

if (mode === "success-after-prompt") {
  const prompt = await readLine();
  writeFrame("cli_started", 0, {
    protocol_version: "v0",
    observedLine: prompt?.toString("utf8") ?? "",
  });
  writeFrame("task_started", 1, null);
  writeFrame("final_result", 2, null);
  writeFrame("oracle_verdict", 3, {});
  writeFrame("run_frame_summary", 4, {});
  writeFrame("cli_complete", 5, {
    exit_code: 0,
    outcome: "success",
    final_seq: 4,
  });
  process.exit(0);
}

if (mode === "tool-approval") {
  await readLine();
  writeFrame("cli_started", 0, { protocol_version: "v0" });
  writeFrame("cli_tool_request", 1, {
    correlation_id: "approval-1",
    tool_name: "apply_patch",
    tool_args_redacted: {},
    node_id: "n1",
    model_id: "fixture-model",
  });
  const approval = await readLine();
  writeFrame("cli_complete", 2, {
    exit_code: 0,
    outcome: "success",
    final_seq: 1,
    observedLine: approval?.toString("utf8") ?? "",
  });
  process.exit(0);
}

if (mode === "cancel-after-prompt") {
  await readLine();
  const cancel = await readLine();
  writeFrame("cli_started", 0, { protocol_version: "v0" });
  writeFrame(
    "failure",
    1,
    null,
    { kind: "cli_cancel", error_type: "cancelled" },
  );
  writeFrame("cli_complete", 2, {
    exit_code: 130,
    outcome: "cancelled",
    final_seq: 1,
    observedLine: cancel?.toString("utf8") ?? "",
  });
  process.exit(130);
}

if (mode === "budget-after-prompt") {
  await readLine();
  const budget = await readLine();
  writeFrame("cli_started", 0, { protocol_version: "v0" });
  writeFrame("budget", 1, null, { kind: "budget_tightened" });
  writeFrame("cli_complete", 2, {
    exit_code: 0,
    outcome: "success",
    final_seq: 1,
    observedLine: budget?.toString("utf8") ?? "",
  });
  process.exit(0);
}

if (mode === "post-terminal") {
  writeFrame("cli_started", 0, { protocol_version: "v0" });
  writeFrame("cli_complete", 1, {
    exit_code: 0,
    outcome: "success",
    final_seq: 0,
  });
  writeFrame("task_started", 2, null);
  process.exit(0);
}

if (mode === "no-complete") {
  writeFrame("cli_started", 0, { protocol_version: "v0" });
  process.exit(0);
}

if (mode === "exit-mismatch") {
  writeFrame("cli_started", 0, { protocol_version: "v0" });
  writeFrame("cli_complete", 1, {
    exit_code: 0,
    outcome: "success",
    final_seq: 0,
  });
  process.exit(7);
}

process.stderr.write(`unknown fixture mode: ${mode}\n`);
process.exit(2);
