/**
 * @ygn-sage/pi-adapter.
 *
 * Strict TypeScript bridge for the SAGE CLI v0 subprocess protocol. This is
 * deliberately a YGN-SAGE pass-through bridge, not a pi-mono UI extension:
 * model selection, topology, tool policy, budget truth, and learning gates
 * remain backend-owned.
 */

import {
  spawn,
  type ChildProcessWithoutNullStreams,
  type SpawnOptionsWithoutStdio,
} from "node:child_process";
import { TextDecoder } from "node:util";

export const SAGE_CLI_PROTOCOL_VERSION = "v0" as const;

export const SAGE_RUNTIME_EVENT_TYPES = [
  "task_started",
  "routing_decision",
  "bandit_attribution_mismatch",
  "topology_selected",
  "model_assigned",
  "node_started",
  "node_completed",
  "controller_decision",
  "state_applied",
  "failure",
  "budget",
  "final_result",
  "oracle_verdict",
  "run_frame_summary",
  "prompt_injection_detected",
] as const;

export const SAGE_CLI_ENVELOPE_EVENT_TYPES = [
  "cli_started",
  "cli_progress",
  "cli_tool_request",
  "cli_complete",
] as const;

export const SAGE_OUTBOUND_EVENT_TYPES = [
  ...SAGE_CLI_ENVELOPE_EVENT_TYPES,
  ...SAGE_RUNTIME_EVENT_TYPES,
] as const;

export type SageRuntimeEventType = typeof SAGE_RUNTIME_EVENT_TYPES[number];
export type SageCliEnvelopeEventType =
  typeof SAGE_CLI_ENVELOPE_EVENT_TYPES[number];
export type SageOutboundEventType = typeof SAGE_OUTBOUND_EVENT_TYPES[number];

const OUTBOUND_EVENT_SET = new Set<string>(SAGE_OUTBOUND_EVENT_TYPES);
const INBOUND_COMMAND_SET = new Set<string>([
  "prompt",
  "approve_tool_call",
  "deny_tool_call",
  "cancel",
  "set_budget",
]);

export const SAGE_INBOUND_COMMAND_TYPES = [
  "prompt",
  "approve_tool_call",
  "deny_tool_call",
  "cancel",
  "set_budget",
] as const;

export type SageInboundCommandType = typeof SAGE_INBOUND_COMMAND_TYPES[number];

export type SagePayloadSchemaVersion =
  | `v${number}`
  | `v${number}_${string}`
  | "cli_v1";

export interface SageOutboundEvent {
  protocol_version: typeof SAGE_CLI_PROTOCOL_VERSION;
  event_type: SageOutboundEventType;
  seq: number;
  run_id: string;
  ts_ms: number;
  payload_schema_version: SagePayloadSchemaVersion;
  payload?: Record<string, unknown> | null | string | number | boolean;

  // RuntimeEventLog failure/state events may carry flat top-level fields.
  [key: string]: unknown;
}

export type SageInboundCommand =
  | {
      command: "prompt";
      args: { task: string; budget_usd?: number; system_hint?: 1 | 2 | 3 };
    }
  | {
      command: "approve_tool_call";
      id: string;
      args?: Record<string, never>;
    }
  | {
      command: "deny_tool_call";
      id: string;
      args?: { reason?: string };
    }
  | {
      command: "cancel";
      args?: { reason?: string };
    }
  | {
      command: "set_budget";
      args: { budget_usd: number };
    };

export type SageDisplaySeverity = "info" | "progress" | "warning" | "error";

export interface SageDisplayEvent {
  eventType: SageOutboundEventType;
  label: string;
  severity: SageDisplaySeverity;
  message: string;
  raw: SageOutboundEvent;
}

export interface SageBridgeCompletion {
  exitCode: number | null;
  signal: NodeJS.Signals | null;
  finalEvent: SageOutboundEvent;
  stderr: string;
}

export interface SageBridge {
  events: AsyncIterable<SageOutboundEvent>;
  completed: Promise<SageBridgeCompletion>;
  send(command: SageInboundCommand): Promise<void>;
  cancel(reason?: string): Promise<void>;
  close(): Promise<void>;
}

export interface SageBridgeOptions {
  command?: string;
  args?: readonly string[];
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  maxLineBytes?: number;
  maxStderrBytes?: number;
  killOnProtocolError?: boolean;
}

export interface SageCliJsonlParserOptions {
  maxLineBytes?: number;
}

export class SageBridgeProtocolError extends Error {
  readonly code: string;
  readonly details?: unknown;

  constructor(code: string, message: string, details?: unknown) {
    super(message);
    this.name = "SageBridgeProtocolError";
    this.code = code;
    this.details = details;
  }
}

export class SageBridgeProcessError extends Error {
  readonly code: string;
  readonly details?: unknown;

  constructor(code: string, message: string, details?: unknown) {
    super(message);
    this.name = "SageBridgeProcessError";
    this.code = code;
    this.details = details;
  }
}

type PendingResolver<T> = {
  resolve: (value: IteratorResult<T>) => void;
  reject: (reason?: unknown) => void;
};

class AsyncEventQueue<T> implements AsyncIterable<T> {
  private readonly values: T[] = [];
  private readonly waiters: PendingResolver<T>[] = [];
  private closed = false;
  private failure: unknown;

  push(value: T): void {
    if (this.closed) {
      return;
    }
    const waiter = this.waiters.shift();
    if (waiter !== undefined) {
      waiter.resolve({ value, done: false });
      return;
    }
    this.values.push(value);
  }

  close(): void {
    if (this.closed) {
      return;
    }
    this.closed = true;
    for (const waiter of this.waiters.splice(0)) {
      waiter.resolve({ value: undefined, done: true });
    }
  }

  fail(error: unknown): void {
    if (this.closed) {
      return;
    }
    this.closed = true;
    this.failure = error;
    for (const waiter of this.waiters.splice(0)) {
      waiter.reject(error);
    }
  }

  [Symbol.asyncIterator](): AsyncIterator<T> {
    return {
      next: () => this.next(),
    };
  }

  private next(): Promise<IteratorResult<T>> {
    const value = this.values.shift();
    if (value !== undefined) {
      return Promise.resolve({ value, done: false });
    }
    if (this.failure !== undefined) {
      return Promise.reject(this.failure);
    }
    if (this.closed) {
      return Promise.resolve({ value: undefined, done: true });
    }
    return new Promise<IteratorResult<T>>((resolve, reject) => {
      this.waiters.push({ resolve, reject });
    });
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isSafeNonNegativeInteger(value: unknown): value is number {
  return (
    typeof value === "number" &&
    Number.isSafeInteger(value) &&
    value >= 0
  );
}

function isPositiveFiniteNumber(value: unknown): value is number {
  return (
    typeof value === "number" &&
    !Number.isNaN(value) &&
    Number.isFinite(value) &&
    value > 0
  );
}

function requiredString(
  frame: Record<string, unknown>,
  field: string,
): string {
  const value = frame[field];
  if (typeof value !== "string" || value.length === 0) {
    throw new SageBridgeProtocolError(
      "invalid_envelope",
      `frame field ${field} must be a non-empty string`,
      frame,
    );
  }
  return value;
}

function requiredRecord(
  frame: Record<string, unknown>,
  field: string,
): Record<string, unknown> {
  const value = frame[field];
  if (!isRecord(value)) {
    throw new SageBridgeProtocolError(
      "invalid_payload",
      `frame field ${field} must be an object`,
      frame,
    );
  }
  return value;
}

export class SageCliJsonlParser {
  private readonly decoder = new TextDecoder("utf-8", { fatal: true });
  private readonly maxLineBytes: number;
  private buffer = Buffer.alloc(0);
  private expectedSeq = 0;
  private runId: string | undefined;
  private started = false;
  private completed = false;
  private lastEvent: SageOutboundEvent | undefined;
  private cancelFailureCount = 0;
  private lineNumber = 0;
  private seenTopologySelected = false;
  private seenModelAssigned = false;

  constructor(options: SageCliJsonlParserOptions = {}) {
    this.maxLineBytes = options.maxLineBytes ?? 1024 * 1024;
  }

  feed(chunk: Uint8Array): SageOutboundEvent[] {
    const incoming = Buffer.from(chunk);
    if (incoming.includes(0x0d)) {
      throw new SageBridgeProtocolError(
        "cr_forbidden",
        "SAGE CLI v0 requires LF-only JSONL; raw CR bytes are forbidden",
      );
    }

    this.buffer = Buffer.concat([this.buffer, incoming]);
    if (this.buffer.length > this.maxLineBytes) {
      throw new SageBridgeProtocolError(
        "line_too_large",
        `JSONL line exceeds ${this.maxLineBytes} bytes`,
      );
    }

    const events: SageOutboundEvent[] = [];
    for (;;) {
      const lf = this.buffer.indexOf(0x0a);
      if (lf === -1) {
        break;
      }
      const line = this.buffer.subarray(0, lf);
      this.buffer = this.buffer.subarray(lf + 1);
      events.push(this.parseLine(line));
      if (this.buffer.length > this.maxLineBytes) {
        throw new SageBridgeProtocolError(
          "line_too_large",
          `JSONL line exceeds ${this.maxLineBytes} bytes`,
        );
      }
    }
    return events;
  }

  finish(): SageOutboundEvent {
    if (this.buffer.length > 0) {
      throw new SageBridgeProtocolError(
        "partial_line",
        "process closed with an unterminated JSONL frame",
      );
    }
    if (!this.completed || this.lastEvent?.event_type !== "cli_complete") {
      throw new SageBridgeProtocolError(
        "missing_cli_complete",
        "process closed before cli_complete",
      );
    }
    return this.lastEvent;
  }

  private parseLine(line: Buffer): SageOutboundEvent {
    if (line.length === 0) {
      throw new SageBridgeProtocolError(
        "blank_line",
        "blank JSONL frames are not allowed",
      );
    }
    if (
      line.length >= 3 &&
      line[0] === 0xef &&
      line[1] === 0xbb &&
      line[2] === 0xbf
    ) {
      throw new SageBridgeProtocolError(
        "bom_forbidden",
        "UTF-8 BOM is forbidden in SAGE CLI JSONL",
      );
    }

    let text: string;
    try {
      text = this.decoder.decode(line);
    } catch (error) {
      throw new SageBridgeProtocolError(
        "invalid_utf8",
        "JSONL frame is not valid UTF-8",
        error,
      );
    }

    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch (error) {
      throw new SageBridgeProtocolError(
        "invalid_json",
        "JSONL frame is not valid JSON",
        error,
      );
    }
    this.lineNumber += 1;
    return this.validateFrame(parsed);
  }

  private validateFrame(value: unknown): SageOutboundEvent {
    if (!isRecord(value)) {
      throw new SageBridgeProtocolError(
        "non_object_frame",
        "JSONL frame must be a JSON object",
        value,
      );
    }
    if (this.completed) {
      throw new SageBridgeProtocolError(
        "frame_after_complete",
        "no frame may appear after cli_complete",
        value,
      );
    }

    const protocolVersion = requiredString(value, "protocol_version");
    if (protocolVersion !== SAGE_CLI_PROTOCOL_VERSION) {
      throw new SageBridgeProtocolError(
        "protocol_version_mismatch",
        `unsupported SAGE CLI protocol_version ${protocolVersion}`,
        value,
      );
    }

    const eventType = requiredString(value, "event_type");
    if (!OUTBOUND_EVENT_SET.has(eventType)) {
      throw new SageBridgeProtocolError(
        "unknown_event_type",
        `unknown SAGE CLI v0 event_type ${eventType}`,
        value,
      );
    }

    const seq = value.seq;
    if (!isSafeNonNegativeInteger(seq)) {
      throw new SageBridgeProtocolError(
        "invalid_seq",
        "frame seq must be a safe non-negative integer",
        value,
      );
    }
    if (seq !== this.expectedSeq) {
      throw new SageBridgeProtocolError(
        "seq_not_contiguous",
        `expected seq ${this.expectedSeq}, received ${seq}`,
        value,
      );
    }

    const runId = requiredString(value, "run_id");
    if (this.runId === undefined) {
      this.runId = runId;
    } else if (this.runId !== runId) {
      throw new SageBridgeProtocolError(
        "run_id_changed",
        "run_id changed mid-stream",
        value,
      );
    }

    const payloadSchemaVersion = requiredString(
      value,
      "payload_schema_version",
    );
    if (!isSafeNonNegativeInteger(value.ts_ms)) {
      throw new SageBridgeProtocolError(
        "invalid_ts_ms",
        "frame ts_ms must be a safe non-negative integer",
        value,
      );
    }
    if (SAGE_CLI_ENVELOPE_EVENT_TYPES.includes(eventType as SageCliEnvelopeEventType)) {
      requiredRecord(value, "payload");
    }
    if ("payload" in value) {
      const payload = value.payload;
      if (
        payload !== null &&
        !isRecord(payload) &&
        typeof payload !== "string" &&
        typeof payload !== "number" &&
        typeof payload !== "boolean"
      ) {
        throw new SageBridgeProtocolError(
          "invalid_payload",
          "payload must be JSON scalar, object, or null",
          value,
        );
      }
    }

    if (!this.started) {
      if (eventType !== "cli_started" || seq !== 0) {
        throw new SageBridgeProtocolError(
          "first_frame_not_cli_started",
          "first frame must be cli_started with seq 0",
          value,
        );
      }
      this.started = true;
    } else if (eventType === "cli_started") {
      throw new SageBridgeProtocolError(
        "duplicate_cli_started",
        "cli_started may only appear as the first frame",
        value,
      );
    }

    if (eventType === "failure") {
      if (value.kind === "cli_cancel" && value.error_type === "cancelled") {
        this.cancelFailureCount += 1;
      }
    }
    if (eventType === "topology_selected") {
      this.seenTopologySelected = true;
    }
    if (eventType === "model_assigned") {
      this.seenModelAssigned = true;
    }
    if (
      eventType === "node_started" &&
      (!this.seenTopologySelected || !this.seenModelAssigned)
    ) {
      throw new SageBridgeProtocolError(
        "node_started_before_topology_or_model",
        "node_started requires preceding topology_selected and model_assigned",
        value,
      );
    }
    if (eventType === "cli_tool_request") {
      const payload = requiredRecord(value, "payload");
      const correlationId = payload.correlation_id;
      if (typeof correlationId !== "string" || correlationId.length === 0) {
        throw new SageBridgeProtocolError(
          "invalid_tool_correlation",
          "cli_tool_request.payload.correlation_id must be a non-empty string",
          value,
        );
      }
    }

    const event = value as SageOutboundEvent;
    this.expectedSeq += 1;

    if (eventType === "cli_complete") {
      const payload = requiredRecord(value, "payload");
      const finalSeq = payload.final_seq;
      if (!isSafeNonNegativeInteger(finalSeq)) {
        throw new SageBridgeProtocolError(
          "invalid_final_seq",
          "cli_complete.payload.final_seq must be a safe non-negative integer",
          value,
        );
      }
      if (finalSeq !== seq - 1) {
        throw new SageBridgeProtocolError(
          "final_seq_mismatch",
          `cli_complete final_seq ${finalSeq} does not point to previous seq ${seq - 1}`,
          value,
        );
      }
      const outcome = payload.outcome;
      if (!["success", "failure", "cancelled"].includes(String(outcome))) {
        throw new SageBridgeProtocolError(
          "invalid_outcome",
          "cli_complete.payload.outcome must be success, failure, or cancelled",
          value,
        );
      }
      if (outcome === "cancelled") {
        if (payload.exit_code !== 130) {
          throw new SageBridgeProtocolError(
            "invalid_cancel_exit_code",
            "cancelled runs must complete with exit_code 130",
            value,
          );
        }
        if (
          this.cancelFailureCount !== 1 ||
          this.lastEvent?.event_type !== "failure" ||
          this.lastEvent.kind !== "cli_cancel" ||
          this.lastEvent.error_type !== "cancelled"
        ) {
          throw new SageBridgeProtocolError(
            "invalid_cancel_sequence",
            "cancelled runs must have exactly one immediately preceding cli_cancel failure",
            value,
          );
        }
      } else if (this.cancelFailureCount !== 0) {
        throw new SageBridgeProtocolError(
          "cancel_failure_without_cancelled_outcome",
          "cli_cancel failure is only valid with cli_complete outcome cancelled",
          value,
        );
      }
      this.completed = true;
    }

    // Keep TypeScript aware that payload_schema_version was string-validated.
    event.payload_schema_version =
      payloadSchemaVersion as SagePayloadSchemaVersion;
    this.lastEvent = event;
    return event;
  }
}

const DISPLAY_LABELS: Record<SageOutboundEventType, string> = {
  cli_started: "CLI started",
  cli_progress: "Progress",
  cli_tool_request: "Tool approval requested",
  cli_complete: "CLI complete",
  task_started: "Task started",
  routing_decision: "Routing decision",
  bandit_attribution_mismatch: "Bandit attribution mismatch",
  topology_selected: "Topology selected",
  model_assigned: "Model assigned",
  node_started: "Node started",
  node_completed: "Node completed",
  controller_decision: "Controller decision",
  state_applied: "State applied",
  failure: "Failure",
  budget: "Budget",
  final_result: "Final result",
  oracle_verdict: "Oracle verdict",
  run_frame_summary: "Run frame summary",
  prompt_injection_detected: "Prompt injection detected",
};

export function toSageDisplayEvent(event: SageOutboundEvent): SageDisplayEvent {
  let severity: SageDisplaySeverity = "info";
  if (event.event_type === "cli_progress") {
    severity = "progress";
  } else if (
    event.event_type === "failure" ||
    event.event_type === "prompt_injection_detected"
  ) {
    severity = "error";
  } else if (event.event_type === "bandit_attribution_mismatch") {
    severity = "warning";
  }
  return {
    eventType: event.event_type,
    label: DISPLAY_LABELS[event.event_type],
    severity,
    message: DISPLAY_LABELS[event.event_type],
    raw: event,
  };
}

class SageSubprocessBridge implements SageBridge {
  readonly events: AsyncIterable<SageOutboundEvent>;
  readonly completed: Promise<SageBridgeCompletion>;

  private readonly child: ChildProcessWithoutNullStreams;
  private readonly parser: SageCliJsonlParser;
  private readonly queue = new AsyncEventQueue<SageOutboundEvent>();
  private readonly killOnProtocolError: boolean;
  private readonly maxStderrBytes: number;
  private readonly stderrChunks: Buffer[] = [];
  private stderrBytes = 0;
  private completedSettled = false;
  private failed = false;
  private resolveCompleted!: (value: SageBridgeCompletion) => void;
  private rejectCompleted!: (reason?: unknown) => void;
  private promptSent = false;
  private cancelSent = false;
  private knownRemainingBudget: number | undefined;
  private readonly pendingToolRequests = new Set<string>();
  private readonly resolvedToolRequests = new Set<string>();
  private finalEvent: SageOutboundEvent | undefined;

  constructor(options: SageBridgeOptions = {}) {
    const command = options.command ?? "sage";
    const args = [...(options.args ?? ["run", "--jsonl"])];
    const spawnOptions: SpawnOptionsWithoutStdio = {
      cwd: options.cwd,
      env: options.env ? { ...process.env, ...options.env } : process.env,
      shell: false,
      stdio: "pipe",
      windowsHide: true,
    };
    this.child = spawn(command, args, spawnOptions);
    this.parser = new SageCliJsonlParser({
      maxLineBytes: options.maxLineBytes,
    });
    this.killOnProtocolError = options.killOnProtocolError ?? true;
    this.maxStderrBytes = options.maxStderrBytes ?? 64 * 1024;
    this.events = this.queue;
    this.completed = new Promise<SageBridgeCompletion>((resolve, reject) => {
      this.resolveCompleted = resolve;
      this.rejectCompleted = reject;
    });
    this.wireChild();
  }

  async send(command: SageInboundCommand): Promise<void> {
    const normalized = this.validateInboundCommand(command);
    if (normalized === null) {
      return;
    }
    await this.writeCommand(normalized.command);
    normalized.commit();
  }

  async cancel(reason?: string): Promise<void> {
    await this.send({ command: "cancel", args: reason ? { reason } : {} });
  }

  async close(): Promise<void> {
    if (!this.completedSettled && !this.child.killed) {
      this.child.stdin.end();
      this.child.kill();
    }
    try {
      await this.completed;
    } catch {
      // close() is cleanup-oriented; protocol errors remain visible via
      // completed/events for callers that need them.
    }
  }

  private wireChild(): void {
    this.child.stdout.on("data", (chunk: Buffer) => {
      if (this.failed) {
        return;
      }
      try {
        for (const event of this.parser.feed(chunk)) {
          this.observeOutboundEvent(event);
          this.queue.push(event);
        }
      } catch (error) {
        this.fail(error);
      }
    });

    this.child.stderr.on("data", (chunk: Buffer) => {
      if (this.stderrBytes >= this.maxStderrBytes) {
        return;
      }
      const remaining = this.maxStderrBytes - this.stderrBytes;
      const kept = chunk.subarray(0, remaining);
      this.stderrChunks.push(kept);
      this.stderrBytes += kept.length;
    });

    this.child.on("error", (error) => {
      this.fail(
        new SageBridgeProcessError(
          "spawn_error",
          `failed to spawn SAGE CLI process: ${error.message}`,
          error,
        ),
      );
    });

    this.child.on("close", (code, signal) => {
      if (this.failed) {
        return;
      }
      try {
        const finalEvent = this.parser.finish();
        this.finalEvent = finalEvent;
        const payload = requiredRecord(
          finalEvent as Record<string, unknown>,
          "payload",
        );
        const frameExitCode = payload.exit_code;
        if (
          typeof frameExitCode === "number" &&
          Number.isFinite(frameExitCode) &&
          code !== null &&
          frameExitCode !== code
        ) {
          throw new SageBridgeProcessError(
            "exit_code_mismatch",
            `process exit code ${code} disagrees with cli_complete exit_code ${frameExitCode}`,
            { code, signal, finalEvent },
          );
        }
        const completion = {
          exitCode: code,
          signal,
          finalEvent,
          stderr: this.stderrText(),
        };
        this.completedSettled = true;
        this.queue.close();
        this.resolveCompleted(completion);
      } catch (error) {
        this.fail(error);
      }
    });
  }

  private observeOutboundEvent(event: SageOutboundEvent): void {
    if (event.event_type === "cli_tool_request" && isRecord(event.payload)) {
      const correlationId = event.payload.correlation_id;
      if (typeof correlationId === "string" && correlationId.length > 0) {
        if (
          this.pendingToolRequests.has(correlationId) ||
          this.resolvedToolRequests.has(correlationId)
        ) {
          throw new SageBridgeProtocolError(
            "duplicate_tool_correlation",
            `duplicate cli_tool_request correlation id ${correlationId}`,
            event,
          );
        }
        this.pendingToolRequests.add(correlationId);
      }
    }
    if (event.event_type === "node_started" && this.pendingToolRequests.size > 0) {
      throw new SageBridgeProtocolError(
        "tool_request_unresolved_before_node_started",
        "pending cli_tool_request must be resolved before node_started",
        event,
      );
    }
  }

  private validateInboundCommand(
    command: SageInboundCommand,
  ): { command: SageInboundCommand; commit: () => void } | null {
    if (!isRecord(command) || !INBOUND_COMMAND_SET.has(String(command.command))) {
      throw new SageBridgeProtocolError(
        "invalid_command",
        "inbound command is not part of SAGE CLI v0",
        command,
      );
    }

    switch (command.command) {
      case "prompt": {
        if (this.promptSent) {
          throw new SageBridgeProtocolError(
            "prompt_already_sent",
            "prompt must be the first and only prompt command",
            command,
          );
        }
        if (!isRecord(command.args)) {
          throw new SageBridgeProtocolError(
            "invalid_prompt",
            "prompt args must be an object",
            command,
          );
        }
        const task = command.args.task;
        if (typeof task !== "string" || task.trim().length === 0) {
          throw new SageBridgeProtocolError(
            "invalid_prompt",
            "prompt args.task must be a non-empty string",
            command,
          );
        }
        const budget = command.args.budget_usd;
        if (budget !== undefined && !isPositiveFiniteNumber(budget)) {
          throw new SageBridgeProtocolError(
            "invalid_budget",
            "prompt budget_usd must be a finite positive number",
            command,
          );
        }
        const systemHint = command.args.system_hint;
        if (
          systemHint !== undefined &&
          ![1, 2, 3].includes(systemHint)
        ) {
          throw new SageBridgeProtocolError(
            "invalid_system_hint",
            "prompt system_hint must be 1, 2, or 3",
            command,
          );
        }
        return {
          command,
          commit: () => {
            this.promptSent = true;
            this.knownRemainingBudget = budget;
          },
        };
      }
      case "approve_tool_call":
      case "deny_tool_call": {
        if (!this.promptSent) {
          throw new SageBridgeProtocolError(
            "command_before_prompt",
            `${command.command} cannot be sent before prompt`,
            command,
          );
        }
        const id = command.id;
        if (typeof id !== "string" || id.length === 0) {
          throw new SageBridgeProtocolError(
            "invalid_tool_correlation",
            `${command.command} requires a non-empty id`,
            command,
          );
        }
        if (!this.pendingToolRequests.has(id) || this.resolvedToolRequests.has(id)) {
          throw new SageBridgeProtocolError(
            "unknown_tool_correlation",
            `no pending cli_tool_request for id ${id}`,
            command,
          );
        }
        return {
          command,
          commit: () => {
            this.pendingToolRequests.delete(id);
            this.resolvedToolRequests.add(id);
          },
        };
      }
      case "cancel": {
        if (!this.promptSent) {
          throw new SageBridgeProtocolError(
            "command_before_prompt",
            "cancel cannot be sent before prompt",
            command,
          );
        }
        if (this.cancelSent) {
          return null;
        }
        return {
          command,
          commit: () => {
            this.cancelSent = true;
          },
        };
      }
      case "set_budget": {
        if (!this.promptSent) {
          throw new SageBridgeProtocolError(
            "command_before_prompt",
            "set_budget cannot be sent before prompt",
            command,
          );
        }
        if (!isRecord(command.args)) {
          throw new SageBridgeProtocolError(
            "invalid_budget",
            "set_budget args must be an object",
            command,
          );
        }
        const budget = command.args.budget_usd;
        if (!isPositiveFiniteNumber(budget)) {
          throw new SageBridgeProtocolError(
            "invalid_budget",
            "set_budget budget_usd must be a finite positive number",
            command,
          );
        }
        if (
          this.knownRemainingBudget !== undefined &&
          budget > this.knownRemainingBudget
        ) {
          throw new SageBridgeProtocolError(
            "budget_loosen_rejected",
            "adapter-known set_budget update would loosen remaining budget",
            command,
          );
        }
        return {
          command,
          commit: () => {
            this.knownRemainingBudget = budget;
          },
        };
      }
    }
  }

  private writeCommand(command: SageInboundCommand): Promise<void> {
    if (this.completedSettled || this.child.stdin.destroyed) {
      return Promise.reject(
        new SageBridgeProcessError(
          "stdin_closed",
          "cannot write command after SAGE CLI stdin closed",
          command,
        ),
      );
    }
    const bytes = Buffer.from(`${JSON.stringify(command)}\n`, "utf8");
    return new Promise<void>((resolve, reject) => {
      this.child.stdin.write(bytes, (error) => {
        if (error) {
          reject(
            new SageBridgeProcessError(
              "stdin_write_failed",
              `failed to write SAGE CLI command: ${error.message}`,
              error,
            ),
          );
        } else {
          resolve();
        }
      });
    });
  }

  private fail(error: unknown): void {
    if (this.failed || this.completedSettled) {
      return;
    }
    this.failed = true;
    this.completedSettled = true;
    this.queue.fail(error);
    this.rejectCompleted(error);
    if (this.killOnProtocolError && !this.child.killed) {
      this.child.stdin.destroy();
      this.child.kill();
    }
  }

  private stderrText(): string {
    return Buffer.concat(this.stderrChunks).toString("utf8");
  }
}

export function createSageBridge(
  options: SageBridgeOptions = {},
): SageBridge {
  return new SageSubprocessBridge(options);
}
