/**
 * @ygn-sage/pi-adapter - cycle-13 contract scaffolding (bridge NYI).
 *
 * This package currently exports the SAGE_CLI_PROTOCOL.md v0 type/catalog
 * contract only. The subprocess bridge is intentionally not shipped yet.
 */

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
  payload?: Record<string, unknown>;

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

/**
 * Future implementation entry point.
 * Currently throws - the subprocess bridge is not shipped.
 */
export function createSageBridge(_options: { cwd?: string }): never {
  throw new Error(
    "@ygn-sage/pi-adapter: implementation NYI. " +
      "This package currently exports the v0 type/catalog contract only; " +
      "the subprocess bridge is not shipped."
  );
}
