/**
 * @ygn-sage/pi-adapter — cycle-13 scaffolding (implementation NYI).
 *
 * This module will spawn `python -m sage.cli run --jsonl` as a subprocess
 * and bridge stdin/stdout per SAGE_CLI_PROTOCOL.md v0 + pi-mono RPC.
 *
 * Cycle-12 ships the type contract only. Cycle-13 implements.
 */

/**
 * Outbound event types per SAGE_CLI_PROTOCOL.md v0.
 *
 * 14 RuntimeEventLog events + 4 CLI-shell envelope events.
 */
export type SageOutboundEventType =
  // CLI-shell envelope events
  | "cli_started"
  | "cli_progress"
  | "cli_tool_request"
  | "cli_complete"
  // RuntimeEventLog v0 events (cycle-7 R6.1c versioned)
  | "task_started"
  | "task_ended"
  | "routing_decision"
  | "topology_selected"
  | "model_assigned"
  | "node_started"
  | "node_completed"
  | "controller_decision"
  | "state_applied"
  | "failure"
  | "budget"
  | "final_result"
  | "oracle_verdict"
  | "run_frame_summary"
  | "bandit_attribution_mismatch";

/**
 * Inbound command types per SAGE_CLI_PROTOCOL.md v0.
 */
export type SageInboundCommandType =
  | "prompt"
  | "approve_tool_call"
  | "deny_tool_call"
  | "cancel"
  | "set_budget";

export interface SageOutboundEvent {
  event_type: SageOutboundEventType;
  payload_schema_version: number;
  seq: number;
  run_id: string;
  payload: Record<string, unknown>;
}

export interface SageInboundCommand {
  command: SageInboundCommandType;
  payload: Record<string, unknown>;
  command_id?: string;
}

/**
 * Cycle-13 implementation entry point.
 * Currently throws — placeholder for the subprocess bridge.
 */
export function createSageBridge(_options: {
  pythonPath?: string;
  cwd?: string;
}): never {
  throw new Error(
    "@ygn-sage/pi-adapter: implementation NYI. " +
      "Cycle-12 ships the type contract; cycle-13 implements the bridge."
  );
}
