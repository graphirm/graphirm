// TypeScript mirror of Rust NodeType / EdgeType / GraphNode / GraphEdge.
// Keep in sync with crates/graph/src/nodes.rs and crates/graph/src/edges.rs.

import type { ReplyScores } from '../chat/replyHints';

export type NodeRole = 'user' | 'assistant' | 'tool' | 'system';
export type TaskStatus = 'pending' | 'running' | 'completed' | 'failed';
export type AgentStatus =
  | 'idle'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'token_cap_exceeded';
export type ContentType = 'code' | 'reasoning' | 'observation' | 'plan' | 'answer' | string;

export type NodeType =
  | { type: 'Interaction'; role: NodeRole; content: string; token_count?: number }
  | { type: 'Agent'; name: string; model: string; system_prompt?: string; status: AgentStatus }
  | { type: 'Content'; content_type: ContentType; path?: string; body: string; language?: string }
  | { type: 'Task'; title: string; description: string; status: TaskStatus; priority?: number }
  | { type: 'Knowledge'; entity: string; entity_type: string; summary: string; confidence: number };

export type EdgeType =
  | 'responds_to'
  | 'spawned_by'
  | 'delegates_to'
  | 'depends_on'
  | 'produces'
  | 'reads'
  | 'modifies'
  | 'summarizes'
  | 'contains'
  | 'follows_up'
  | 'steers'
  | 'relates_to'
  | 'derived_from'
  | 'approved_by'
  | 'rejected_by';

export interface GraphNode {
  id: string;
  node_type: NodeType;
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
}

export interface GraphEdge {
  id: string;
  edge_type: EdgeType;
  source: string;
  target: string;
  weight: number;
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface Session {
  id: string;
  name?: string;
  agent?: string;
  status?: AgentStatus;
  created_at?: string;
  tokens_used?: number;
  max_session_tokens?: number | null;
  /** Server-side auto-approve state for destructive tools (seeds the UI toggle). */
  auto_approve?: boolean;
}

export interface SegmentPart {
  type: ContentType;
  content: string;
  language?: string;
  /** 1-based block number from the numbered-segment contract. */
  n?: number;
  title?: string;
  items?: string[];
  /** In-flight stream only. Persisted segments omit this and render as done. */
  state?: 'done' | 'streaming';
}

/** One entry from an assistant Interaction's `metadata.tool_calls`. */
export interface ToolCall {
  id: string;
  name: string;
  arguments: unknown;
}

export interface Message {
  id: string;
  role: NodeRole;
  content: string;
  created_at: string;
  /** Tool name from Interaction `metadata.tool_name` (tool-role messages). */
  toolName?: string;
  /** `metadata.tool_call_id` on a tool Interaction; matches `ToolCall.id`. */
  toolCallId?: string;
  /** `metadata.tool_calls` on an assistant Interaction. */
  toolCalls?: ToolCall[];
  /** `metadata.model_tier` on an assistant Interaction. */
  modelTier?: string;
  /** `metadata.routing_strategy` on an assistant Interaction. */
  routingStrategy?: string;
  /** `metadata.routing_confidence` on an assistant Interaction. */
  routingConfidence?: number;
  /** `metadata.routing_reason` on an assistant Interaction. */
  routingReason?: string;
  /** `metadata.reply_judge` on an assistant Interaction. */
  replyJudge?: {
    version?: string;
    scores?: ReplyScores;
    latency_ms?: number;
  };
  /** True when structured segments were persisted (`metadata.segmented`). */
  segmented?: boolean;
  /** Populated from graph Contains children when `segmented` (see `segmentPartsForInteraction`). */
  segments?: SegmentPart[];
}

export interface PendingApproval {
  node_id: string;
  tool_name: string;
  arguments: Record<string, unknown> | string;
  is_pause: boolean;
  session_id: string;
  /** Present when the destructive-tool judge scored this call. */
  hitl_judge?: { p_irreversible?: number };
}
