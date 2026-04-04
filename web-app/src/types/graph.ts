// TypeScript mirror of Rust NodeType / EdgeType / GraphNode / GraphEdge.
// Keep in sync with crates/graph/src/nodes.rs and crates/graph/src/edges.rs.

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
}

export interface Message {
  id: string;
  role: NodeRole;
  content: string;
  created_at: string;
}

export interface PendingApproval {
  node_id: string;
  tool_name: string;
  arguments: Record<string, unknown> | string;
  is_pause: boolean;
  session_id: string;
}
