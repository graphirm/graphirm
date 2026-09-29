import type { ReplyScores } from '../chat/replyHints';
import type { GraphData, GraphNode, Message, Session, ToolCall } from '../types/graph';
import { getApiKey } from './apiKey';

function authHeaders(): Record<string, string> {
  const key = getApiKey();
  return key ? { Authorization: `Bearer ${key}` } : {};
}

async function apiFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  const extra = (options.headers as Record<string, string> | undefined) ?? {};
  const res = await fetch(path, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(),
      ...extra,
    },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  if (res.status === 204 || res.headers.get('content-length') === '0') {
    return undefined as T;
  }
  return res.json() as Promise<T>;
}

function assistantMetaString(
  role: string,
  metadata: Record<string, unknown> | undefined,
  key: string,
): string | undefined {
  if (role !== 'assistant') return undefined;
  const raw = metadata?.[key];
  return typeof raw === 'string' && raw.length > 0 ? raw : undefined;
}

function assistantMetaNumber(
  role: string,
  metadata: Record<string, unknown> | undefined,
  key: string,
): number | undefined {
  if (role !== 'assistant') return undefined;
  const raw = metadata?.[key];
  return typeof raw === 'number' && Number.isFinite(raw) ? raw : undefined;
}

const REPLY_SCORE_KEYS = [
  'move',
  'claims_completion',
  'cites_evidence',
  'presents_decision',
  'presents_as_options',
] as const;

function finiteNumber(raw: unknown): number | undefined {
  return typeof raw === 'number' && Number.isFinite(raw) ? raw : undefined;
}

function assistantReplyJudge(
  role: string,
  metadata: Record<string, unknown> | undefined,
): Message['replyJudge'] | undefined {
  if (role !== 'assistant') return undefined;
  const raw = metadata?.reply_judge;
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return undefined;
  const obj = raw as Record<string, unknown>;
  const version = typeof obj.version === 'string' && obj.version.length > 0 ? obj.version : undefined;
  const latencyMs = finiteNumber(obj.latency_ms);
  const scoresRaw = obj.scores;
  let scores: ReplyScores | undefined;
  if (scoresRaw && typeof scoresRaw === 'object' && !Array.isArray(scoresRaw)) {
    const rec = scoresRaw as Record<string, unknown>;
    const picked: ReplyScores = {};
    for (const key of REPLY_SCORE_KEYS) {
      const value = finiteNumber(rec[key]);
      if (value !== undefined) picked[key] = value;
    }
    if (Object.keys(picked).length > 0) scores = picked;
  }
  if (!version && latencyMs === undefined && !scores) return undefined;
  return {
    ...(version ? { version } : {}),
    ...(scores ? { scores } : {}),
    ...(latencyMs !== undefined ? { latency_ms: latencyMs } : {}),
  };
}

function parseToolCalls(raw: unknown): ToolCall[] | undefined {
  if (!Array.isArray(raw)) return undefined;
  const calls: ToolCall[] = [];
  for (const item of raw) {
    if (!item || typeof item !== 'object') continue;
    const rec = item as Record<string, unknown>;
    if (typeof rec.id !== 'string' || typeof rec.name !== 'string') continue;
    calls.push({
      id: rec.id,
      name: rec.name,
      arguments: rec.arguments ?? {},
    });
  }
  return calls.length > 0 ? calls : undefined;
}

export const api = {
  listSessions: (): Promise<Session[]> =>
    apiFetch('/api/sessions'),

  createSession: (name: string, workspace?: string): Promise<Session> =>
    apiFetch('/api/sessions', {
      method: 'POST',
      body: JSON.stringify({
        agent: name,
        ...(workspace ? { workspace } : {}),
      }),
    }),

  getSession: (id: string): Promise<Session> =>
    apiFetch(`/api/sessions/${id}`),

  renameSession: (id: string, name: string): Promise<Session> =>
    apiFetch(`/api/sessions/${id}`, {
      method: 'PATCH',
      body: JSON.stringify({ name }),
    }),

  getMessages: async (id: string): Promise<Message[]> => {
    const nodes = await apiFetch<GraphNode[]>(`/api/sessions/${id}/messages`);
    return (nodes ?? [])
      .filter((n) => n.node_type.type === 'Interaction' && 'role' in n.node_type)
      .map((n) => {
        const nt = n.node_type as Extract<typeof n.node_type, { type: 'Interaction' }>;
        const toolNameRaw = n.metadata?.tool_name;
        const toolName = typeof toolNameRaw === 'string' ? toolNameRaw : undefined;
        const toolCallIdRaw = n.metadata?.tool_call_id;
        const toolCallId = typeof toolCallIdRaw === 'string' ? toolCallIdRaw : undefined;
        const toolCalls = nt.role === 'assistant' ? parseToolCalls(n.metadata?.tool_calls) : undefined;
        const modelTier = assistantMetaString(nt.role, n.metadata, 'model_tier');
        const routingStrategy = assistantMetaString(nt.role, n.metadata, 'routing_strategy');
        const routingReason = assistantMetaString(nt.role, n.metadata, 'routing_reason');
        const routingConfidence = assistantMetaNumber(nt.role, n.metadata, 'routing_confidence');
        const replyJudge = assistantReplyJudge(nt.role, n.metadata);
        return {
          id: n.id,
          role: nt.role,
          content: nt.content ?? '',
          created_at: n.created_at,
          segmented: Boolean(n.metadata?.segmented),
          ...(toolName ? { toolName } : {}),
          ...(toolCallId ? { toolCallId } : {}),
          ...(toolCalls ? { toolCalls } : {}),
          ...(modelTier ? { modelTier } : {}),
          ...(routingStrategy ? { routingStrategy } : {}),
          ...(routingReason ? { routingReason } : {}),
          ...(routingConfidence !== undefined ? { routingConfidence } : {}),
          ...(replyJudge ? { replyJudge } : {}),
        };
      });
  },

  sendPrompt: (
    id: string,
    content: string,
    options?: {
      context_root?: string;
      steer_context?: { outline_node_id?: string; interaction_id?: string };
    },
  ): Promise<void> =>
    apiFetch(`/api/sessions/${id}/prompt`, {
      method: 'POST',
      body: JSON.stringify({ content, ...options }),
    }),

  patchKnowledge: (
    nodeId: string,
    patch: { dismissed?: boolean; summary?: string; pinned?: boolean },
  ): Promise<void> =>
    apiFetch(`/api/knowledge/${nodeId}`, {
      method: 'PATCH',
      body: JSON.stringify(patch),
    }),

  markInteractionEdited: (nodeId: string, originalContent: string): Promise<void> =>
    apiFetch(`/api/interactions/${nodeId}/edit`, {
      method: 'PATCH',
      body: JSON.stringify({ original_content: originalContent }),
    }),

  postRoutingFeedback: (interactionId: string, verdict: 'wrong' | 'keep'): Promise<void> =>
    apiFetch(`/api/interactions/${interactionId}/routing-feedback`, {
      method: 'POST',
      body: JSON.stringify({ verdict }),
    }),

  steerFromNode: (id: string, content: string, contextRoot: string): Promise<void> =>
    api.sendPrompt(id, content, { context_root: contextRoot }),

  /** Outline rows for an assistant interaction (`outline_item` Content nodes). */
  getOutline: (sessionId: string, interactionId: string): Promise<GraphNode[]> => {
    const q = new URLSearchParams({ interaction_id: interactionId });
    return apiFetch(`/api/sessions/${sessionId}/outline?${q.toString()}`);
  },

  patchGraphNode: (
    sessionId: string,
    nodeId: string,
    patch: { body?: string; metadata?: Record<string, unknown> },
  ): Promise<GraphNode> =>
    apiFetch(`/api/graph/${sessionId}/node/${nodeId}`, {
      method: 'PATCH',
      body: JSON.stringify(patch),
    }),

  createOutlineItem: (
    sessionId: string,
    body: {
      parent_interaction_id: string;
      title: string;
      body?: string;
      kind?: string;
    },
  ): Promise<GraphNode> =>
    apiFetch(`/api/sessions/${sessionId}/outline`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  abortSession: (id: string): Promise<void> =>
    apiFetch(`/api/sessions/${id}/abort`, { method: 'POST' }),

  pauseSession: (id: string): Promise<void> =>
    apiFetch(`/api/sessions/${id}/pause`, { method: 'POST' }),

  resumeSession: (id: string): Promise<void> =>
    apiFetch(`/api/sessions/${id}/resume`, { method: 'POST' }),

  setAutoApprove: (id: string, enabled: boolean): Promise<void> =>
    apiFetch(`/api/sessions/${id}/auto-approve`, {
      method: 'POST',
      body: JSON.stringify({ enabled }),
    }),

  getGraph: (id: string): Promise<GraphData> =>
    apiFetch(`/api/graph/${id}`),

  getNode: (sessionId: string, nodeId: string): Promise<GraphNode> =>
    apiFetch(`/api/graph/${sessionId}/node/${nodeId}`),

  getSubgraph: (sessionId: string, nodeId: string): Promise<GraphData> =>
    apiFetch(`/api/graph/${sessionId}/subgraph/${nodeId}`),

  nodeAction: (
    sessionId: string,
    nodeId: string,
    action: 'approve' | 'reject',
    reason?: string,
    modifiedArgs?: string,
  ): Promise<void> =>
    apiFetch(`/api/graph/${sessionId}/node/${nodeId}/action`, {
      method: 'POST',
      body: JSON.stringify({ action, reason, modified_args: modifiedArgs }),
    }),

  createAnnotation: (
    sessionId: string,
    entity: string,
    entityType: string,
    summary: string,
    options?: { position?: { x: number; y: number }; relatesTo?: string },
  ): Promise<GraphNode> => {
    const body: Record<string, unknown> = {
      entity,
      entity_type: entityType,
      summary,
    };
    if (options?.position) body.position = options.position;
    if (options?.relatesTo) body.relates_to = options.relatesTo;
    return apiFetch(`/api/graph/${sessionId}/annotate`, {
      method: 'POST',
      body: JSON.stringify(body),
    });
  },

  rateTurn: (sessionId: string, turnId: string, rating: number): Promise<void> =>
    apiFetch(`/api/sessions/${sessionId}/turns/${turnId}/rating`, {
      method: 'PATCH',
      body: JSON.stringify({ rating }),
    }),

  updateTaskStatus: (sessionId: string, nodeId: string, status: string): Promise<void> =>
    apiFetch(`/api/graph/${sessionId}/tasks/${nodeId}`, {
      method: 'PATCH',
      body: JSON.stringify({ status }),
    }),

  toggleKnowledgePin: (_sessionId: string, nodeId: string, pinned: boolean): Promise<void> =>
    apiFetch(`/api/knowledge/${nodeId}`, {
      method: 'PATCH',
      body: JSON.stringify({ pinned }),
    }),

  editKnowledgeSummary: (_sessionId: string, nodeId: string, summary: string): Promise<void> =>
    apiFetch(`/api/knowledge/${nodeId}`, {
      method: 'PATCH',
      body: JSON.stringify({ summary }),
    }),

  listPinnedKnowledge: (limit?: number): Promise<GraphNode[]> => {
    const q = limit != null ? `?${new URLSearchParams({ limit: String(limit) })}` : '';
    return apiFetch(`/api/knowledge/pinned${q}`);
  },

  createKnowledge: (body: {
    entity: string;
    entity_type: string;
    summary: string;
    confidence?: number;
    pinned?: boolean;
    session_id?: string;
  }): Promise<GraphNode> =>
    apiFetch('/api/knowledge', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  deleteKnowledge: (nodeId: string): Promise<void> =>
    apiFetch(`/api/knowledge/${nodeId}`, { method: 'DELETE' }),
};
