import type { GraphData, GraphNode, Message, Session } from '../types/graph';

async function apiFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
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
        return {
          id: n.id,
          role: nt.role,
          content: nt.content ?? '',
          created_at: n.created_at,
        };
      });
  },

  sendPrompt: (id: string, content: string): Promise<void> =>
    apiFetch(`/api/sessions/${id}/prompt`, {
      method: 'POST',
      body: JSON.stringify({ content }),
    }),

  patchKnowledge: (
    nodeId: string,
    patch: { dismissed?: boolean; summary?: string },
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

  steerFromNode: (id: string, content: string, contextRoot: string): Promise<void> =>
    apiFetch(`/api/sessions/${id}/prompt`, {
      method: 'POST',
      body: JSON.stringify({ content, context_root: contextRoot }),
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

  updateTaskStatus: (sessionId: string, nodeId: string, status: string): Promise<void> => {
    console.warn('updateTaskStatus not implemented yet', { sessionId, nodeId, status });
    return Promise.resolve();
  },

  toggleKnowledgePin: (sessionId: string, nodeId: string, pinned: boolean): Promise<void> => {
    console.warn('toggleKnowledgePin not implemented yet', { sessionId, nodeId, pinned });
    return Promise.resolve();
  },

  editKnowledgeSummary: (_sessionId: string, nodeId: string, summary: string): Promise<void> =>
    apiFetch(`/api/knowledge/${nodeId}`, {
      method: 'PATCH',
      body: JSON.stringify({ summary }),
    }),
};
