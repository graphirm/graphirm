import * as vscode from 'vscode';

export interface Session {
  id: string;
  name: string;
  status: 'idle' | 'running' | 'aborted';
  created_at: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'tool';
  content: string;
  created_at: string;
}

// Raw node as returned by the server — node_type is a tagged union object.
interface RawNode {
  id: string;
  node_type: { type: string; role?: string; content?: string; [k: string]: unknown };
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
}

export interface GraphNode {
  id: string;
  node_type: { type: string; [k: string]: unknown };
  created_at: string;
  metadata: Record<string, unknown>;
}

export interface GraphEdge {
  id: string;
  edge_type: string;
  source: string;
  target: string;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

function serverUrl(): string {
  return vscode.workspace
    .getConfiguration('graphirm')
    .get<string>('serverUrl', 'http://localhost:3000');
}

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${serverUrl()}${path}`;
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${await res.text()}`);
  }
  if (res.status === 204 || res.headers.get('content-length') === '0') {
    return undefined as T;
  }
  return res.json() as Promise<T>;
}

export const ApiClient = {
  listSessions: () => apiFetch<Session[]>('/api/sessions'),

  createSession: (name: string) =>
    apiFetch<Session>('/api/sessions', {
      method: 'POST',
      body: JSON.stringify({ name }),
    }),

  getSession: (id: string) => apiFetch<Session>(`/api/sessions/${id}`),

  getMessages: async (id: string): Promise<Message[]> => {
    const nodes = await apiFetch<RawNode[]>(`/api/sessions/${id}/messages`);
    return nodes
      .filter(n => n.node_type.type === 'Interaction' && n.node_type.role)
      .map(n => ({
        id: n.id,
        role: n.node_type.role as 'user' | 'assistant' | 'tool',
        content: (n.node_type.content as string) ?? '',
        created_at: n.created_at,
      }));
  },

  sendPrompt: (id: string, content: string) =>
    apiFetch<{ node_id: string }>(`/api/sessions/${id}/prompt`, {
      method: 'POST',
      body: JSON.stringify({ content }),
    }),

  abortSession: (id: string) =>
    apiFetch<void>(`/api/sessions/${id}/abort`, { method: 'POST' }),

  getGraph: (id: string) => apiFetch<GraphData>(`/api/graph/${id}`),

  getNode: (sessionId: string, nodeId: string) =>
    apiFetch<RawNode>(`/api/graph/${sessionId}/node/${nodeId}`),

  getSubgraph: (sessionId: string, nodeId: string) =>
    apiFetch<GraphData>(`/api/graph/${sessionId}/subgraph/${nodeId}`),
};
