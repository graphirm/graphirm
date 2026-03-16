/**
 * HTTP client for Graphirm browser UI.
 * Vanilla JS port of graphirm-vscode ApiClient — uses fetch() with relative URLs (same-origin).
 */

async function apiFetch(path, options = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  if (res.status === 204 || res.headers.get('content-length') === '0') {
    return undefined;
  }
  return res.json();
}

export const api = {
  listSessions: () => apiFetch('/api/sessions'),

  createSession: (name) =>
    apiFetch('/api/sessions', {
      method: 'POST',
      body: JSON.stringify({ agent: name }),
    }),

  getSession: (id) => apiFetch(`/api/sessions/${id}`),

  getMessages: async (id) => {
    const nodes = await apiFetch(`/api/sessions/${id}/messages`);
    return (nodes ?? [])
      .filter((n) => n.node_type?.type === 'Interaction' && n.node_type?.role)
      .map((n) => ({
        id: n.id,
        role: n.node_type.role,
        content: n.node_type.content ?? '',
        created_at: n.created_at,
      }));
  },

  sendPrompt: (id, content) =>
    apiFetch(`/api/sessions/${id}/prompt`, {
      method: 'POST',
      body: JSON.stringify({ content }),
    }),

  abortSession: (id) =>
    apiFetch(`/api/sessions/${id}/abort`, { method: 'POST' }),

  getGraph: (id) => apiFetch(`/api/graph/${id}`),

  getNode: (sessionId, nodeId) =>
    apiFetch(`/api/graph/${sessionId}/node/${nodeId}`),

  getSubgraph: (sessionId, nodeId) =>
    apiFetch(`/api/graph/${sessionId}/subgraph/${nodeId}`),
};
