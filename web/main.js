/**
 * Orchestrator for Graphirm browser UI.
 * Wires modules together with direct function calls and manages the SSE connection.
 * Replaces the VS Code extension's message bridge.
 */

import { api } from './api.js';
import { SseClient } from './sse.js';
import { initSessions, loadSessionList } from './sessions.js';
import {
  initChat,
  handleChatMessage,
  flushPendingPrompt,
  renderApprovalCard,
  renderPauseButton,
  syncPauseButtonState,
} from './chat.js';
import { initGraph, renderGraphData } from './graph.js';

let _currentSessionId = null;
let _sse = null;

export function setCurrentSessionId(id) {
  _currentSessionId = id;
}

export function getCurrentSessionId() {
  return _currentSessionId;
}

// Called by sessions.js after selecting or creating a session
export function onSessionLoaded(type, session, messages, graph) {
  setCurrentSessionId(session.id);
  subscribeSse(session.id);

  if (type === 'session_loaded') {
    handleChatMessage({ type: 'session_loaded', messages });
    renderGraphData(graph);
  }

  renderPauseButton(session.id);
}

// Called by chat.js when a session is created from the first message (no session existed)
export function onSessionCreatedFromChat(session) {
  setCurrentSessionId(session.id);
  subscribeSse(session.id);
  renderPauseButton(session.id);
}

function subscribeSse(sessionId) {
  if (_sse) _sse.unsubscribe();
  _sse = new SseClient(handleSseEvent);
  _sse.subscribe(sessionId);
}

async function handleSseEvent({ event, data }) {
  if (event === 'agent_start') {
    handleChatMessage({ type: 'thinking_start' });
  } else if (event === 'agent_end') {
    await refreshCurrentSession();
    handleChatMessage({ type: 'thinking_end' });
  } else if (event === 'error') {
    handleChatMessage({ type: 'thinking_end' });
  } else if (event === 'graph_update') {
    await refreshCurrentSession();
  } else if (event === 'awaiting_approval') {
    renderApprovalCard({ ...data, session_id: _currentSessionId });
    if (data.is_pause) syncPauseButtonState(true);
  }
}

async function refreshCurrentSession() {
  if (!_currentSessionId) return;
  try {
    const [messages, graph] = await Promise.all([
      api.getMessages(_currentSessionId),
      api.getGraph(_currentSessionId),
    ]);
    handleChatMessage({ type: 'refreshed', messages });
    renderGraphData(graph);
  } catch (err) {
    console.error('Failed to refresh session:', err);
  }
}

// Boot
initSessions();
initChat();
initGraph();
loadSessionList();
