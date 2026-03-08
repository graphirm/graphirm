import { initSessions, handleSessionsMessage } from './sessions.js';
import { initChat, handleChatMessage, flushPendingPrompt, renderApprovalCard, renderPauseButton, syncPauseButtonState } from './chat.js';
import { initGraph, handleGraphMessage } from './graph.js';

const vscode = acquireVsCodeApi();

export function send(msg) {
  vscode.postMessage(msg);
}

// Dispatch messages from extension host to the appropriate module
window.addEventListener('message', ({ data: msg }) => {
  switch (msg.type) {
    case 'sessions':
      handleSessionsMessage(msg);
      handleChatMessage(msg);
      handleGraphMessage(msg);
      break;
    case 'session_loaded': {
      handleSessionsMessage(msg);
      handleChatMessage(msg);
      handleGraphMessage(msg);
      const loadedId = getCurrentSessionId();
      if (loadedId) renderPauseButton(loadedId);
      break;
    }
    case 'session_created': {
      handleSessionsMessage(msg);
      handleChatMessage(msg);
      handleGraphMessage(msg);
      flushPendingPrompt(msg.session?.id);
      if (msg.session?.id) renderPauseButton(msg.session.id);
      break;
    }
    case 'refreshed':
      handleChatMessage(msg);
      handleGraphMessage(msg);
      break;
    case 'node_detail':
    case 'subgraph':
      handleGraphMessage(msg);
      break;
    case 'sse':
      handleSseEvent(msg.event);
      break;
    case 'error':
      console.error('Graphirm error:', msg.message);
      break;
  }
});

function handleSseEvent({ event, data }) {
  if (event === 'agent_start') {
    handleChatMessage({ type: 'thinking_start' });
  } else if (event === 'agent_end') {
    const sessionId = getCurrentSessionId();
    if (sessionId) {
      send({ type: 'refresh', session_id: sessionId });
    }
    handleChatMessage({ type: 'thinking_end' });
  } else if (event === 'error') {
    handleChatMessage({ type: 'thinking_end' });
  } else if (event === 'graph_update') {
    const sessionId = getCurrentSessionId();
    if (sessionId) {
      send({ type: 'refresh', session_id: sessionId });
    }
  } else if (event === 'awaiting_approval') {
    const sessionId = getCurrentSessionId();
    renderApprovalCard({ ...data, session_id: sessionId });
    if (data.is_pause) syncPauseButtonState(true);
  }
}

let _currentSessionId = null;
export function setCurrentSessionId(id) { _currentSessionId = id; }
export function getCurrentSessionId() { return _currentSessionId; }

// Boot
initSessions(send);
initChat(send);
initGraph(send);
send({ type: 'list_sessions' });
