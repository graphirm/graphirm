/**
 * Session management for Graphirm browser UI.
 * Adapted from graphirm-vscode/media/sessions.js — uses direct API calls instead of VS Code messaging.
 */

import { api } from './api.js';
import { setCurrentSessionId, getCurrentSessionId, onSessionLoaded } from './main.js';

let _sessions = [];

export function initSessions() {
  document.getElementById('new-session-btn').addEventListener('click', async () => {
    const input = prompt('Session name', `session-${Date.now()}`);
    if (input === null) return;
    const name = input.trim() || `session-${Date.now()}`;
    try {
      const session = await api.createSession(name);
      _sessions.unshift(session);
      renderSessionList();
      await selectSession(session.id);
    } catch (err) {
      console.error('Failed to create session:', err);
    }
  });

  document.getElementById('session-select').addEventListener('change', (e) => {
    const id = e.target.value;
    if (id) selectSession(id);
  });
}

export async function loadSessionList() {
  try {
    _sessions = await api.listSessions();
    renderSessionList();
  } catch (err) {
    console.error('Failed to list sessions:', err);
  }
}

async function selectSession(id) {
  setCurrentSessionId(id);
  document.getElementById('session-select').value = id;
  try {
    const [messages, graph, session] = await Promise.all([
      api.getMessages(id),
      api.getGraph(id),
      api.getSession(id),
    ]);
    if (getCurrentSessionId() !== id) return;
    onSessionLoaded('session_loaded', session, messages, graph);
  } catch (err) {
    console.error('Failed to load session:', err);
  }
}

function renderSessionList() {
  const sel = document.getElementById('session-select');
  if (_sessions.length === 0) {
    sel.innerHTML = '<option value="">— no sessions —</option>';
    return;
  }
  sel.innerHTML = '';
  for (const s of _sessions) {
    const opt = document.createElement('option');
    opt.value = s.id;
    opt.textContent = `[${s.status}] ${s.agent}`;
    sel.appendChild(opt);
  }
}
