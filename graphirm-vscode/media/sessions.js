import { setCurrentSessionId } from './main.js';

let _send;
let _sessions = [];

export function initSessions(send) {
  _send = send;

  document.getElementById('new-session-btn').addEventListener('click', () => {
    // prompt() is blocked in VS Code webviews — delegate to extension host
    _send({ type: 'request_session_name' });
  });

  document.getElementById('session-select').addEventListener('change', (e) => {
    const id = e.target.value;
    if (id) {
      setCurrentSessionId(id);
      _send({ type: 'select_session', id });
    }
  });
}

export function handleSessionsMessage(msg) {
  if (msg.type === 'sessions') {
    _sessions = msg.sessions;
    renderSessionList();
  } else if (msg.type === 'session_created') {
    _sessions.unshift(msg.session);
    renderSessionList();
    selectSession(msg.session.id);
  } else if (msg.type === 'session_loaded') {
    // handled by chat.js and graph.js
  }
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
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

function selectSession(id) {
  document.getElementById('session-select').value = id;
  setCurrentSessionId(id);
  _send({ type: 'select_session', id });
}
