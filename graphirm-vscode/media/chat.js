import { getCurrentSessionId } from './main.js';

let _send;
let _thinking = false;

export function initChat(send) {
  _send = send;

  const sendBtn = document.getElementById('send-btn');
  const abortBtn = document.getElementById('abort-btn');
  const input = document.getElementById('prompt-input');

  sendBtn.addEventListener('click', sendPrompt);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendPrompt();
    }
  });

  abortBtn.addEventListener('click', () => {
    const id = getCurrentSessionId();
    if (id) _send({ type: 'abort', session_id: id });
  });
}

function sendPrompt() {
  const id = getCurrentSessionId();
  const input = document.getElementById('prompt-input');
  const content = input.value.trim();
  if (!content || _thinking) return;

  if (!id) {
    // No session yet — request one, then the extension host will reply with
    // session_created; we re-queue the prompt via a one-shot listener.
    _pendingPrompt = content;
    input.value = '';
    _send({ type: 'request_session_name' });
    return;
  }

  _send({ type: 'send_prompt', session_id: id, content });
  input.value = '';
  appendMessage({ role: 'user', content, id: `optimistic-${Date.now()}` });
}

// Holds a prompt that was typed before a session existed.
let _pendingPrompt = null;

export function flushPendingPrompt(sessionId) {
  if (!_pendingPrompt) return;
  const content = _pendingPrompt;
  _pendingPrompt = null;
  _send({ type: 'send_prompt', session_id: sessionId, content });
  appendMessage({ role: 'user', content, id: `optimistic-${Date.now()}` });
}

export function handleChatMessage(msg) {
  if (msg.type === 'session_loaded' || msg.type === 'refreshed') {
    renderMessages(msg.messages || []);
  } else if (msg.type === 'thinking_start') {
    setThinking(true);
  } else if (msg.type === 'thinking_end') {
    setThinking(false);
  }
}

function setThinking(on) {
  _thinking = on;
  document.getElementById('thinking-bar').hidden = !on;
  document.getElementById('abort-btn').hidden = !on;
  document.getElementById('send-btn').hidden = on;
}

function renderMessages(messages) {
  const container = document.getElementById('messages');
  container.innerHTML = '';
  messages.forEach(m => appendMessage(m, false));
  container.scrollTop = container.scrollHeight;
}

function appendMessage(msg, scroll = true) {
  const container = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = `message ${msg.role}`;
  div.dataset.nodeId = msg.id;

  const label = document.createElement('div');
  label.className = 'role-label';
  label.textContent = msg.role;
  div.appendChild(label);

  const content = document.createElement('div');
  // marked.parse is safe here — content comes from our own server
  content.innerHTML = msg.role === 'tool'
    ? `<code>${escapeHtml(msg.content)}</code>`
    : marked.parse(msg.content || '');
  div.appendChild(content);

  container.appendChild(div);
  if (scroll) container.scrollTop = container.scrollHeight;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ---------------------------------------------------------------------------
// HITL Approval Card
// ---------------------------------------------------------------------------

export function renderApprovalCard({ node_id, tool_name, arguments: args, is_pause, session_id }) {
  const card = document.createElement('div');
  card.className = 'hitl-approval-card';
  card.dataset.nodeId = node_id;

  const argsStr = typeof args === 'string' ? args : JSON.stringify(args, null, 2);
  const title = is_pause
    ? '&#9646;&#9646; Agent paused &mdash; waiting for resume'
    : `&#9888; Agent wants to run: <strong>${escapeHtml(tool_name)}</strong>`;

  // Escape IDs used in onclick attribute strings to prevent attribute injection.
  const eSid = escapeHtml(session_id);
  const eNid = escapeHtml(node_id);

  card.innerHTML = `
    <div class="hitl-header">${title}</div>
    ${!is_pause ? `
    <details class="hitl-args">
      <summary>Arguments</summary>
      <pre>${escapeHtml(argsStr)}</pre>
    </details>
    ` : ''}
    <div class="hitl-actions" id="hitl-actions-${eNid}">
      ${is_pause ? `
      <button class="hitl-btn hitl-resume" onclick="hitlResume('${eSid}', '${eNid}')">&#9654; Resume</button>
      ` : `
      <button class="hitl-btn hitl-approve" onclick="hitlApprove('${eSid}', '${eNid}')">&#10003; Approve</button>
      <button class="hitl-btn hitl-modify" onclick="hitlModify('${eSid}', '${eNid}')">&#9998; Modify</button>
      <button class="hitl-btn hitl-reject" onclick="hitlShowReject('${eNid}')">&#10005; Reject</button>
      `}
    </div>
    <div class="hitl-reject-form" id="hitl-reject-${eNid}" style="display:none">
      <textarea class="hitl-reason" id="hitl-reason-${eNid}" placeholder="Reason for rejection..."></textarea>
      <button class="hitl-btn hitl-reject-confirm" onclick="hitlReject('${eSid}', '${eNid}')">Send rejection</button>
    </div>
    <div class="hitl-modify-form" id="hitl-modify-${eNid}" style="display:none">
      <textarea class="hitl-args-edit" id="hitl-args-edit-${eNid}">${escapeHtml(argsStr)}</textarea>
      <button class="hitl-btn hitl-modify-confirm" onclick="hitlModifyConfirm('${eSid}', '${eNid}')">Run modified</button>
    </div>
  `;

  const container = document.getElementById('messages');
  container.appendChild(card);
  card.scrollIntoView({ behavior: 'smooth' });
}

function postHitlAction(sessionId, nodeId, body) {
  fetch(`/api/graph/${sessionId}/node/${nodeId}/action`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).catch((err) => console.error('[HITL] action request failed:', err));
}

function resolveHitlCard(nodeId, summary) {
  const card = document.querySelector(`.hitl-approval-card[data-node-id="${nodeId}"]`);
  if (card) {
    card.innerHTML = `<div class="hitl-resolved">${escapeHtml(summary)}</div>`;
    card.classList.add('hitl-resolved-card');
  }
}

// Exposed on window so inline onclick handlers work inside the ES module.
window.hitlApprove = function hitlApprove(sessionId, nodeId) {
  postHitlAction(sessionId, nodeId, { action: 'approve' });
  resolveHitlCard(nodeId, '✓ Approved');
};

window.hitlShowReject = function hitlShowReject(nodeId) {
  document.getElementById(`hitl-reject-${nodeId}`).style.display = 'block';
};

window.hitlReject = function hitlReject(sessionId, nodeId) {
  const reason = document.getElementById(`hitl-reason-${nodeId}`).value;
  postHitlAction(sessionId, nodeId, { action: 'reject', reason });
  resolveHitlCard(nodeId, `✕ Rejected: ${reason}`);
};

window.hitlModify = function hitlModify(sessionId, nodeId) {
  document.getElementById(`hitl-modify-${nodeId}`).style.display = 'block';
};

window.hitlModifyConfirm = function hitlModifyConfirm(sessionId, nodeId) {
  const raw = document.getElementById(`hitl-args-edit-${nodeId}`).value;
  let args;
  try {
    args = JSON.parse(raw);
  } catch (_e) {
    alert('Invalid JSON in modified arguments');
    return;
  }
  postHitlAction(sessionId, nodeId, { action: 'modify', modified_args: args });
  resolveHitlCard(nodeId, '✎ Modified and approved');
};

window.hitlResume = function hitlResume(sessionId, nodeId) {
  fetch(`/api/sessions/${sessionId}/resume`, { method: 'POST' })
    .catch((err) => console.error('[HITL] resume request failed:', err));
  resolveHitlCard(nodeId, '▶ Resumed');
};
