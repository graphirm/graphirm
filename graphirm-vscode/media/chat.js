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
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}
