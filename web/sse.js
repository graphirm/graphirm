/**
 * SSE client for Graphirm browser UI.
 * Uses the native EventSource API — handles reconnection automatically.
 * Replaces the manual fetch + stream parsing from graphirm-vscode SseSubscriber.
 */

const SSE_EVENT_TYPES = [
  'agent_start',
  'agent_end',
  'turn_start',
  'turn_end',
  'message_start',
  'message_delta',
  'message_end',
  'tool_start',
  'tool_end',
  'graph_update',
  'error',
  'heartbeat',
  'awaiting_approval',
];

function parseEventData(raw) {
  if (raw === '' || raw === undefined) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return raw;
  }
}

export class SseClient {
  constructor(onEvent) {
    this._onEvent = onEvent;
    this._source = null;
    this._sessionId = null;
  }

  subscribe(sessionId) {
    this.unsubscribe();
    this._sessionId = sessionId;
    const url = `/api/events/${encodeURIComponent(sessionId)}`;
    this._source = new EventSource(url);

    for (const type of SSE_EVENT_TYPES) {
      this._source.addEventListener(type, (e) => {
        const parsedData = parseEventData(e.data);
        this._onEvent({ event: type, data: parsedData });
      });
    }

    this._source.onerror = () => {};
  }

  unsubscribe() {
    if (this._source) {
      this._source.close();
      this._source = null;
    }
    this._sessionId = null;
  }
}
