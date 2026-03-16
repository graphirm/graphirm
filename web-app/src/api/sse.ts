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
] as const;

export type SseEventType = typeof SSE_EVENT_TYPES[number];

export interface SseEvent {
  event: SseEventType;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  data: any;
}

export class SseClient {
  private _onEvent: (event: SseEvent) => void;
  private _source: EventSource | null = null;

  constructor(onEvent: (event: SseEvent) => void) {
    this._onEvent = onEvent;
  }

  subscribe(sessionId: string): void {
    this.unsubscribe();
    const url = `/api/events/${encodeURIComponent(sessionId)}`;
    this._source = new EventSource(url);

    for (const type of SSE_EVENT_TYPES) {
      this._source.addEventListener(type, (e: MessageEvent) => {
        let parsed: unknown;
        try {
          parsed = e.data ? JSON.parse(e.data as string) : null;
        } catch {
          parsed = e.data;
        }
        this._onEvent({ event: type, data: parsed });
      });
    }

    this._source.onerror = () => {
      // EventSource reconnects automatically; no action needed.
    };
  }

  unsubscribe(): void {
    if (this._source) {
      this._source.close();
      this._source = null;
    }
  }
}
