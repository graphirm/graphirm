import * as vscode from 'vscode';

export interface SseEvent {
  event: string;
  data: unknown;
}

type EventCallback = (event: SseEvent) => void;

export class SseSubscriber implements vscode.Disposable {
  private abortController: AbortController | null = null;
  private sessionId: string | null = null;
  private reconnectDelay = 1000;
  private readonly maxDelay = 30_000;

  constructor(
    private readonly serverUrl: () => string,
    private readonly onEvent: EventCallback
  ) {}

  subscribe(sessionId: string) {
    this.unsubscribe();
    this.sessionId = sessionId;
    this.reconnectDelay = 1000;
    this.connect();
  }

  private async connect() {
    if (!this.sessionId) return;

    this.abortController = new AbortController();
    const url = `${this.serverUrl()}/api/events/${this.sessionId}`;

    try {
      const res = await fetch(url, {
        signal: this.abortController.signal,
        headers: { Accept: 'text/event-stream' },
      });

      if (!res.ok || !res.body) {
        this.scheduleReconnect();
        return;
      }
      // Reset backoff on successful connection
      this.reconnectDelay = 1000;

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        let eventName = '';
        let dataLine = '';

        for (const line of lines) {
          if (line.startsWith('event:')) {
            eventName = line.slice(6).trim();
          } else if (line.startsWith('data:')) {
            dataLine = line.slice(5).trim();
          } else if (line === '' && eventName) {
            try {
              this.onEvent({ event: eventName, data: JSON.parse(dataLine) });
            } catch {
              this.onEvent({ event: eventName, data: dataLine });
            }
            eventName = '';
            dataLine = '';
          }
        }
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name !== 'AbortError') {
        this.scheduleReconnect();
      }
    }
  }

  private scheduleReconnect() {
    if (!this.sessionId) return;
    setTimeout(() => this.connect(), this.reconnectDelay);
    this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxDelay);
  }

  unsubscribe() {
    this.abortController?.abort();
    this.abortController = null;
    this.sessionId = null;
  }

  dispose() {
    this.unsubscribe();
  }
}
