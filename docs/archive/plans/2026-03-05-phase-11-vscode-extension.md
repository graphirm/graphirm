# Phase 11: VS Code Extension Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a VS Code/Cursor extension (`graphirm-vscode`) that provides a two-pane UI — chat panel and live d3-force graph — connecting to a locally running `graphirm serve` server over HTTP and SSE.

**Architecture:** A TypeScript VS Code extension lives at `graphirm-vscode/` inside the monorepo. It registers a webview panel that renders the full UI (HTML/CSS/JS with d3.js). Because VS Code webviews are sandboxed, the extension host acts as an HTTP proxy: the webview sends `postMessage` requests to the extension host, which calls the Graphirm REST API and SSE stream, then forwards responses back to the webview. For SSE, the extension host subscribes once per session and fans out events to the webview. The extension also adds a status bar item showing session state (idle / thinking / failed).

**Tech Stack:** TypeScript, VS Code Extension API (`vscode` npm package), `@types/vscode`, `esbuild` (bundler — no webpack complexity), d3.js v7 (CDN in webview), marked.js (CDN, markdown rendering), `vsce` (packaging), vanilla HTML5/CSS3/ES modules (webview frontend)

**Distribution:** `.vsix` file installable in Cursor (`Extensions: Install from VSIX`) and code-server on Hetzner spokes (`code-server --install-extension graphirm.vsix`). No marketplace publish required to use.

---

## Key Design Decisions (from Phase 10 findings)

1. **Two-pane layout always visible** — session list + chat left, graph right. No tabs.
2. **SSE-first** — extension host subscribes to `/api/sessions/:id/events`. On `agent.done`, triggers parallel refresh of messages + graph.
3. **Persistent loading state** — 15–40s real LLM turns. "Thinking…" must persist until `agent.done` arrives.
4. **Session state badge** — status bar item always shows idle / thinking / failed.
5. **Click-to-inspect nodes** — inline node detail panel inside the graph pane, no navigation.
6. **Prompt + abort are the only primary mutations** — delete goes in a context menu.

---

## Monorepo Structure

```
graphirm/
├── Cargo.toml              ← unchanged (workspace root, ignores graphirm-vscode/)
├── crates/                 ← unchanged
├── graphirm-vscode/        ← NEW: VS Code extension
│   ├── package.json
│   ├── tsconfig.json
│   ├── esbuild.mjs         ← build script
│   ├── src/
│   │   ├── extension.ts    ← activation, command registration
│   │   ├── GraphirmPanel.ts ← webview panel lifecycle + message proxy
│   │   ├── ApiClient.ts    ← HTTP client (extension host side)
│   │   ├── SseSubscriber.ts ← SSE subscription + event forwarding
│   │   └── statusBar.ts    ← status bar item
│   └── media/              ← webview frontend (HTML/CSS/JS)
│       ├── index.html
│       ├── styles.css
│       ├── main.js         ← entry point, message bridge
│       ├── sessions.js     ← session list + creation
│       ├── chat.js         ← message rendering + prompt submit
│       └── graph.js        ← d3-force rendering + node inspect
```

---

## Prerequisites

### Running Graphirm server (Phase 8)

The extension connects to a locally running `graphirm serve`. The user must have it running before activating the extension. The extension reads the server URL from VS Code settings (default: `http://localhost:3000`).

### API endpoints used

```
GET  /api/sessions                    → session list
POST /api/sessions                    → create session
GET  /api/sessions/:id                → session status
GET  /api/sessions/:id/messages       → message list
POST /api/sessions/:id/prompt         → send prompt
POST /api/sessions/:id/abort          → abort running turn
GET  /api/sessions/:id/events         → SSE stream
GET  /api/graph/:id                   → full session graph
GET  /api/graph/:id/node/:node_id     → single node detail
GET  /api/graph/:id/subgraph/:node_id → node + neighbours
```

### SSE events consumed

```
agent.start   → show "thinking" state
agent.done    → refresh messages + graph in parallel
agent.error   → show error state, clear "thinking"
graph.update  → incremental node added to graph
```

---

## Task 1: Extension scaffold

- [x] Complete

**Files:**
- Create: `graphirm-vscode/package.json`
- Create: `graphirm-vscode/tsconfig.json`
- Create: `graphirm-vscode/esbuild.mjs`
- Create: `graphirm-vscode/src/extension.ts`
- Create: `graphirm-vscode/.vscodeignore`

**Step 1: Create the package.json**

```json
{
  "name": "graphirm",
  "displayName": "Graphirm",
  "description": "Graph-native coding agent UI — chat + live graph explorer",
  "version": "0.1.0",
  "engines": { "vscode": "^1.85.0" },
  "categories": ["AI", "Other"],
  "activationEvents": ["onCommand:graphirm.open"],
  "main": "./out/extension.js",
  "contributes": {
    "commands": [
      {
        "command": "graphirm.open",
        "title": "Graphirm: Open Panel"
      }
    ],
    "configuration": {
      "title": "Graphirm",
      "properties": {
        "graphirm.serverUrl": {
          "type": "string",
          "default": "http://localhost:3000",
          "description": "URL of the running graphirm server"
        }
      }
    }
  },
  "scripts": {
    "build": "node esbuild.mjs",
    "watch": "node esbuild.mjs --watch",
    "package": "vsce package --no-dependencies"
  },
  "devDependencies": {
    "@types/vscode": "^1.85.0",
    "@types/node": "^20",
    "typescript": "^5",
    "esbuild": "^0.20",
    "@vscode/vsce": "^2"
  }
}
```

**Step 2: Create tsconfig.json**

```json
{
  "compilerOptions": {
    "module": "commonjs",
    "target": "ES2020",
    "outDir": "out",
    "lib": ["ES2020"],
    "sourceMap": true,
    "rootDir": "src",
    "strict": true
  },
  "include": ["src"]
}
```

**Step 3: Create esbuild.mjs**

```js
import * as esbuild from 'esbuild';

const watch = process.argv.includes('--watch');

const ctx = await esbuild.context({
  entryPoints: ['src/extension.ts'],
  bundle: true,
  outfile: 'out/extension.js',
  external: ['vscode'],
  format: 'cjs',
  platform: 'node',
  sourcemap: true,
});

if (watch) {
  await ctx.watch();
  console.log('watching...');
} else {
  await ctx.rebuild();
  await ctx.dispose();
}
```

**Step 4: Create src/extension.ts**

```typescript
import * as vscode from 'vscode';
import { GraphirmPanel } from './GraphirmPanel';
import { StatusBarManager } from './statusBar';

export function activate(context: vscode.ExtensionContext) {
  const statusBar = new StatusBarManager();
  context.subscriptions.push(statusBar);

  const openCommand = vscode.commands.registerCommand('graphirm.open', () => {
    GraphirmPanel.createOrShow(context, statusBar);
  });

  context.subscriptions.push(openCommand);
}

export function deactivate() {}
```

**Step 5: Create .vscodeignore**

```
.vscode/**
src/**
node_modules/**
*.ts
tsconfig.json
esbuild.mjs
```

**Step 6: Install dependencies and verify build**

```bash
cd graphirm-vscode
npm install
npm run build
```

Expected: `out/extension.js` created, no TypeScript errors.

**Step 7: Commit**

```bash
git add graphirm-vscode/
git commit -m "feat(vscode): extension scaffold — package.json, tsconfig, esbuild, activation"
```

---

## Task 2: Status bar item

- [x] Complete

**Files:**
- Create: `graphirm-vscode/src/statusBar.ts`

**Step 1: Create statusBar.ts**

```typescript
import * as vscode from 'vscode';

export type SessionStatus = 'idle' | 'thinking' | 'failed' | 'disconnected';

export class StatusBarManager implements vscode.Disposable {
  private item: vscode.StatusBarItem;

  constructor() {
    this.item = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Right,
      100
    );
    this.item.command = 'graphirm.open';
    this.setStatus('disconnected');
    this.item.show();
  }

  setStatus(status: SessionStatus, sessionName?: string) {
    const icons: Record<SessionStatus, string> = {
      idle: '$(pass-filled)',
      thinking: '$(loading~spin)',
      failed: '$(error)',
      disconnected: '$(circle-slash)',
    };
    const labels: Record<SessionStatus, string> = {
      idle: 'Graphirm: idle',
      thinking: 'Graphirm: thinking…',
      failed: 'Graphirm: failed',
      disconnected: 'Graphirm: disconnected',
    };
    this.item.text = `${icons[status]} ${sessionName ? `[${sessionName}] ` : ''}${labels[status].split(': ')[1]}`;
    this.item.backgroundColor =
      status === 'failed'
        ? new vscode.ThemeColor('statusBarItem.errorBackground')
        : undefined;
  }

  dispose() {
    this.item.dispose();
  }
}
```

**Step 2: Verify build**

```bash
cd graphirm-vscode && npm run build
```

Expected: no errors.

**Step 3: Commit**

```bash
git add graphirm-vscode/src/statusBar.ts
git commit -m "feat(vscode): status bar item — idle/thinking/failed/disconnected states"
```

---

## Task 3: HTTP API client (extension host)

- [x] Complete

**Files:**
- Create: `graphirm-vscode/src/ApiClient.ts`

The extension host makes all HTTP calls on behalf of the webview (which is sandboxed). This is the single place where the server URL is read from settings.

**Step 1: Create ApiClient.ts**

```typescript
import * as vscode from 'vscode';

export interface Session {
  id: string;
  name: string;
  status: 'idle' | 'running' | 'aborted';
  created_at: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'tool';
  content: string;
  created_at: string;
}

export interface GraphNode {
  id: string;
  node_type: string;
  data: Record<string, unknown>;
  created_at: string;
}

export interface GraphEdge {
  id: string;
  edge_type: string;
  source: string;
  target: string;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

function serverUrl(): string {
  return vscode.workspace
    .getConfiguration('graphirm')
    .get<string>('serverUrl', 'http://localhost:3000');
}

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${serverUrl()}${path}`;
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${await res.text()}`);
  }
  return res.json() as Promise<T>;
}

export const ApiClient = {
  listSessions: () => apiFetch<Session[]>('/api/sessions'),

  createSession: (name: string) =>
    apiFetch<Session>('/api/sessions', {
      method: 'POST',
      body: JSON.stringify({ name }),
    }),

  getSession: (id: string) => apiFetch<Session>(`/api/sessions/${id}`),

  getMessages: (id: string) =>
    apiFetch<Message[]>(`/api/sessions/${id}/messages`),

  sendPrompt: (id: string, content: string) =>
    apiFetch<{ node_id: string }>(`/api/sessions/${id}/prompt`, {
      method: 'POST',
      body: JSON.stringify({ content }),
    }),

  abortSession: (id: string) =>
    apiFetch<void>(`/api/sessions/${id}/abort`, { method: 'POST' }),

  getGraph: (id: string) => apiFetch<GraphData>(`/api/graph/${id}`),

  getNode: (sessionId: string, nodeId: string) =>
    apiFetch<GraphNode>(`/api/graph/${sessionId}/node/${nodeId}`),

  getSubgraph: (sessionId: string, nodeId: string) =>
    apiFetch<GraphData>(`/api/graph/${sessionId}/subgraph/${nodeId}`),
};
```

**Step 2: Verify build**

```bash
cd graphirm-vscode && npm run build
```

**Step 3: Commit**

```bash
git add graphirm-vscode/src/ApiClient.ts
git commit -m "feat(vscode): ApiClient — typed HTTP wrappers for all graphirm REST endpoints"
```

---

## Task 4: SSE subscriber (extension host)

- [x] Complete

**Files:**
- Create: `graphirm-vscode/src/SseSubscriber.ts`

The extension host subscribes to the SSE stream for the active session. It parses events and forwards them to the webview via `panel.webview.postMessage`. It handles reconnection on disconnect.

**Step 1: Create SseSubscriber.ts**

```typescript
import * as vscode from 'vscode';

export interface SseEvent {
  event: string;
  data: unknown;
}

type EventCallback = (event: SseEvent) => void;

export class SseSubscriber implements vscode.Disposable {
  private abortController: AbortController | null = null;
  private sessionId: string | null = null;

  constructor(
    private readonly serverUrl: () => string,
    private readonly onEvent: EventCallback
  ) {}

  subscribe(sessionId: string) {
    this.unsubscribe();
    this.sessionId = sessionId;
    this.connect();
  }

  private async connect() {
    if (!this.sessionId) return;

    this.abortController = new AbortController();
    const url = `${this.serverUrl()}/api/sessions/${this.sessionId}/events`;

    try {
      const res = await fetch(url, {
        signal: this.abortController.signal,
        headers: { Accept: 'text/event-stream' },
      });

      if (!res.ok || !res.body) return;

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
        // Reconnect after 3 seconds on unexpected disconnect
        setTimeout(() => this.connect(), 3000);
      }
    }
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
```

**Step 2: Verify build**

```bash
cd graphirm-vscode && npm run build
```

**Step 3: Commit**

```bash
git add graphirm-vscode/src/SseSubscriber.ts
git commit -m "feat(vscode): SseSubscriber — SSE stream client with auto-reconnect"
```

---

## Task 5: Webview panel + message bridge

- [x] Complete

**Files:**
- Create: `graphirm-vscode/src/GraphirmPanel.ts`
- Create: `graphirm-vscode/media/index.html` (stub — full content in Task 6)

This is the core of the extension. `GraphirmPanel` owns the webview, wires the `ApiClient` to webview messages, and routes SSE events into the webview.

**Step 1: Create GraphirmPanel.ts**

```typescript
import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { ApiClient } from './ApiClient';
import { SseSubscriber } from './SseSubscriber';
import { StatusBarManager } from './statusBar';

export class GraphirmPanel implements vscode.Disposable {
  static currentPanel: GraphirmPanel | undefined;

  private readonly panel: vscode.WebviewPanel;
  private readonly sse: SseSubscriber;
  private disposables: vscode.Disposable[] = [];

  static createOrShow(
    context: vscode.ExtensionContext,
    statusBar: StatusBarManager
  ) {
    if (GraphirmPanel.currentPanel) {
      GraphirmPanel.currentPanel.panel.reveal();
      return;
    }
    const panel = vscode.window.createWebviewPanel(
      'graphirm',
      'Graphirm',
      vscode.ViewColumn.Beside,
      {
        enableScripts: true,
        localResourceRoots: [
          vscode.Uri.file(path.join(context.extensionPath, 'media')),
        ],
        retainContextWhenHidden: true,
      }
    );
    GraphirmPanel.currentPanel = new GraphirmPanel(panel, context, statusBar);
  }

  private constructor(
    panel: vscode.WebviewPanel,
    private readonly context: vscode.ExtensionContext,
    private readonly statusBar: StatusBarManager
  ) {
    this.panel = panel;

    const serverUrl = () =>
      vscode.workspace
        .getConfiguration('graphirm')
        .get<string>('serverUrl', 'http://localhost:3000');

    this.sse = new SseSubscriber(serverUrl, (event) => {
      this.panel.webview.postMessage({ type: 'sse', event });

      if (event.event === 'agent.start') {
        this.statusBar.setStatus('thinking');
      } else if (event.event === 'agent.done') {
        this.statusBar.setStatus('idle');
      } else if (event.event === 'agent.error') {
        this.statusBar.setStatus('failed');
      }
    });

    this.panel.webview.html = this.getHtml();

    this.panel.webview.onDidReceiveMessage(
      async (msg) => this.handleMessage(msg),
      null,
      this.disposables
    );

    this.panel.onDidDispose(() => this.dispose(), null, this.disposables);
  }

  private async handleMessage(msg: { type: string; [k: string]: unknown }) {
    try {
      switch (msg.type) {
        case 'list_sessions': {
          const sessions = await ApiClient.listSessions();
          this.post({ type: 'sessions', sessions });
          this.statusBar.setStatus('idle');
          break;
        }
        case 'create_session': {
          const session = await ApiClient.createSession(msg.name as string);
          this.post({ type: 'session_created', session });
          this.sse.subscribe(session.id);
          this.statusBar.setStatus('idle', session.name);
          break;
        }
        case 'select_session': {
          const id = msg.id as string;
          this.sse.subscribe(id);
          const [messages, graph, session] = await Promise.all([
            ApiClient.getMessages(id),
            ApiClient.getGraph(id),
            ApiClient.getSession(id),
          ]);
          this.post({ type: 'session_loaded', messages, graph, session });
          this.statusBar.setStatus(
            session.status === 'running' ? 'thinking' : 'idle',
            session.name
          );
          break;
        }
        case 'send_prompt': {
          await ApiClient.sendPrompt(msg.session_id as string, msg.content as string);
          break;
        }
        case 'abort': {
          await ApiClient.abortSession(msg.session_id as string);
          this.statusBar.setStatus('idle');
          break;
        }
        case 'get_node': {
          const node = await ApiClient.getNode(
            msg.session_id as string,
            msg.node_id as string
          );
          this.post({ type: 'node_detail', node });
          break;
        }
        case 'get_subgraph': {
          const subgraph = await ApiClient.getSubgraph(
            msg.session_id as string,
            msg.node_id as string
          );
          this.post({ type: 'subgraph', subgraph });
          break;
        }
        case 'refresh': {
          const id = msg.session_id as string;
          const [messages, graph] = await Promise.all([
            ApiClient.getMessages(id),
            ApiClient.getGraph(id),
          ]);
          this.post({ type: 'refreshed', messages, graph });
          break;
        }
      }
    } catch (err) {
      this.post({ type: 'error', message: String(err) });
      this.statusBar.setStatus('failed');
    }
  }

  private post(msg: unknown) {
    this.panel.webview.postMessage(msg);
  }

  private getHtml(): string {
    const mediaPath = path.join(this.context.extensionPath, 'media');
    const htmlPath = path.join(mediaPath, 'index.html');
    let html = fs.readFileSync(htmlPath, 'utf-8');

    // Replace asset paths with webview URIs
    const mediaUri = this.panel.webview.asWebviewUri(
      vscode.Uri.file(mediaPath)
    );
    html = html.replace(/\{\{mediaUri\}\}/g, mediaUri.toString());
    return html;
  }

  dispose() {
    GraphirmPanel.currentPanel = undefined;
    this.sse.dispose();
    this.panel.dispose();
    this.disposables.forEach((d) => d.dispose());
  }
}
```

**Step 2: Create media/index.html stub**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta http-equiv="Content-Security-Policy"
    content="default-src 'none';
             style-src 'unsafe-inline' https://fonts.googleapis.com;
             font-src https://fonts.gstatic.com;
             script-src 'unsafe-inline' https://cdn.jsdelivr.net;
             img-src data:;" />
  <title>Graphirm</title>
  <link rel="stylesheet" href="{{mediaUri}}/styles.css" />
</head>
<body>
  <div id="app">Loading Graphirm…</div>
  <script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/marked@9/marked.min.js"></script>
  <script src="{{mediaUri}}/main.js" type="module"></script>
</body>
</html>
```

**Step 3: Verify build**

```bash
cd graphirm-vscode && npm run build
```

**Step 4: Commit**

```bash
git add graphirm-vscode/src/GraphirmPanel.ts graphirm-vscode/media/index.html
git commit -m "feat(vscode): GraphirmPanel — webview lifecycle, message bridge, SSE routing"
```

---

## Task 6: Two-pane layout (HTML/CSS)

- [x] Complete

**Files:**
- Modify: `graphirm-vscode/media/index.html` (replace stub)
- Create: `graphirm-vscode/media/styles.css`

**Step 1: Replace index.html with full layout**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta http-equiv="Content-Security-Policy"
    content="default-src 'none';
             style-src 'unsafe-inline' https://fonts.googleapis.com;
             font-src https://fonts.gstatic.com;
             script-src 'unsafe-inline' https://cdn.jsdelivr.net;
             img-src data:;" />
  <title>Graphirm</title>
  <link rel="stylesheet" href="{{mediaUri}}/styles.css" />
</head>
<body>
  <header class="header">
    <span class="logo">graphirm</span>
    <div class="session-controls">
      <select id="session-select"><option>— no sessions —</option></select>
      <button id="new-session-btn">+ New Session</button>
    </div>
  </header>

  <main class="main">
    <!-- LEFT: Chat pane -->
    <section class="chat-pane">
      <div id="messages" class="messages"></div>
      <div class="thinking-bar" id="thinking-bar" hidden>
        <span class="thinking-dot"></span> Agent is thinking…
      </div>
      <div class="input-bar">
        <textarea id="prompt-input" placeholder="Type your message…" rows="2"></textarea>
        <div class="input-actions">
          <button id="send-btn">Send</button>
          <button id="abort-btn" class="danger" hidden>Abort</button>
        </div>
      </div>
    </section>

    <!-- RIGHT: Graph pane -->
    <section class="graph-pane">
      <div class="graph-toolbar">
        <span class="graph-title">Graph</span>
        <label>Depth <input type="range" id="depth-slider" min="1" max="5" value="3" /></label>
        <button id="reset-zoom-btn">Reset</button>
      </div>
      <svg id="graph-svg"></svg>
      <!-- Node detail panel, hidden by default -->
      <div id="node-detail" class="node-detail" hidden>
        <div class="node-detail-header">
          <span id="node-detail-type"></span>
          <button id="node-detail-close">✕</button>
        </div>
        <pre id="node-detail-content"></pre>
        <div class="node-actions">
          <button data-action="subgraph">Expand subgraph</button>
        </div>
      </div>
    </section>
  </main>
</body>
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked@9/marked.min.js"></script>
<script src="{{mediaUri}}/main.js" type="module"></script>
</html>
```

**Step 2: Create styles.css**

```css
:root {
  --bg: var(--vscode-editor-background, #1e1e1e);
  --fg: var(--vscode-editor-foreground, #d4d4d4);
  --border: var(--vscode-panel-border, #333);
  --accent: var(--vscode-button-background, #0e639c);
  --accent-fg: var(--vscode-button-foreground, #fff);
  --danger: #c72e2e;
  --node-interaction: #4fc3f7;
  --node-content: #81c784;
  --node-task: #ffb74d;
  --node-knowledge: #ce93d8;
  --node-agent: #ef9a9a;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--fg);
  font-family: var(--vscode-font-family, 'Segoe UI', sans-serif);
  font-size: 13px;
  height: 100vh;
  display: flex;
  flex-direction: column;
}

.header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 6px 12px;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}

.logo { font-weight: 700; font-size: 15px; letter-spacing: -0.3px; }

.session-controls { display: flex; gap: 8px; align-items: center; margin-left: auto; }

select {
  background: var(--vscode-dropdown-background, #3c3c3c);
  color: var(--fg);
  border: 1px solid var(--border);
  padding: 3px 6px;
  border-radius: 3px;
}

button {
  background: var(--accent);
  color: var(--accent-fg);
  border: none;
  padding: 4px 10px;
  border-radius: 3px;
  cursor: pointer;
  font-size: 12px;
}

button:hover { opacity: 0.85; }
button.danger { background: var(--danger); }
button.secondary {
  background: transparent;
  border: 1px solid var(--border);
  color: var(--fg);
}

.main {
  display: flex;
  flex: 1;
  overflow: hidden;
}

/* Chat pane */
.chat-pane {
  width: 45%;
  min-width: 280px;
  display: flex;
  flex-direction: column;
  border-right: 1px solid var(--border);
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.message {
  padding: 8px 10px;
  border-radius: 4px;
  max-width: 100%;
  line-height: 1.5;
}

.message.user {
  background: var(--vscode-inputValidation-infoBorder, #1a3a5c);
  align-self: flex-end;
  max-width: 85%;
}

.message.assistant {
  background: var(--vscode-editor-inactiveSelectionBackground, #2a2a2a);
}

.message.tool {
  background: transparent;
  border-left: 3px solid var(--node-content);
  padding-left: 8px;
  font-family: monospace;
  font-size: 11px;
  color: var(--node-content);
}

.message .role-label {
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  opacity: 0.6;
  margin-bottom: 4px;
}

.thinking-bar {
  padding: 6px 12px;
  background: var(--vscode-editor-inactiveSelectionBackground, #2a2a2a);
  border-top: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--accent);
}

.thinking-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--accent);
  animation: pulse 1s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

.input-bar {
  padding: 8px;
  border-top: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 6px;
}

textarea {
  background: var(--vscode-input-background, #3c3c3c);
  color: var(--fg);
  border: 1px solid var(--border);
  border-radius: 3px;
  padding: 6px 8px;
  resize: none;
  font-family: inherit;
  font-size: 13px;
  width: 100%;
}

.input-actions { display: flex; gap: 6px; justify-content: flex-end; }

/* Graph pane */
.graph-pane {
  flex: 1;
  display: flex;
  flex-direction: column;
  position: relative;
}

.graph-toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 6px 10px;
  border-bottom: 1px solid var(--border);
  font-size: 12px;
}

.graph-title { font-weight: 600; }

#graph-svg {
  flex: 1;
  width: 100%;
}

/* Node detail panel */
.node-detail {
  position: absolute;
  right: 0; top: 36px; bottom: 0;
  width: 300px;
  background: var(--vscode-sideBar-background, #252526);
  border-left: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.node-detail-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 10px;
  border-bottom: 1px solid var(--border);
  font-weight: 600;
  font-size: 12px;
}

.node-detail-header button {
  background: transparent;
  border: none;
  color: var(--fg);
  cursor: pointer;
  padding: 2px 4px;
}

#node-detail-content {
  flex: 1;
  overflow-y: auto;
  padding: 10px;
  font-size: 11px;
  white-space: pre-wrap;
  word-break: break-word;
}

.node-actions {
  padding: 8px;
  border-top: 1px solid var(--border);
  display: flex;
  gap: 6px;
}

.node-actions button { font-size: 11px; padding: 3px 8px; }
```

**Step 3: Verify build**

```bash
cd graphirm-vscode && npm run build
```

**Step 4: Commit**

```bash
git add graphirm-vscode/media/
git commit -m "feat(vscode): two-pane layout — header, chat pane, graph pane, node detail panel"
```

---

## Task 7: Frontend message bridge (main.js)

- [x] Complete

**Files:**
- Create: `graphirm-vscode/media/main.js`

`main.js` is the webview entry point. It sets up the VS Code message bridge (`acquireVsCodeApi()`), imports the other modules, and wires the initial `list_sessions` call.

**Step 1: Create main.js**

```js
import { initSessions, handleSessionsMessage } from './sessions.js';
import { initChat, handleChatMessage } from './chat.js';
import { initGraph, handleGraphMessage } from './graph.js';

const vscode = acquireVsCodeApi();

export function send(msg) {
  vscode.postMessage(msg);
}

// Dispatch messages from extension host to the appropriate module
window.addEventListener('message', ({ data: msg }) => {
  switch (msg.type) {
    case 'sessions':
    case 'session_created':
    case 'session_loaded':
      handleSessionsMessage(msg);
      handleChatMessage(msg);
      handleGraphMessage(msg);
      break;
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
  if (event === 'agent.start') {
    handleChatMessage({ type: 'thinking_start' });
  } else if (event === 'agent.done') {
    const sessionId = getCurrentSessionId();
    if (sessionId) {
      send({ type: 'refresh', session_id: sessionId });
    }
    handleChatMessage({ type: 'thinking_end' });
  } else if (event === 'agent.error') {
    handleChatMessage({ type: 'thinking_end' });
  } else if (event === 'graph.update') {
    // Incremental update — trigger a graph refresh
    const sessionId = getCurrentSessionId();
    if (sessionId) {
      send({ type: 'refresh', session_id: sessionId });
    }
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
```

**Step 2: Commit**

```bash
git add graphirm-vscode/media/main.js
git commit -m "feat(vscode): main.js — message bridge, SSE event dispatch, boot sequence"
```

---

## Task 8: Session list + creation (sessions.js)

- [x] Complete

**Files:**
- Create: `graphirm-vscode/media/sessions.js`

**Step 1: Create sessions.js**

```js
import { setCurrentSessionId } from './main.js';

let _send;
let _sessions = [];

export function initSessions(send) {
  _send = send;

  document.getElementById('new-session-btn').addEventListener('click', () => {
    const name = prompt('Session name:') || `session-${Date.now()}`;
    _send({ type: 'create_session', name });
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

function renderSessionList() {
  const sel = document.getElementById('session-select');
  sel.innerHTML = _sessions.length === 0
    ? '<option value="">— no sessions —</option>'
    : _sessions.map(s =>
        `<option value="${s.id}">[${s.status}] ${s.name}</option>`
      ).join('');
}

function selectSession(id) {
  document.getElementById('session-select').value = id;
  setCurrentSessionId(id);
  _send({ type: 'select_session', id });
}
```

**Step 2: Commit**

```bash
git add graphirm-vscode/media/sessions.js
git commit -m "feat(vscode): sessions.js — session list rendering and selection"
```

---

## Task 9: Chat panel (chat.js)

- [x] Complete

**Files:**
- Create: `graphirm-vscode/media/chat.js`

**Step 1: Create chat.js**

```js
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
  if (!content || !id || _thinking) return;

  _send({ type: 'send_prompt', session_id: id, content });
  input.value = '';
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
```

**Step 2: Commit**

```bash
git add graphirm-vscode/media/chat.js
git commit -m "feat(vscode): chat.js — message rendering, prompt submit, thinking state"
```

---

## Task 10: d3-force graph rendering (graph.js)

- [x] Complete

**Files:**
- Create: `graphirm-vscode/media/graph.js`

**Step 1: Create graph.js**

```js
import { getCurrentSessionId } from './main.js';

let _send;
let _simulation;
let _svg;
let _width = 600;
let _height = 500;

const NODE_COLORS = {
  Interaction: 'var(--node-interaction)',
  Content: 'var(--node-content)',
  Task: 'var(--node-task)',
  Knowledge: 'var(--node-knowledge)',
  Agent: 'var(--node-agent)',
};

export function initGraph(send) {
  _send = send;

  const svgEl = document.getElementById('graph-svg');
  const rect = svgEl.getBoundingClientRect();
  _width = rect.width || 600;
  _height = rect.height || 500;

  _svg = d3.select('#graph-svg');
  const g = _svg.append('g').attr('class', 'graph-root');

  _svg.call(
    d3.zoom().scaleExtent([0.1, 4]).on('zoom', (e) => {
      g.attr('transform', e.transform);
    })
  );

  document.getElementById('reset-zoom-btn').addEventListener('click', () => {
    _svg.transition().call(
      d3.zoom().transform,
      d3.zoomIdentity
    );
  });

  document.getElementById('node-detail-close').addEventListener('click', () => {
    document.getElementById('node-detail').hidden = true;
  });

  document.querySelector('[data-action="subgraph"]').addEventListener('click', () => {
    const detail = document.getElementById('node-detail');
    if (detail.dataset.nodeId) {
      _send({
        type: 'get_subgraph',
        session_id: getCurrentSessionId(),
        node_id: detail.dataset.nodeId,
      });
    }
  });
}

export function handleGraphMessage(msg) {
  if (msg.type === 'session_loaded' || msg.type === 'refreshed') {
    renderGraph(msg.graph || { nodes: [], edges: [] });
  } else if (msg.type === 'subgraph') {
    renderGraph(msg.subgraph);
  } else if (msg.type === 'node_detail') {
    showNodeDetail(msg.node);
  }
}

function renderGraph({ nodes, edges }) {
  const g = _svg.select('.graph-root');
  g.selectAll('*').remove();

  if (_simulation) _simulation.stop();

  // Arrow marker
  _svg.select('defs').remove();
  _svg.append('defs').append('marker')
    .attr('id', 'arrow')
    .attr('viewBox', '0 -5 10 10')
    .attr('refX', 20)
    .attr('markerWidth', 6)
    .attr('markerHeight', 6)
    .attr('orient', 'auto')
    .append('path')
    .attr('d', 'M0,-5L10,0L0,5')
    .attr('fill', '#666');

  const link = g.append('g').selectAll('line')
    .data(edges)
    .join('line')
    .attr('stroke', '#555')
    .attr('stroke-width', 1)
    .attr('marker-end', 'url(#arrow)');

  const linkLabel = g.append('g').selectAll('text')
    .data(edges)
    .join('text')
    .attr('fill', '#888')
    .attr('font-size', 9)
    .text(d => d.edge_type);

  const node = g.append('g').selectAll('circle')
    .data(nodes)
    .join('circle')
    .attr('r', 10)
    .attr('fill', d => NODE_COLORS[d.node_type] || '#999')
    .attr('stroke', '#fff')
    .attr('stroke-width', 0.5)
    .style('cursor', 'pointer')
    .on('click', (_, d) => {
      _send({
        type: 'get_node',
        session_id: getCurrentSessionId(),
        node_id: d.id,
      });
    })
    .call(d3.drag()
      .on('start', (e, d) => {
        if (!e.active) _simulation.alphaTarget(0.3).restart();
        d.fx = d.x; d.fy = d.y;
      })
      .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
      .on('end', (e, d) => {
        if (!e.active) _simulation.alphaTarget(0);
        d.fx = null; d.fy = null;
      })
    );

  const nodeLabel = g.append('g').selectAll('text')
    .data(nodes)
    .join('text')
    .attr('fill', '#ccc')
    .attr('font-size', 9)
    .attr('text-anchor', 'middle')
    .attr('dy', 20)
    .text(d => d.node_type);

  const nodeMap = new Map(nodes.map(n => [n.id, n]));
  const simEdges = edges.map(e => ({
    ...e,
    source: nodeMap.get(e.source) || e.source,
    target: nodeMap.get(e.target) || e.target,
  }));

  _simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(simEdges).id(d => d.id).distance(80))
    .force('charge', d3.forceManyBody().strength(-200))
    .force('center', d3.forceCenter(_width / 2, _height / 2))
    .on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);
      linkLabel
        .attr('x', d => (d.source.x + d.target.x) / 2)
        .attr('y', d => (d.source.y + d.target.y) / 2);
      node.attr('cx', d => d.x).attr('cy', d => d.y);
      nodeLabel.attr('x', d => d.x).attr('y', d => d.y);
    });
}

function showNodeDetail(node) {
  const detail = document.getElementById('node-detail');
  detail.dataset.nodeId = node.id;
  detail.hidden = false;

  document.getElementById('node-detail-type').textContent =
    `${node.node_type} · ${node.id.slice(0, 8)}`;

  document.getElementById('node-detail-content').textContent =
    JSON.stringify(node.data, null, 2);
}
```

**Step 2: Commit**

```bash
git add graphirm-vscode/media/graph.js
git commit -m "feat(vscode): graph.js — d3-force rendering, node click-to-inspect, drag, zoom"
```

---

## Task 11: VSIX packaging and verification

- [x] Complete

**Files:**
- Modify: `graphirm-vscode/package.json` (add `icon`, `publisher` fields for vsce)

**Step 1: Add required vsce fields to package.json**

Add to the existing `package.json`:

```json
{
  "publisher": "graphirm",
  "icon": "icon.png",
  "repository": {
    "type": "git",
    "url": "https://github.com/graphirm/graphirm"
  },
  "license": "MIT"
}
```

**Step 2: Create a placeholder icon** (128×128 PNG — add a real one before publishing)

```bash
cd graphirm-vscode
# Use any 128×128 PNG — placeholder for now
convert -size 128x128 xc:#0e639c -fill white -gravity Center \
  -font DejaVu-Sans -pointsize 24 -draw 'text 0,0 "G"' icon.png 2>/dev/null \
  || echo "ImageMagick not available — add icon.png manually (128×128)"
```

**Step 3: Build and package**

```bash
cd graphirm-vscode
npm run build
npx vsce package --no-dependencies
```

Expected: `graphirm-0.1.0.vsix` created.

**Step 4: Install in Cursor**

```
Cursor → Extensions → ⋯ → Install from VSIX → select graphirm-0.1.0.vsix
```

**Step 5: Install on Hetzner spoke (code-server)**

```bash
code-server --install-extension graphirm-0.1.0.vsix
```

**Step 6: Smoke test**

1. Start `graphirm serve` (with an LLM key configured)
2. Open Cursor command palette → "Graphirm: Open Panel"
3. Verify two-pane panel appears
4. Create a session, send a prompt, observe chat + graph updating on `agent.done`
5. Click a graph node — verify node detail panel appears

**Step 7: Commit**

```bash
git add graphirm-vscode/package.json
git commit -m "feat(vscode): vsix packaging — publisher fields, build verified, install instructions"
```

---

## Progress Tracking

| Task | Status | Description |
|------|--------|-------------|
| 1 | ✅ | Extension scaffold (package.json, tsconfig, esbuild, activation) |
| 2 | ✅ | Status bar item (idle / thinking / failed / disconnected) |
| 3 | ✅ | ApiClient — typed HTTP wrappers |
| 4 | ✅ | SseSubscriber — SSE stream with auto-reconnect |
| 5 | ✅ | GraphirmPanel — webview lifecycle + message bridge |
| 6 | ✅ | Two-pane layout (HTML + CSS) |
| 7 | ✅ | main.js — message bridge + SSE dispatch |
| 8 | ✅ | sessions.js — session list + creation |
| 9 | ✅ | chat.js — message rendering + prompt submit |
| 10 | ✅ | graph.js — d3-force + click-to-inspect |
| 11 | ✅ | VSIX packaging + Cursor + code-server verification |

---

## Phase 12 (deferred): graphirm.ai

The hosted demo + marketing site is deferred until:
- The extension is validated by real daily use
- Infrastructure decisions are made (Coolify vs managed hosting, Stripe tier model, rate limiting strategy)

**graphirm.ai for now:** A static landing page — what it is, install instructions, link to GitHub. Can be deployed to GitHub Pages or Cloudflare Pages in an afternoon with no backend.

**Phase 12 scope (future):** Hosted Graphirm server with auth, rate limiting, Stripe billing, and the same extension frontend pointing to `https://api.graphirm.ai` instead of `localhost:3000`.
