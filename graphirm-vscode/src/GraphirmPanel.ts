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
        .get<string>('serverUrl', 'http://localhost:5555');

    this.sse = new SseSubscriber(serverUrl, (event) => {
      this.panel.webview.postMessage({ type: 'sse', event });

      if (event.event === 'agent_start') {
        this.statusBar.setStatus('thinking');
      } else if (event.event === 'agent_end') {
        this.statusBar.setStatus('idle');
      } else if (event.event === 'error') {
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
        case 'request_session_name': {
          const input = await vscode.window.showInputBox({
            prompt: 'Session name',
            placeHolder: `session-${Date.now()}`,
          });
          if (input === undefined) break; // user cancelled
          const name = input.trim() || `session-${Date.now()}`;
          const session = await ApiClient.createSession(name);
          this.post({ type: 'session_created', session });
          this.sse.subscribe(session.id);
          this.statusBar.setStatus('idle', session.name);
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
        default:
          console.warn('GraphirmPanel: unhandled message type', msg.type);
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

    let html: string;
    try {
      html = fs.readFileSync(htmlPath, 'utf-8');
    } catch (err) {
      vscode.window.showErrorMessage(
        `Graphirm: failed to load webview (${htmlPath}): ${String(err)}`
      );
      return '<html><body>Failed to load Graphirm panel. See error notification.</body></html>';
    }

    const mediaUri = this.panel.webview.asWebviewUri(vscode.Uri.file(mediaPath));
    html = html.replace(/\{\{mediaUri\}\}/g, mediaUri.toString());
    html = html.replace(/\{\{cspSource\}\}/g, this.panel.webview.cspSource);
    return html;
  }

  dispose() {
    GraphirmPanel.currentPanel = undefined;
    this.sse.dispose();
    this.panel.dispose();
    this.disposables.forEach((d) => d.dispose());
  }
}
