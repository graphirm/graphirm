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
    const suffixes: Record<SessionStatus, string> = {
      idle: 'idle',
      thinking: 'thinking…',
      failed: 'failed',
      disconnected: 'disconnected',
    };
    const session = sessionName ? `[${sessionName}] ` : '';
    this.item.text = `${icons[status]} Graphirm ${session}${suffixes[status]}`;
    this.item.backgroundColor =
      status === 'failed'
        ? new vscode.ThemeColor('statusBarItem.errorBackground')
        : undefined;
  }

  dispose() {
    this.item.dispose();
  }
}
