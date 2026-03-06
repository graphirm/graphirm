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
