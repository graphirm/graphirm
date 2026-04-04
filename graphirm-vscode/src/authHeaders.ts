import * as vscode from 'vscode';

/** Headers for `Authorization: Bearer` when `graphirm.apiKey` is set. */
export function graphirmAuthHeaders(): Record<string, string> {
  const key = vscode.workspace.getConfiguration('graphirm').get<string>('apiKey');
  if (!key?.trim()) return {};
  return { Authorization: `Bearer ${key.trim()}` };
}
