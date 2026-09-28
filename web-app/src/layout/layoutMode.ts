export type LayoutMode = 'chat' | 'legacy';

export const LAYOUT_STORAGE_KEY = 'graphirm.layout';

export function readLayoutMode(storage: { getItem(k: string): string | null }): LayoutMode {
  return storage.getItem(LAYOUT_STORAGE_KEY) === 'legacy' ? 'legacy' : 'chat';
}

export function writeLayoutMode(
  storage: { setItem(k: string, value: string): void },
  mode: LayoutMode,
): void {
  storage.setItem(LAYOUT_STORAGE_KEY, mode);
}
