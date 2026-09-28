import assert from 'node:assert/strict';
import { test } from 'node:test';
import { LAYOUT_STORAGE_KEY, readLayoutMode, writeLayoutMode } from './layoutMode.ts';

function memory(initial?: string) {
  const data = new Map<string, string>();
  if (initial !== undefined) data.set(LAYOUT_STORAGE_KEY, initial);
  return {
    getItem(k: string): string | null {
      return data.has(k) ? data.get(k)! : null;
    },
    setItem(k: string, value: string): void {
      data.set(k, value);
    },
  };
}

test('missing key reads as chat', () => {
  assert.equal(readLayoutMode(memory()), 'chat');
});

test("stored 'legacy' reads as legacy", () => {
  assert.equal(readLayoutMode(memory('legacy')), 'legacy');
});

test("stored 'chat' and garbage read as chat", () => {
  assert.equal(readLayoutMode(memory('chat')), 'chat');
  assert.equal(readLayoutMode(memory('whiteboard')), 'chat');
  assert.equal(readLayoutMode(memory('')), 'chat');
});

test('writeLayoutMode persists the mode under the layout key', () => {
  const storage = memory();
  writeLayoutMode(storage, 'legacy');
  assert.equal(storage.getItem(LAYOUT_STORAGE_KEY), 'legacy');
  writeLayoutMode(storage, 'chat');
  assert.equal(storage.getItem(LAYOUT_STORAGE_KEY), 'chat');
});
