import assert from 'node:assert/strict';
import { test } from 'node:test';
import { confirmSections } from './confirmBody.ts';

test('bash body is the command string', () => {
  const { body } = confirmSections('bash', { command: 'ls' });
  assert.equal(body, 'ls');
});

test('write body contains the path and content', () => {
  const { body } = confirmSections('write', { path: 'a.py', content: 'print(1)' });
  assert.match(body, /a\.py/);
  assert.match(body, /print\(1\)/);
});

test('edit body contains the path and a short preview', () => {
  const { body } = confirmSections('edit', {
    path: 'a.py',
    old_string: 'a',
    new_string: 'b',
  });
  assert.match(body, /a\.py/);
  assert.match(body, /- a/);
  assert.match(body, /\+ b/);
});

test('delegate_pi body is the task and the note says Pi will not pause', () => {
  const { body, note } = confirmSections('delegate_pi', { task: 'add hello' });
  assert.match(body, /add hello/);
  assert.match(note ?? '', /not pause/);
});

test('grep body is pretty JSON', () => {
  const { body } = confirmSections('grep', { pattern: 'x' });
  assert.equal(body, JSON.stringify({ pattern: 'x' }, null, 2));
});

test('a string argument is the body as-is', () => {
  const { body } = confirmSections('bash', 'raw command text');
  assert.equal(body, 'raw command text');
});
