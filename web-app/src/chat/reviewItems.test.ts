import assert from 'node:assert/strict';
import { test } from 'node:test';
import { buildReviewItems } from './reviewItems.ts';

test('a pending approval is one approval item whose label includes the tool name', () => {
  const items = buildReviewItems({
    pending: { node_id: 'n-approve', tool_name: 'bash' },
    sessions: [],
    tasks: [],
  });
  assert.deepEqual(items, [
    { kind: 'approval', id: 'n-approve', label: 'bash' },
  ]);
});

test('failed and token-capped sessions are failed, paused stays, completed and others drop', () => {
  const items = buildReviewItems({
    pending: null,
    sessions: [
      { id: 's-run', status: 'running', name: 'Running' },
      { id: 's-fail', status: 'failed', name: 'Broke' },
      { id: 's-cap', status: 'token_cap_exceeded' },
      { id: 's-pause', status: 'paused', name: 'Hold' },
      { id: 's-done', status: 'completed', name: 'Done' },
      { id: 's-idle', status: 'idle', name: 'Idle' },
    ],
    tasks: [],
  });
  assert.deepEqual(items, [
    { kind: 'failed', id: 's-fail', label: 'Broke' },
    { kind: 'failed', id: 's-cap', label: 's-cap' },
    { kind: 'paused', id: 's-pause', label: 'Hold' },
  ]);
});

test('pending and running pi tasks stay; completed, failed, and non-pi tasks are omitted', () => {
  const items = buildReviewItems({
    pending: null,
    sessions: [],
    tasks: [
      { id: 't-run', status: 'running', executor: 'pi', title: 'Ship it' },
      { id: 't-done', status: 'completed', executor: 'pi', title: 'Already done' },
      { id: 't-fail', status: 'failed', executor: 'pi' },
      { id: 't-pend', status: 'pending', executor: 'pi', title: 'Queued' },
      { id: 't-other', status: 'running', executor: 'graphirm', title: 'Local' },
    ],
  });
  assert.deepEqual(items, [
    { kind: 'pi', id: 't-run', label: 'Ship it' },
    { kind: 'pi', id: 't-pend', label: 'Queued' },
  ]);
});

test('a pause gate is one paused row labeled with the session name, not the tool', () => {
  const items = buildReviewItems({
    pending: {
      node_id: 'n-pause',
      tool_name: 'pause',
      is_pause: true,
      session_name: 'Hold',
    },
    sessions: [{ id: 's-run', status: 'running', name: 'Hold' }],
    tasks: [],
  });
  assert.deepEqual(items, [
    { kind: 'paused', id: 'n-pause', label: 'Hold' },
  ]);
});

test('a pause gate without a session name is labeled Paused', () => {
  const items = buildReviewItems({
    pending: { node_id: 'n-pause', tool_name: 'pause', is_pause: true, session_name: '' },
    sessions: [],
    tasks: [],
  });
  assert.deepEqual(items, [
    { kind: 'paused', id: 'n-pause', label: 'Paused' },
  ]);
});

test('order is approval, then sessions, then pi tasks', () => {
  const items = buildReviewItems({
    pending: { node_id: 'n1', tool_name: 'write' },
    sessions: [
      { id: 's1', status: 'failed', name: 'Alpha' },
      { id: 's2', status: 'paused' },
    ],
    tasks: [{ id: 'p1', status: 'running', executor: 'pi', title: 'Pi work' }],
  });
  assert.deepEqual(
    items.map((item) => item.kind),
    ['approval', 'failed', 'paused', 'pi'],
  );
  assert.deepEqual(
    items.map((item) => item.id),
    ['n1', 's1', 's2', 'p1'],
  );
  assert.equal(items[0].label.includes('write'), true);
});
