import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  applyLiveToolStart,
  closeLiveDirectors,
  dropCoveredLiveDirectors,
  toLiveStepRows,
} from './liveSteps.ts';

test('a start while delegate_pi is the latest director nests under that row', () => {
  let directors = applyLiveToolStart([], { callId: 'd1', toolName: 'delegate_pi' });
  directors = applyLiveToolStart(directors, { callId: 'p1', toolName: 'read' });
  directors = applyLiveToolStart(directors, { callId: 'p2', toolName: 'bash' });

  assert.equal(directors.length, 1);
  assert.equal(directors[0].callId, 'd1');
  assert.equal(directors[0].toolName, 'delegate_pi');
  assert.deepEqual(directors[0].nested, ['read', 'bash']);

  const rows = toLiveStepRows(directors);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].callId, 'd1');
  assert.equal(rows[0].title, 'delegated to Pi');
  assert.equal(rows[0].detail, 'judged, not gated');
  assert.deepEqual(rows[0].nestedNames, ['read', 'bash']);
});

test('a director start stays a peer when the latest director is not delegate_pi', () => {
  let directors = applyLiveToolStart([], { callId: 'b1', toolName: 'bash' });
  directors = applyLiveToolStart(directors, { callId: 'd1', toolName: 'delegate_pi' });
  directors = applyLiveToolStart(directors, { callId: 'p1', toolName: 'read' });

  assert.equal(directors.length, 2);
  assert.equal(directors[0].toolName, 'bash');
  assert.deepEqual(directors[0].nested, []);
  assert.equal(directors[1].callId, 'd1');
  assert.deepEqual(directors[1].nested, ['read']);
});

test('a start after directors close is a peer, and no error is stored', () => {
  let directors = applyLiveToolStart([], { callId: 'd1', toolName: 'delegate_pi' });
  directors = applyLiveToolStart(directors, { callId: 'p1', toolName: 'read' });
  directors = closeLiveDirectors(directors);
  directors = applyLiveToolStart(directors, { callId: 'w1', toolName: 'write' });

  assert.equal(directors.length, 2);
  assert.equal(directors[0].open, false);
  assert.deepEqual(directors[0].nested, ['read']);
  assert.equal('isError' in directors[0], false);
  assert.equal(directors[1].callId, 'w1');
  assert.equal(directors[1].toolName, 'write');
  assert.equal(directors[1].open, true);
  assert.deepEqual(directors[1].nested, []);
});

test('covered call ids drop and an uncovered director stays', () => {
  const left = dropCoveredLiveDirectors(
    [
      { callId: 'd1', toolName: 'delegate_pi', nested: ['read'] },
      { callId: 'b1', toolName: 'bash', nested: [] },
    ],
    new Set(['d1']),
  );
  assert.deepEqual(left, [{ callId: 'b1', toolName: 'bash', nested: [] }]);
});
