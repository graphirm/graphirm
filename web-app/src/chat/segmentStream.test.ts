import assert from 'node:assert/strict';
import { test } from 'node:test';
import { parseSegmentPrefix } from './segmentStream.ts';

test('empty buffer has no segments', () => {
  assert.deepEqual(parseSegmentPrefix(''), {
    segments: [],
    plainText: null,
    showRecovery: false,
  });
});

test('plain text is returned as plainText', () => {
  const result = parseSegmentPrefix('Hello');
  assert.equal(result.plainText, 'Hello');
  assert.equal(result.showRecovery, false);
  assert.deepEqual(result.segments, []);
});

test('incomplete segment streams its content', () => {
  const buffer = '{"segments":[{"type":"reasoning","content":"Hi"}';
  const result = parseSegmentPrefix(buffer);
  assert.equal(result.showRecovery, false);
  assert.equal(result.plainText, null);
  assert.deepEqual(result.segments, [
    { type: 'reasoning', content: 'Hi', state: 'streaming' },
  ]);
});

test('complete envelope marks every segment done', () => {
  const result = parseSegmentPrefix(
    '{"segments":[{"type":"reasoning","content":"Hi"},{"type":"answer","content":"Ok"}]}',
  );
  assert.equal(result.showRecovery, false);
  assert.equal(result.plainText, null);
  assert.deepEqual(result.segments, [
    { type: 'reasoning', content: 'Hi', state: 'done' },
    { type: 'answer', content: 'Ok', state: 'done' },
  ]);
});

test('broken brace shows recovery and hides the buffer', () => {
  const buffer = '{not json';
  const result = parseSegmentPrefix(buffer);
  assert.equal(result.showRecovery, true);
  assert.deepEqual(result.segments, []);
  assert.equal(result.plainText, null);
  assert.notEqual(result.plainText, buffer);
});

test('a closed segment stays done while the next object is streaming', () => {
  const result = parseSegmentPrefix(
    '{"segments":[{"type":"reasoning","content":"Hi"},{"type":"answer","content":"O"',
  );
  assert.equal(result.showRecovery, false);
  assert.equal(result.plainText, null);
  assert.deepEqual(result.segments, [
    { type: 'reasoning', content: 'Hi', state: 'done' },
    { type: 'answer', content: 'O', state: 'streaming' },
  ]);
});

test('a buffer that starts with a brace is not returned as plainText', () => {
  const samples = [
    '{',
    '{not json',
    '{"segments":[{"type":"reasoning","content":"Hi"}',
    '{"segments":[{"type":"reasoning","content":"Hi"},{"type":"answer","content":"Ok"}]}',
  ];
  for (const sample of samples) {
    const result = parseSegmentPrefix(sample);
    assert.notEqual(result.plainText, sample);
  }
});
