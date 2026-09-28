import assert from 'node:assert/strict';
import { test } from 'node:test';
import { buildSteps } from './steps.ts';

test('bash step uses the tool name and argument summary', () => {
  const [row] = buildSteps([
    { toolName: 'bash', callId: 'c1', argsSummary: 'ls -la' },
  ]);
  assert.equal(row.callId, 'c1');
  assert.equal(row.title, 'bash');
  assert.equal(row.detail, 'ls -la');
  assert.equal(row.nestedPi, false);
});

test('delegate_pi without a summary is judged, not gated', () => {
  const [row] = buildSteps([{ toolName: 'delegate_pi', callId: 'c2' }]);
  assert.equal(row.callId, 'c2');
  assert.equal(row.title, 'delegated to Pi');
  assert.equal(row.detail, 'judged, not gated');
  assert.equal(row.nestedPi, true);
});

test('delegate_pi summary tool-call count is included in the detail', () => {
  const [row] = buildSteps([
    {
      toolName: 'delegate_pi',
      callId: 'c3',
      piSummary: 'Pi completed (exit 0, 1.2s)\nTool calls: 2 (0 errors)\n\nResult:\nok',
    },
  ]);
  assert.equal(row.callId, 'c3');
  assert.equal(row.title, 'delegated to Pi');
  assert.equal(row.nestedPi, true);
  assert.match(row.detail, /2 tool calls/);
  assert.match(row.detail, /judged, not gated/);
});

test('other tools use the name and an empty detail when args are missing', () => {
  const [row] = buildSteps([{ toolName: 'read', callId: 'c4' }]);
  assert.equal(row.title, 'read');
  assert.equal(row.detail, '');
  assert.equal(row.nestedPi, false);
});
