import assert from 'node:assert/strict';
import { test } from 'node:test';
import { replyHints } from './replyHints.ts';

test('all three hints fire in order when every cutoff matches', () => {
  assert.deepEqual(
    replyHints({
      claims_completion: 0.8,
      cites_evidence: 0.39,
      presents_decision: 0.6,
      presents_as_options: 0.69,
    }),
    [
      'Completion claimed without evidence',
      'This reply asks for a decision',
      'Ask for the choices as a list',
    ],
  );
});

test('boundary values that miss a cutoff produce no hint for that rule', () => {
  assert.deepEqual(
    replyHints({
      claims_completion: 0.8,
      cites_evidence: 0.4,
      presents_decision: 0.59,
      presents_as_options: 0.7,
    }),
    [],
  );
  assert.deepEqual(
    replyHints({
      claims_completion: 0.79,
      cites_evidence: 0.1,
    }),
    [],
  );
});

test('a missing score does not fire that hint', () => {
  assert.deepEqual(replyHints(undefined), []);
  assert.deepEqual(replyHints({}), []);
  assert.deepEqual(replyHints({ claims_completion: 0.95 }), []);
  assert.deepEqual(replyHints({ cites_evidence: 0.1 }), []);
  assert.deepEqual(
    replyHints({ claims_completion: 0.9, cites_evidence: 0.2 }),
    ['Completion claimed without evidence'],
  );
  assert.deepEqual(replyHints({ presents_decision: 0.6 }), ['This reply asks for a decision']);
  assert.deepEqual(replyHints({ presents_as_options: 0 }), ['Ask for the choices as a list']);
});
