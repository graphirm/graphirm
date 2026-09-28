import assert from 'node:assert/strict';
import { test } from 'node:test';
import { formatJevChip } from './jevChip.ts';

test('all routing fields missing yields null', () => {
  assert.equal(formatJevChip({}), null);
});

test('tier, strategy, and confidence join with a middle dot', () => {
  assert.equal(
    formatJevChip({
      model_tier: 'smart',
      routing_strategy: 'jev_router',
      routing_confidence: 0.99,
    }),
    'smart · jev_router · p=0.99',
  );
});

test('only tier is the tier string', () => {
  assert.equal(formatJevChip({ model_tier: 'smart' }), 'smart');
});

test('present pieces are joined and missing pieces are omitted', () => {
  assert.equal(formatJevChip({ routing_strategy: 'jev_router' }), 'jev_router');
  assert.equal(
    formatJevChip({ model_tier: 'smart', routing_confidence: 0.99 }),
    'smart · p=0.99',
  );
});
