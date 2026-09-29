import assert from 'node:assert/strict';
import { test } from 'node:test';
import { planStepsFromTasks, runSteerText, type PlanStep } from './planCard.ts';

test('executor pi stays pi; missing or anything else is graphirm', () => {
  const steps = planStepsFromTasks([
    { id: 'pi', metadata: { executor: 'pi' } },
    { id: 'missing' },
    { id: 'other', metadata: { executor: 'graphirm' } },
    { id: 'upper', metadata: { executor: 'PI' } },
    { id: 'num', metadata: { executor: 1 } },
  ]);
  assert.deepEqual(
    steps.map((step) => step.executor),
    ['pi', 'graphirm', 'graphirm', 'graphirm', 'graphirm'],
  );
});

test('risky is true only when metadata risky is true', () => {
  const steps = planStepsFromTasks([
    { id: 'yes', metadata: { risky: true } },
    { id: 'no', metadata: { risky: false } },
    { id: 'missing' },
    { id: 'string', metadata: { risky: 'true' } },
  ]);
  assert.deepEqual(
    steps.map((step) => step.risky),
    [true, false, false, false],
  );
});

test('enabled is always true from the pure function', () => {
  const steps = planStepsFromTasks([
    { id: 'a', metadata: { enabled: false } },
    { id: 'b' },
  ]);
  assert.deepEqual(
    steps.map((step) => step.enabled),
    [true, true],
  );
});

test('completed tasks are omitted; pending, running, and failed stay in order', () => {
  const steps = planStepsFromTasks([
    { id: 'pend', status: 'pending', title: 'Pend' },
    { id: 'done', status: 'completed', title: 'Done' },
    { id: 'run', status: 'running', title: 'Run' },
    { id: 'fail', status: 'failed', title: 'Fail' },
    { id: 'none', title: 'None' },
  ]);
  assert.deepEqual(
    steps.map((step) => step.id),
    ['pend', 'run', 'fail', 'none'],
  );
});

test('missing title becomes the id and missing note becomes empty', () => {
  const steps = planStepsFromTasks([
    { id: 'bare' },
    { id: 'noted', title: 'Ship', note: 'watch the migration' },
  ]);
  assert.deepEqual(steps, [
    {
      id: 'bare',
      title: 'bare',
      note: '',
      risky: false,
      enabled: true,
      executor: 'graphirm',
    },
    {
      id: 'noted',
      title: 'Ship',
      note: 'watch the migration',
      risky: false,
      enabled: true,
      executor: 'graphirm',
    },
  ]);
});

function step(partial: Pick<PlanStep, 'id' | 'title' | 'enabled'> & Partial<PlanStep>): PlanStep {
  return {
    note: '',
    risky: false,
    executor: 'graphirm',
    ...partial,
  };
}

test('runSteerText names enabled titles in input order and omits disabled steps', () => {
  const text = runSteerText([
    step({ id: 'a', title: 'Alpha', enabled: true }),
    step({ id: 'b', title: 'Beta', enabled: false }),
    step({ id: 'c', title: 'Gamma', enabled: true, executor: 'pi', risky: true }),
  ]);
  assert.equal(text, 'Resume these steps: Alpha, Gamma.');
  assert.equal(text.includes('Beta'), false);
});

test('runSteerText with no enabled steps still asks to resume and names nothing', () => {
  const text = runSteerText([
    step({ id: 'b', title: 'Beta', enabled: false }),
  ]);
  assert.equal(text, 'Resume these steps.');
  assert.equal(text.includes('Beta'), false);
});
