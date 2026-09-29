export interface PlanStep {
  id: string;
  title: string;
  note: string;
  risky: boolean;
  enabled: boolean;
  executor: 'graphirm' | 'pi';
}

export function planStepsFromTasks(tasks: {
  id: string;
  title?: string;
  status?: string;
  metadata?: Record<string, unknown>;
  note?: string;
}[]): PlanStep[] {
  const steps: PlanStep[] = [];
  for (const task of tasks) {
    if (task.status === 'completed') continue;
    steps.push({
      id: task.id,
      title: typeof task.title === 'string' ? task.title : task.id,
      note: typeof task.note === 'string' ? task.note : '',
      risky: task.metadata?.risky === true,
      enabled: true,
      executor: task.metadata?.executor === 'pi' ? 'pi' : 'graphirm',
    });
  }
  return steps;
}

/** Names each enabled step, in input order, and asks to resume those steps. */
export function runSteerText(steps: PlanStep[]): string {
  const titles = steps.filter((step) => step.enabled).map((step) => step.title);
  if (titles.length === 0) return 'Resume these steps.';
  return `Resume these steps: ${titles.join(', ')}.`;
}
