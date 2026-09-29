import { buildSteps, type StepRow } from './steps.ts';

export interface LiveToolStart {
  callId: string;
  toolName: string;
}

/** One director tool call. Nested names are Pi lines under an open `delegate_pi`. */
export interface LiveDirector {
  callId: string;
  toolName: string;
  nested: string[];
  /** False after the next assistant `message_end`. Still shown until messages cover the call id. */
  open: boolean;
}

export type LiveStepRow = StepRow & { nestedNames: string[] };

function latestOpenIndex(directors: readonly LiveDirector[]): number {
  for (let i = directors.length - 1; i >= 0; i--) {
    if (directors[i].open) return i;
  }
  return -1;
}

/**
 * Fold one `tool_start` into the live list.
 * While the latest open (unmatched) director is `delegate_pi`, the start is a
 * nested tool name under that row. Otherwise it is a new director call.
 */
export function applyLiveToolStart(
  directors: readonly LiveDirector[],
  start: LiveToolStart,
): LiveDirector[] {
  const index = latestOpenIndex(directors);
  const latest = index === -1 ? undefined : directors[index];
  if (latest?.toolName === 'delegate_pi') {
    const next = directors.slice();
    next[index] = {
      ...latest,
      nested: [...latest.nested, start.toolName],
    };
    return next;
  }
  return [
    ...directors,
    {
      callId: start.callId,
      toolName: start.toolName,
      nested: [],
      open: true,
    },
  ];
}

/**
 * The next assistant message has ended, so every open director is matched.
 * Later `tool_start` events are peer steps. `tool_end` is not used: it has no
 * call id and no error is recorded from it.
 */
export function closeLiveDirectors(directors: readonly LiveDirector[]): LiveDirector[] {
  return directors.map((director) => (director.open ? { ...director, open: false } : director));
}

/** Drop directors whose call ids are already persisted tool messages. */
export function dropCoveredLiveDirectors<T extends { callId: string }>(
  directors: readonly T[],
  coveredCallIds: ReadonlySet<string>,
): T[] {
  return directors.filter((director) => !coveredCallIds.has(director.callId));
}

/** Director rows for the live STEPS list. Nested Pi lines stay under the row. */
export function toLiveStepRows(directors: readonly LiveDirector[]): LiveStepRow[] {
  return directors.map((director) => {
    const [row] = buildSteps([{ toolName: director.toolName, callId: director.callId }]);
    return { ...row, nestedNames: [...director.nested] };
  });
}
