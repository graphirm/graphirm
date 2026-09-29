export interface ReviewItem {
  kind: 'approval' | 'failed' | 'paused' | 'pi';
  id: string;
  label: string;
}

function preferredLabel(value: string | undefined, id: string): string {
  return value ? value : id;
}

function pauseLabel(sessionName: string | undefined): string {
  return sessionName ? sessionName : 'Paused';
}

/** Attention rows: one pending gate, then flagged sessions, then open Pi tasks. */
export function buildReviewItems(input: {
  pending: {
    node_id: string;
    tool_name: string;
    is_pause?: boolean;
    session_name?: string;
  } | null;
  sessions: { id: string; status?: string; name?: string }[];
  tasks: { id: string; status?: string; executor?: string; title?: string }[];
}): ReviewItem[] {
  const items: ReviewItem[] = [];

  if (input.pending?.is_pause) {
    items.push({
      kind: 'paused',
      id: input.pending.node_id,
      label: pauseLabel(input.pending.session_name),
    });
  } else if (input.pending) {
    items.push({
      kind: 'approval',
      id: input.pending.node_id,
      label: input.pending.tool_name,
    });
  }

  for (const session of input.sessions) {
    if (session.status === 'failed' || session.status === 'token_cap_exceeded') {
      items.push({
        kind: 'failed',
        id: session.id,
        label: preferredLabel(session.name, session.id),
      });
    } else if (session.status === 'paused') {
      items.push({
        kind: 'paused',
        id: session.id,
        label: preferredLabel(session.name, session.id),
      });
    }
  }

  for (const task of input.tasks) {
    if (task.executor !== 'pi' || task.status === 'completed' || task.status === 'failed') continue;
    items.push({
      kind: 'pi',
      id: task.id,
      label: preferredLabel(task.title, task.id),
    });
  }

  return items;
}
