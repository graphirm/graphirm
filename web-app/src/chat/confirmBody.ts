export interface ConfirmSections {
  heading: string;
  body: string;
  note?: string;
}

const PREVIEW_LIMIT = 400;

function pretty(args: Record<string, unknown>): string {
  return JSON.stringify(args, null, 2);
}

function asString(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined;
}

function shortPreview(text: string): string {
  if (text.length <= PREVIEW_LIMIT) return text;
  return `${text.slice(0, PREVIEW_LIMIT)}…`;
}

/**
 * Tool-specific body for the confirm card.
 * A string `args` is shown as-is. `bash` shows the command, `write` and `edit`
 * show the path plus a short preview, and `delegate_pi` shows the task.
 */
export function confirmSections(
  toolName: string,
  args: Record<string, unknown> | string,
): ConfirmSections {
  const heading = `Agent wants to run: ${toolName}`;
  if (typeof args === 'string') {
    return { heading, body: args };
  }

  if (toolName === 'bash') {
    const command = asString(args.command);
    return { heading, body: command ?? pretty(args) };
  }

  if (toolName === 'write') {
    const path = asString(args.path) ?? '';
    const content = shortPreview(asString(args.content) ?? '');
    return { heading, body: `${path}\n${content}` };
  }

  if (toolName === 'edit') {
    const path = asString(args.path) ?? '';
    const oldString = shortPreview(asString(args.old_string) ?? '');
    const newString = shortPreview(asString(args.new_string) ?? '');
    return { heading, body: `${path}\n- ${oldString}\n+ ${newString}` };
  }

  if (toolName === 'delegate_pi') {
    const task = asString(args.task) ?? pretty(args);
    return {
      heading,
      body: task,
      note: 'Pi will not pause for its own calls',
    };
  }

  return { heading, body: pretty(args) };
}
