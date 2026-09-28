export interface StepInput {
  toolName: string;
  callId: string;
  isError?: boolean;
  argsSummary?: string;
  /** delegate_pi tool result text, if the call has ended. */
  piSummary?: string;
}

export interface StepRow {
  callId: string;
  title: string;
  detail: string;
  nestedPi: boolean;
}

const PI_TOOL_CALLS = /Tool calls:\s*(\d+)/;

function delegateDetail(piSummary: string | undefined): string {
  const count = piSummary?.match(PI_TOOL_CALLS)?.[1];
  if (!count) return 'judged, not gated';
  return `${count} tool calls · judged, not gated`;
}

/** One collapsed-row step per tool call. `nestedPi` is set only for `delegate_pi`. */
export function buildSteps(calls: StepInput[]): StepRow[] {
  return calls.map((call) => {
    if (call.toolName === 'delegate_pi') {
      return {
        callId: call.callId,
        title: 'delegated to Pi',
        detail: delegateDetail(call.piSummary),
        nestedPi: true,
      };
    }
    return {
      callId: call.callId,
      title: call.toolName,
      detail: call.argsSummary ?? '',
      nestedPi: false,
    };
  });
}
