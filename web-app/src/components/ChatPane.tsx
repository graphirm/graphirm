import React, { useCallback, useMemo, useRef, useState } from 'react';
import type { LiveDirector } from '../chat/liveSteps';
import { toLiveStepRows } from '../chat/liveSteps';
import type { PlanStep } from '../chat/planCard';
import type { StepInput, StepRow } from '../chat/steps';
import { buildSteps } from '../chat/steps';
import { formatJevChip } from '../chat/jevChip';
import { replyHints } from '../chat/replyHints';
import { parseSegmentPrefix } from '../chat/segmentStream';
import type { Message, PendingApproval } from '../types/graph';
import { MarkdownBody } from './nodes/MarkdownBody';
import { BlockView } from './BlockView';
import { JevChip, JevSheet } from './JevChip';
import { StepsRow } from './StepsRow';
import { cleanLegacyAssistantContent } from '../utils/chatSegments';
import { HitlOverlay } from './HitlOverlay';
import { OutlinePanel } from './OutlinePanel';
import { PlanCard } from './PlanCard';
import styles from '../styles/chat.module.css';

interface SteerContext {
  nodeId: string;
}

interface ChatPaneProps {
  messages: Message[];
  /** In-flight assistant text from SSE message_delta (cleared on message_end). */
  streamingMessage?: Message | null;
  /** Director tool calls still running. Stays up after `message_end` clears the stream. */
  liveSteps?: readonly LiveDirector[];
  isThinking: boolean;
  pendingApproval: PendingApproval | null;
  sessionId: string | null;
  steerContext: SteerContext | null;
  inputRef?: React.RefObject<HTMLTextAreaElement | null>;
  onSend: (content: string) => void;
  onAbort: () => void;
  onApprove: (nodeId: string) => void;
  onReject: (nodeId: string, reason?: string) => void;
  onModify: (nodeId: string, modifiedArgs: string) => void;
  onClearSteer: () => void;
  chatCollapsed?: boolean;
  onToggleCollapse?: () => void;
  /** Open plan steps. The card renders above the composer when this list is non-empty. */
  planSteps?: PlanStep[];
  /** Per-step enabled flags for the current session. A missing id is enabled. */
  planEnabledById?: Record<string, boolean>;
  onTogglePlanStep?: (stepId: string) => void;
  /** True while this session's plan run is in flight. Owned by DecisionShell. */
  planRunLocked?: boolean;
  onPlanRunLock?: (sessionId: string) => void;
  onPlanRunUnlock?: (sessionId: string) => void;
  /** Session resume. Plan Run awaits this, then sends via `onPlanSend`. */
  onPlanResume?: () => void | Promise<void>;
  /** Plain prompt send for plan Run. Does not attach or clear steer scope. */
  onPlanSend?: (content: string) => void;
  /** Scoped steer targeting an outline row (server adds steer_context to prompt). */
  outlineSteer?: { outlineNodeId: string; interactionId: string } | null;
  onClearOutlineSteer?: () => void;
  onOutlineSteer?: (outlineNodeId: string, interactionId: string) => void;
}

type ChatEntry =
  | { kind: 'message'; message: Message }
  | { kind: 'steps'; id: string; steps: StepRow[] };

const ARGS_SUMMARY_CAP = 120;

function stringArg(obj: Record<string, unknown>, key: string): string | undefined {
  const value = obj[key];
  return typeof value === 'string' ? value : undefined;
}

function capSummary(text: string): string {
  if (text.length <= ARGS_SUMMARY_CAP) return text;
  return `${text.slice(0, ARGS_SUMMARY_CAP - 1)}…`;
}

/** One-line call arguments. Result text is never used. */
function summarizeArguments(args: unknown): string {
  if (args == null) return '';
  if (typeof args !== 'object' || Array.isArray(args)) return capSummary(JSON.stringify(args));
  const obj = args as Record<string, unknown>;
  const command = stringArg(obj, 'command');
  if (command !== undefined) return capSummary(command);
  const path = stringArg(obj, 'path');
  const file = stringArg(obj, 'file');
  const pattern = stringArg(obj, 'pattern');
  if (path !== undefined && pattern !== undefined) return capSummary(`${pattern} ${path}`);
  if (path !== undefined) return capSummary(path);
  if (file !== undefined) return capSummary(file);
  if (pattern !== undefined) return capSummary(pattern);
  return capSummary(JSON.stringify(obj));
}

function callsInTurn(turn: Message[]): Map<string, unknown> {
  const calls = new Map<string, unknown>();
  for (const message of turn) {
    if (message.role !== 'assistant' || !message.toolCalls) continue;
    for (const call of message.toolCalls) {
      if (!calls.has(call.id)) calls.set(call.id, call.arguments);
    }
  }
  return calls;
}

function stepInputFor(message: Message, calls: ReadonlyMap<string, unknown>): StepInput {
  const toolName = message.toolName?.trim() || 'tool';
  if (toolName === 'delegate_pi') {
    return { toolName, callId: message.id, piSummary: message.content };
  }
  const matched = message.toolCallId ? calls.get(message.toolCallId) : undefined;
  return {
    toolName,
    callId: message.id,
    argsSummary: matched === undefined ? '' : summarizeArguments(matched),
  };
}

/** A turn starts at each user message, matching `buildTurns`. Tool messages fold into one STEPS row before the final assistant reply. */
function chatEntries(messages: Message[]): ChatEntry[] {
  const sorted = [...messages].sort(
    (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
  );
  const turns: Message[][] = [];
  let current: Message[] = [];
  for (const message of sorted) {
    if (message.role === 'user' && current.length > 0) {
      turns.push(current);
      current = [];
    }
    current.push(message);
  }
  if (current.length > 0) turns.push(current);

  return turns.flatMap((turn) => {
    const tools = turn.filter((message) => message.role === 'tool');
    const calls = callsInTurn(turn);
    const steps = buildSteps(tools.map((message) => stepInputFor(message, calls)));
    const bubbles = turn.filter((message) => message.role !== 'tool');
    let finalAssistant = -1;
    for (let i = bubbles.length - 1; i >= 0; i--) {
      if (bubbles[i].role === 'assistant') {
        finalAssistant = i;
        break;
      }
    }
    const stepsEntry: ChatEntry | null = steps.length
      ? { kind: 'steps', id: steps.map((step) => step.callId).join(':'), steps }
      : null;
    const entries: ChatEntry[] = [];
    const head = finalAssistant === -1 ? bubbles : bubbles.slice(0, finalAssistant);
    const tail = finalAssistant === -1 ? [] : bubbles.slice(finalAssistant);
    for (const message of head) entries.push({ kind: 'message', message });
    if (stepsEntry) entries.push(stepsEntry);
    for (const message of tail) entries.push({ kind: 'message', message });
    return entries;
  });
}

function jevLabel(message: Message): string | null {
  if (message.role !== 'assistant') return null;
  return formatJevChip({
    model_tier: message.modelTier,
    routing_strategy: message.routingStrategy,
    routing_confidence: message.routingConfidence,
  });
}

function MessageBody({ message }: { message: Message }) {
  if (message.role === 'user') {
    return <div style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{message.content}</div>;
  }
  if (message.segments && message.segments.length > 0) {
    return (
      <div className={styles.segmentStack}>
        {message.segments.map((seg, i) => (
          <BlockView
            key={`${message.id}-seg-${i}`}
            kicker={seg.type}
            content={seg.content}
            state="done"
          />
        ))}
      </div>
    );
  }
  return <MarkdownBody content={cleanLegacyAssistantContent(message.content)} maxHeight={250} />;
}

function StreamingBody({ message }: { message: Message }) {
  const parsed = parseSegmentPrefix(message.content);
  const segments = parsed.segments.length > 0 ? parsed.segments : (message.segments ?? []);
  const showPlaceholder =
    segments.length === 0 && parsed.plainText === null && !parsed.showRecovery;

  return (
    <>
      {segments.length > 0 && (
        <div className={styles.segmentStack}>
          {segments.map((seg, i) => (
            <BlockView
              key={`${message.id}-live-${i}`}
              kicker={seg.type}
              content={seg.content}
              state={seg.state === 'streaming' ? 'streaming' : 'done'}
            />
          ))}
        </div>
      )}
      {parsed.plainText !== null && (
        <MarkdownBody content={cleanLegacyAssistantContent(parsed.plainText)} maxHeight={250} />
      )}
      {parsed.showRecovery && <div>Fixing the format…</div>}
      {showPlaceholder && <MarkdownBody content="…" maxHeight={250} />}
    </>
  );
}

export function ChatPane({
  messages,
  streamingMessage = null,
  liveSteps = [],
  isThinking,
  pendingApproval,
  sessionId,
  steerContext,
  inputRef,
  onSend,
  onAbort,
  onApprove,
  onReject,
  onModify,
  onClearSteer,
  planSteps = [],
  planEnabledById = {},
  onTogglePlanStep,
  planRunLocked = false,
  onPlanRunLock,
  onPlanRunUnlock,
  onPlanResume,
  onPlanSend,
  chatCollapsed,
  onToggleCollapse,
  outlineSteer = null,
  onClearOutlineSteer,
  onOutlineSteer,
}: ChatPaneProps) {
  const [input, setInput] = useState('');
  const [jevSheetId, setJevSheetId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const closeJevSheet = useCallback(() => setJevSheetId(null), []);
  const jevSheetMessage = useMemo(() => {
    if (!jevSheetId) return null;
    const message = messages.find((m) => m.id === jevSheetId && m.role === 'assistant');
    if (!message || !jevLabel(message)) return null;
    return message;
  }, [jevSheetId, messages]);
  const lastAssistantId = useMemo(
    () => [...messages].reverse().find(m => m.role === 'assistant')?.id ?? null,
    [messages],
  );
  const entries = useMemo(() => chatEntries(messages), [messages]);
  const liveStepRows = useMemo(() => toLiveStepRows(liveSteps), [liveSteps]);
  const showConfirm = pendingApproval != null && !pendingApproval.is_pause;

  const handleSend = useCallback(() => {
    const trimmed = input.trim();
    if (!trimmed || isThinking) return;
    onSend(trimmed);
    setInput('');
    setTimeout(() => messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }), 50);
  }, [input, isThinking, onSend]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  return (
    <div className={`${styles.chatPane} ${chatCollapsed ? styles.collapsed : ''}`}>
      {onToggleCollapse && (
        <button
          className={styles.collapseToggle}
          onClick={onToggleCollapse}
          title={chatCollapsed ? 'Expand chat (C)' : 'Collapse chat (C)'}
        >
          {chatCollapsed ? '▶' : '◀'}
        </button>
      )}
      <div className={styles.messages}>
        {entries.map(entry => {
          if (entry.kind === 'steps') {
            return <StepsRow key={`steps-${entry.id}`} steps={entry.steps} />;
          }
          const label = jevLabel(entry.message);
          return (
            <div
              key={entry.message.id}
              className={[styles.message, styles[entry.message.role as keyof typeof styles] ?? ''].join(' ')}
            >
              <div className={styles.roleLabel}>{entry.message.role}</div>
              <MessageBody message={entry.message} />
              {entry.message.role === 'assistant' &&
                replyHints(entry.message.replyJudge?.scores).map((hint) => (
                  <div key={hint} className={styles.replyHint}>{hint}</div>
                ))}
              {label && (
                <JevChip
                  label={label}
                  onOpen={() =>
                    setJevSheetId((current) => (current === entry.message.id ? null : entry.message.id))
                  }
                />
              )}
            </div>
          );
        })}
        {streamingMessage && (
          <div
            key={streamingMessage.id}
            className={[styles.message, styles.assistant ?? ''].filter(Boolean).join(' ')}
          >
            <div className={styles.roleLabel}>assistant</div>
            <StreamingBody message={streamingMessage} />
          </div>
        )}
        {liveStepRows.length > 0 && <StepsRow key="live-steps" steps={liveStepRows} />}
        {showConfirm && pendingApproval && (
          <HitlOverlay
            approval={pendingApproval}
            onApprove={onApprove}
            onReject={onReject}
            onModify={onModify}
            onAbort={onAbort}
          />
        )}
        <div ref={messagesEndRef} />
      </div>

      {sessionId && lastAssistantId && onOutlineSteer && (
        <OutlinePanel
          sessionId={sessionId}
          interactionId={lastAssistantId}
          onOutlineSteer={onOutlineSteer}
        />
      )}

      {isThinking && (
        <div className={styles.thinkingBar}>
          <span className={styles.thinkingDot} />
          Agent is thinking…
        </div>
      )}

      {planSteps.length > 0 && onPlanResume && onPlanSend && onTogglePlanStep && onPlanRunLock && onPlanRunUnlock && (
        <PlanCard
          steps={planSteps}
          enabledById={planEnabledById}
          onToggle={onTogglePlanStep}
          sessionId={sessionId}
          runLocked={planRunLocked}
          onLockRun={onPlanRunLock}
          onUnlockRun={onPlanRunUnlock}
          onResume={onPlanResume}
          onSend={onPlanSend}
          isThinking={isThinking}
        />
      )}

      <div className={styles.inputBar}>
        {steerContext && (
          <div style={{
            fontSize: 11,
            color: 'var(--node-interaction)',
            background: '#1a3a5c',
            borderRadius: 3,
            padding: '3px 8px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}>
            <span>↩ Steering from node <code>{steerContext.nodeId.slice(0, 8)}</code></span>
            <button
              onClick={onClearSteer}
              style={{ background: 'none', border: 'none', color: 'inherit', fontSize: 12, cursor: 'pointer', padding: '0 4px' }}
            >
              ✕
            </button>
          </div>
        )}
        {outlineSteer && onClearOutlineSteer && (
          <div style={{
            fontSize: 11,
            color: 'var(--accent)',
            background: 'var(--surface-2)',
            borderRadius: 3,
            padding: '3px 8px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}>
            <span>Outline steer: <code>{outlineSteer.outlineNodeId.slice(0, 8)}</code></span>
            <button
              type="button"
              onClick={onClearOutlineSteer}
              style={{ background: 'none', border: 'none', color: 'inherit', fontSize: 12, cursor: 'pointer', padding: '0 4px' }}
            >
              ✕
            </button>
          </div>
        )}
        <textarea
          ref={inputRef}
          rows={2}
          placeholder={
            steerContext
              ? 'Send message from this context node…'
              : outlineSteer
                ? 'Message with outline scope…'
                : 'Type your message… (Enter to send, Shift+Enter for newline)'
          }
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isThinking}
        />
        <div className={styles.inputActions}>
          {isThinking ? (
            <button className="danger" onClick={onAbort}>Abort</button>
          ) : (
            <button onClick={handleSend} disabled={!input.trim()}>Send</button>
          )}
        </div>
      </div>
      {jevSheetMessage && (
        <JevSheet
          interactionId={jevSheetMessage.id}
          reason={jevSheetMessage.routingReason}
          onClose={closeJevSheet}
        />
      )}
    </div>
  );
}
