import React, { useCallback, useMemo, useRef, useState } from 'react';
import type { StepInput, StepRow } from '../chat/steps';
import { buildSteps } from '../chat/steps';
import type { Message, PendingApproval } from '../types/graph';
import { MarkdownBody } from './nodes/MarkdownBody';
import { BlockView } from './BlockView';
import { StepsRow } from './StepsRow';
import { cleanLegacyAssistantContent } from '../utils/chatSegments';
import { HitlOverlay } from './HitlOverlay';
import { OutlinePanel } from './OutlinePanel';
import styles from '../styles/chat.module.css';

interface SteerContext {
  nodeId: string;
}

interface ChatPaneProps {
  messages: Message[];
  /** In-flight assistant text from SSE message_delta (cleared on message_end). */
  streamingMessage?: Message | null;
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

export function ChatPane({
  messages,
  streamingMessage = null,
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
  chatCollapsed,
  onToggleCollapse,
  outlineSteer = null,
  onClearOutlineSteer,
  onOutlineSteer,
}: ChatPaneProps) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const lastAssistantId = useMemo(
    () => [...messages].reverse().find(m => m.role === 'assistant')?.id ?? null,
    [messages],
  );
  const entries = useMemo(() => chatEntries(messages), [messages]);

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
        {entries.map(entry => entry.kind === 'steps' ? (
          <StepsRow key={`steps-${entry.id}`} steps={entry.steps} />
        ) : (
          <div
            key={entry.message.id}
            className={[styles.message, styles[entry.message.role as keyof typeof styles] ?? ''].join(' ')}
          >
            <div className={styles.roleLabel}>{entry.message.role}</div>
            <MessageBody message={entry.message} />
          </div>
        ))}
        {streamingMessage && (
          <div
            key={streamingMessage.id}
            className={[styles.message, styles.assistant ?? ''].filter(Boolean).join(' ')}
          >
            <div className={styles.roleLabel}>assistant</div>
            <MarkdownBody
              content={cleanLegacyAssistantContent(streamingMessage.content || '…')}
              maxHeight={250}
            />
          </div>
        )}
        {pendingApproval && (
          <HitlOverlay
            approval={pendingApproval}
            onApprove={onApprove}
            onReject={onReject}
            onModify={onModify}
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
    </div>
  );
}
