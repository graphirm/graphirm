import React, { useCallback, useMemo, useRef, useState } from 'react';
import type { Message, PendingApproval } from '../types/graph';
import { MarkdownBody } from './nodes/MarkdownBody';
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
        {messages.map(msg => (
          <div key={msg.id} className={[styles.message, styles[msg.role as keyof typeof styles] ?? ''].join(' ')}>
            <div className={styles.roleLabel}>{msg.role}</div>
            {msg.role === 'user' ? (
              <div style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{msg.content}</div>
            ) : (
              <MarkdownBody content={msg.content} maxHeight={250} />
            )}
          </div>
        ))}
        {streamingMessage && (
          <div
            key={streamingMessage.id}
            className={[styles.message, styles.assistant ?? ''].filter(Boolean).join(' ')}
          >
            <div className={styles.roleLabel}>assistant</div>
            <MarkdownBody content={streamingMessage.content || '…'} maxHeight={250} />
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
