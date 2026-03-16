import { useCallback, useRef, useState } from 'react';
import type { Message, PendingApproval } from '../types/graph';
import styles from '../styles/chat.module.css';

interface SteerContext {
  nodeId: string;
}

interface ChatPaneProps {
  messages: Message[];
  isThinking: boolean;
  pendingApproval: PendingApproval | null;
  sessionId: string | null;
  steerContext: SteerContext | null;
  onSend: (content: string) => void;
  onAbort: () => void;
  onApprove: (nodeId: string) => void;
  onReject: (nodeId: string, reason?: string) => void;
  onModify: (nodeId: string, modifiedArgs: string) => void;
  onClearSteer: () => void;
}

function HitlCard({
  approval,
  onApprove,
  onReject,
  onModify,
}: {
  approval: PendingApproval;
  onApprove: (nodeId: string) => void;
  onReject: (nodeId: string, reason?: string) => void;
  onModify: (nodeId: string, modifiedArgs: string) => void;
}) {
  const [mode, setMode] = useState<'idle' | 'reject' | 'modify'>('idle');
  const [reason, setReason] = useState('');
  const [modifiedArgs, setModifiedArgs] = useState(
    typeof approval.arguments === 'string'
      ? approval.arguments
      : JSON.stringify(approval.arguments, null, 2),
  );

  return (
    <div className={styles.hitlCard}>
      <div className={styles.hitlHeader}>
        ⚠ Agent wants to run: <strong>{approval.tool_name}</strong>
      </div>
      <div className={styles.hitlArgs}>
        <pre>{typeof approval.arguments === 'string'
          ? approval.arguments
          : JSON.stringify(approval.arguments, null, 2)}
        </pre>
      </div>

      {mode === 'idle' && (
        <div className={styles.hitlActions}>
          <button className={styles.hitlApprove} onClick={() => onApprove(approval.node_id)}>
            Approve
          </button>
          <button className={styles.hitlReject} onClick={() => setMode('reject')}>
            Reject
          </button>
          <button className={styles.hitlModify} onClick={() => setMode('modify')}>
            Modify
          </button>
        </div>
      )}

      {mode === 'reject' && (
        <>
          <textarea
            className={styles.hitlTextarea}
            placeholder="Reason (optional)"
            value={reason}
            onChange={e => setReason(e.target.value)}
          />
          <div className={styles.hitlActions}>
            <button className={styles.hitlReject} onClick={() => onReject(approval.node_id, reason)}>
              Confirm Reject
            </button>
            <button className="secondary" onClick={() => setMode('idle')}>Cancel</button>
          </div>
        </>
      )}

      {mode === 'modify' && (
        <>
          <textarea
            className={styles.hitlTextarea}
            value={modifiedArgs}
            onChange={e => setModifiedArgs(e.target.value)}
            rows={6}
          />
          <div className={styles.hitlActions}>
            <button className={styles.hitlApprove} onClick={() => onModify(approval.node_id, modifiedArgs)}>
              Approve Modified
            </button>
            <button className="secondary" onClick={() => setMode('idle')}>Cancel</button>
          </div>
        </>
      )}
    </div>
  );
}

export function ChatPane({
  messages,
  isThinking,
  pendingApproval,
  steerContext,
  onSend,
  onAbort,
  onApprove,
  onReject,
  onModify,
  onClearSteer,
}: ChatPaneProps) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

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
    <div className={styles.chatPane}>
      <div className={styles.messages}>
        {messages.map(msg => (
          <div key={msg.id} className={[styles.message, styles[msg.role as keyof typeof styles] ?? ''].join(' ')}>
            <div className={styles.roleLabel}>{msg.role}</div>
            <div style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{msg.content}</div>
          </div>
        ))}
        {pendingApproval && (
          <HitlCard
            approval={pendingApproval}
            onApprove={onApprove}
            onReject={onReject}
            onModify={onModify}
          />
        )}
        <div ref={messagesEndRef} />
      </div>

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
        <textarea
          rows={2}
          placeholder={steerContext ? 'Send message from this context node…' : 'Type your message… (Enter to send, Shift+Enter for newline)'}
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
