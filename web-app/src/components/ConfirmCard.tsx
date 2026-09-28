import { useState } from 'react';
import { confirmSections } from '../chat/confirmBody';
import type { PendingApproval } from '../types/graph';
import styles from '../styles/chat.module.css';

export interface ConfirmCardProps {
  approval: PendingApproval;
  onApprove: (nodeId: string) => void;
  onReject: (nodeId: string, reason?: string) => void;
  onModify: (nodeId: string, modifiedArgs: string) => void;
  /** Session abort. Hidden when the caller has no abort callback. */
  onAbort?: () => void;
  className?: string;
}

export function ConfirmCard({
  approval,
  onApprove,
  onReject,
  onModify,
  onAbort,
  className = '',
}: ConfirmCardProps) {
  const [mode, setMode] = useState<'idle' | 'reject' | 'modify'>('idle');
  const [reason, setReason] = useState('');
  const [modifiedArgs, setModifiedArgs] = useState(
    typeof approval.arguments === 'string'
      ? approval.arguments
      : JSON.stringify(approval.arguments, null, 2),
  );

  const sections = confirmSections(approval.tool_name, approval.arguments);
  const pIrreversible = approval.hitl_judge?.p_irreversible;

  return (
    <div className={`${styles.hitlCard} ${className}`}>
      <div className={styles.hitlHeader}>{sections.heading}</div>
      <div className={styles.hitlArgs}>
        <pre>{sections.body}</pre>
      </div>
      {sections.note && <div className={styles.hitlNote}>{sections.note}</div>}
      {typeof pIrreversible === 'number' && (
        <div className={styles.hitlScore}>P(irreversible) {pIrreversible}</div>
      )}

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

      {onAbort && (
        <div className={styles.hitlActions}>
          <button className="secondary" onClick={onAbort}>
            Cancel the rest
          </button>
        </div>
      )}
    </div>
  );
}
