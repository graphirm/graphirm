import { useState } from 'react';
import type { PendingApproval } from '../types/graph';
import styles from '../styles/chat.module.css';

interface HitlOverlayProps {
  approval: PendingApproval;
  onApprove: (nodeId: string) => void;
  onReject: (nodeId: string, reason?: string) => void;
  onModify: (nodeId: string, modifiedArgs: string) => void;
  className?: string;
}

export function HitlOverlay({
  approval,
  onApprove,
  onReject,
  onModify,
  className = '',
}: HitlOverlayProps) {
  const [mode, setMode] = useState<'idle' | 'reject' | 'modify'>('idle');
  const [reason, setReason] = useState('');
  const [modifiedArgs, setModifiedArgs] = useState(
    typeof approval.arguments === 'string'
      ? approval.arguments
      : JSON.stringify(approval.arguments, null, 2),
  );

  return (
    <div className={`${styles.hitlCard} ${className}`}>
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