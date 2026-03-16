import type { Session } from '../types/graph';
import styles from './SessionBar.module.css';

interface SessionBarProps {
  sessions: Session[];
  currentSession: Session | null;
  onSelectSession: (id: string) => void;
  onCreateSession: () => void;
  onPause: () => void;
  onResume: () => void;
  autoApprove: boolean;
  onToggleAutoApprove: () => void;
}

export function SessionBar({
  sessions,
  currentSession,
  onSelectSession,
  onCreateSession,
  onPause,
  onResume,
  autoApprove,
  onToggleAutoApprove,
}: SessionBarProps) {
  return (
    <header className={styles.header}>
      <span className={styles.logo}>graphirm</span>
      <div className={styles.controls}>
        <select
          value={currentSession?.id ?? ''}
          onChange={e => e.target.value && onSelectSession(e.target.value)}
        >
          {sessions.length === 0 && <option value="">— no sessions —</option>}
          {sessions.map(s => (
            <option key={s.id} value={s.id}>
              {s.name ?? s.id.slice(0, 12)}
            </option>
          ))}
        </select>
        <button onClick={() => onCreateSession()}>+ New</button>
        {currentSession && (
          <>
            <button className="secondary" onClick={onPause} style={{ fontSize: 11 }}>Pause</button>
            <button className="secondary" onClick={onResume} style={{ fontSize: 11 }}>Resume</button>
            <button
              onClick={onToggleAutoApprove}
              style={{
                fontSize: 11,
                background: autoApprove ? '#16a34a' : '#3c3c3c',
                color: autoApprove ? '#fff' : '#d4d4d4',
                border: `1px solid ${autoApprove ? '#16a34a' : '#555'}`,
                borderRadius: 3,
                padding: '2px 8px',
                cursor: 'pointer',
              }}
              title={autoApprove ? 'Auto-approve ON — all tool calls run without confirmation' : 'Auto-approve OFF — destructive tools require confirmation'}
            >
              {autoApprove ? 'Auto-approve ON' : 'Auto-approve'}
            </button>
          </>
        )}
      </div>
    </header>
  );
}
