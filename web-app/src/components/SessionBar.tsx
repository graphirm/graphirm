import { useState } from 'react';
import type { Session } from '../types/graph';
import styles from './SessionBar.module.css';

interface SessionBarProps {
  sessions: Session[];
  currentSession: Session | null;
  onSelectSession: (id: string) => void;
  onCreateSession: (name?: string, workspace?: string) => Promise<Session | void>;
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
  const [showForm, setShowForm] = useState(false);
  const [sessionName, setSessionName] = useState('');
  const [workspaceName, setWorkspaceName] = useState('');

  const handleCreate = async () => {
    const name = sessionName.trim() || undefined;
    const workspace = workspaceName.trim() || undefined;
    try {
      await onCreateSession(name, workspace);
      setShowForm(false);
      setSessionName('');
      setWorkspaceName('');
    } catch {
      // keep form open so user can retry; error handling is upstream
    }
  };

  const handleCancel = () => {
    setShowForm(false);
    setSessionName('');
    setWorkspaceName('');
  };

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
        {showForm ? (
          <>
            <input
              autoFocus
              placeholder="Session name (optional)"
              value={sessionName}
              onChange={e => setSessionName(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter') handleCreate();
                if (e.key === 'Escape') handleCancel();
              }}
              style={{ fontSize: 12, width: 150, padding: '2px 6px' }}
            />
            <input
              placeholder="Workspace (optional)"
              value={workspaceName}
              onChange={e => setWorkspaceName(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter') handleCreate();
                if (e.key === 'Escape') handleCancel();
              }}
              style={{ fontSize: 12, width: 130, padding: '2px 6px' }}
            />
            <button onClick={handleCreate}>Create</button>
            <button className="secondary" onClick={handleCancel} style={{ fontSize: 11 }}>Cancel</button>
          </>
        ) : (
          <button onClick={() => setShowForm(true)}>+ New</button>
        )}
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
