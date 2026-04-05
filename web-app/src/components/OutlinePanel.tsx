import { useEffect, useState } from 'react';
import type { GraphNode } from '../types/graph';
import { api } from '../api/client';
import styles from '../styles/chat.module.css';

interface OutlinePanelProps {
  sessionId: string | null;
  interactionId: string | null;
  onOutlineSteer: (outlineNodeId: string, interactionId: string) => void;
}

export function OutlinePanel({ sessionId, interactionId, onOutlineSteer }: OutlinePanelProps) {
  const [items, setItems] = useState<GraphNode[]>([]);
  useEffect(() => {
    if (!sessionId || !interactionId) {
      setItems([]);
      return;
    }
    api.getOutline(sessionId, interactionId).then(setItems).catch(() => setItems([]));
  }, [sessionId, interactionId]);
  if (!items.length) return null;
  return (
    <div className={styles.outlinePanel}>
      <div className={styles.outlinePanelTitle}>Outline</div>
      <ul className={styles.outlineList}>
        {items.map((n) => {
          const title = n.metadata?.outline_title ?? n.id;
          const kind = typeof n.metadata?.outline_kind === 'string' ? n.metadata.outline_kind : '';
          const iid = interactionId ?? '';
          return (
            <li key={n.id} className={styles.outlineRow}>
              <span className={styles.outlineKind}>{kind}</span>
              <span className={styles.outlineTitleText}>{String(title)}</span>
              <button
                type="button"
                className={styles.outlineSteerBtn}
                onClick={() => iid && onOutlineSteer(n.id, iid)}
              >
                Steer
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
