import { useState } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { GraphNode } from '../../types/graph';
import { BaseCard } from './BaseCard';
import styles from '../../styles/nodes.module.css';

const STATUS_CLASS: Record<string, string> = {
  pending: styles.statusPending,
  running: styles.statusRunning,
  completed: styles.statusCompleted,
  failed: styles.statusFailed,
};

export function TaskNode({ data: rawData, selected }: NodeProps) {
  const [expanded, setExpanded] = useState(false);
  const data = rawData as unknown as GraphNode;
  const nt = data.node_type;
  if (nt.type !== 'Task') return null;

  const color = 'var(--node-task)';
  const preview = nt.title;

  return (
    <BaseCard
      color={color}
      typeLabel="task"
      timestamp={data.created_at}
      preview={preview}
      selected={selected}
      expanded={expanded}
      onToggleExpand={() => setExpanded(e => !e)}
    >
      <div className={styles.header}>
        <span className={styles.typeBadge} style={{ background: '#ffb74d33', color }}>
          task
        </span>
        <span className={[styles.statusChip, STATUS_CLASS[nt.status] ?? styles.statusPending].join(' ')}>
          {nt.status}
        </span>
        {nt.priority != null && (
          <span style={{ fontSize: 10, color: 'var(--fg-muted)', marginLeft: 'auto' }}>
            P{nt.priority}
          </span>
        )}
      </div>
      <div className={styles.body}>
        <strong>{nt.title}</strong>
        {nt.description && (
          <div style={{ marginTop: 4, color: 'var(--fg-muted)' }}>{nt.description}</div>
        )}
      </div>
    </BaseCard>
  );
}
