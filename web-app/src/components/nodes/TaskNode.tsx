import { useMemo, useState } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { GraphNode } from '../../types/graph';
import { useFocusedNodeId } from '../../context/FocusContext';
import { useZoom } from '../../context/ZoomContext';
import { estimateExpandedPlainReserveHeight } from '../../layout/pretextDimensions';
import { BaseCard } from './BaseCard';
import styles from '../../styles/nodes.module.css';

const STATUS_CLASS: Record<string, string> = {
  pending: styles.statusPending,
  running: styles.statusRunning,
  completed: styles.statusCompleted,
  failed: styles.statusFailed,
};

export function TaskNode({ id, data: rawData, selected }: NodeProps) {
  const [expanded, setExpanded] = useState(false);
  const focusedNodeId = useFocusedNodeId();
  const { isLODEnabled } = useZoom();
  const data = rawData as unknown as GraphNode;
  const nt = data.node_type;
  if (nt.type !== 'Task') return null;

  const color = 'var(--node-task)';
  const preview = nt.title;

  const expandedBodyStyle = useMemo(() => {
    if (isLODEnabled || !expanded) return undefined;
    try {
      const text = `${nt.title}\n${nt.description ?? ''}`;
      const minH = estimateExpandedPlainReserveHeight(text, 260, 120);
      return { minHeight: minH, maxHeight: 480, overflowY: 'auto' as const };
    } catch {
      return undefined;
    }
  }, [expanded, isLODEnabled, nt.title, nt.description]);

  return (
    <BaseCard
      color={color}
      typeLabel="task"
      timestamp={data.created_at}
      preview={preview}
      selected={selected}
      expanded={isLODEnabled ? false : expanded}
      onToggleExpand={() => {
        if (!isLODEnabled) {
          setExpanded(e => !e);
        }
      }}
      focused={focusedNodeId === id}
      expandedBodyStyle={expandedBodyStyle}
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
