import { useState } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { GraphNode } from '../../types/graph';
import { useFocusedNodeId } from '../../context/FocusContext';
import { BaseCard } from './BaseCard';
import styles from '../../styles/nodes.module.css';

const STATUS_CLASS: Record<string, string> = {
  idle: styles.statusPending,
  running: styles.statusRunning,
  completed: styles.statusCompleted,
  failed: styles.statusFailed,
};

export function AgentNode({ id, data: rawData, selected }: NodeProps) {
  const [expanded, setExpanded] = useState(false);
  const focusedNodeId = useFocusedNodeId();
  const data = rawData as unknown as GraphNode;
  const nt = data.node_type;
  if (nt.type !== 'Agent') return null;

  const color = 'var(--node-agent)';
  const preview = `${nt.name} · ${nt.model}`;

  return (
    <BaseCard
      color={color}
      typeLabel="agent"
      timestamp={data.created_at}
      preview={preview}
      selected={selected}
      expanded={expanded}
      onToggleExpand={() => setExpanded(e => !e)}
      focused={focusedNodeId === id}
    >
      <div className={styles.header}>
        <span className={styles.typeBadge} style={{ background: '#ef9a9a33', color: 'var(--node-agent)' }}>
          agent
        </span>
        <span className={[styles.statusChip, STATUS_CLASS[nt.status] ?? styles.statusPending].join(' ')}>
          {nt.status}
        </span>
      </div>
      <div className={styles.body}>
        <strong>{nt.name}</strong> · {nt.model}
        {nt.system_prompt && (
          <details style={{ marginTop: 6 }}>
            <summary style={{ cursor: 'pointer', color: 'var(--fg-muted)', fontSize: 11 }}>
              System prompt
            </summary>
            <pre style={{ marginTop: 4, fontSize: 10, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
              {nt.system_prompt}
            </pre>
          </details>
        )}
      </div>
    </BaseCard>
  );
}
