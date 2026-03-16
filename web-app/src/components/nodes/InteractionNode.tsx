import { useState } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { GraphNode } from '../../types/graph';
import { BaseCard } from './BaseCard';
import styles from '../../styles/nodes.module.css';

export function InteractionNode({ data: rawData, selected }: NodeProps) {
  const [expanded, setExpanded] = useState(false);
  const data = rawData as unknown as GraphNode & { onSteer?: (nodeId: string) => void };
  const nt = data.node_type;
  if (nt.type !== 'Interaction') return null;

  const color = 'var(--node-interaction)';
  const roleLabel = nt.role === 'user' ? 'user' : nt.role === 'assistant' ? 'agent' : nt.role;
  const preview = (nt.content ?? '').slice(0, 80) + ((nt.content ?? '').length > 80 ? '…' : '');

  return (
    <BaseCard
      color={color}
      typeLabel={roleLabel}
      timestamp={data.created_at}
      preview={preview}
      selected={selected}
      expanded={expanded}
      onToggleExpand={() => setExpanded(e => !e)}
    >
      <div className={styles.body}>{nt.content}</div>
      {data.onSteer && (
        <button className={styles.steerBtn} onClick={() => data.onSteer?.(data.id)}>
          ↩ Steer from here
        </button>
      )}
    </BaseCard>
  );
}
