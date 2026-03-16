import { useState } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { GraphNode } from '../../types/graph';
import { BaseCard } from './BaseCard';
import styles from '../../styles/nodes.module.css';

export function ContentNode({ data: rawData, selected }: NodeProps) {
  const [expanded, setExpanded] = useState(false);
  const data = rawData as unknown as GraphNode;
  const nt = data.node_type;
  if (nt.type !== 'Content') return null;

  const color = 'var(--node-content)';
  const label = nt.path ?? nt.content_type;
  const bodyPreview = (nt.body ?? '').slice(0, 60) + ((nt.body ?? '').length > 60 ? '…' : '');
  const preview = `${label}: ${bodyPreview}`;

  return (
    <BaseCard
      color={color}
      typeLabel={nt.content_type}
      timestamp={data.created_at}
      preview={preview}
      selected={selected}
      expanded={expanded}
      onToggleExpand={() => setExpanded(e => !e)}
    >
      {nt.path && (
        <div style={{ fontSize: 10, color: 'var(--fg-muted)', marginBottom: 4 }}>
          📄 {nt.path}
          {nt.language && <span style={{ marginLeft: 6, color }}>{nt.language}</span>}
        </div>
      )}
      <pre className={styles.body} style={{ fontFamily: 'monospace', fontSize: 11 }}>
        {nt.body}
      </pre>
    </BaseCard>
  );
}
