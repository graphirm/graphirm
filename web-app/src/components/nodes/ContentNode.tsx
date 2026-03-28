import { useState } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { GraphNode } from '../../types/graph';
import { useFocusedNodeId } from '../../context/FocusContext';
import { BaseCard } from './BaseCard';
import { CodeBody } from './CodeBody';

export function ContentNode({ id, data: rawData, selected }: NodeProps) {
  const [expanded, setExpanded] = useState(false);
  const focusedNodeId = useFocusedNodeId();
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
      focused={focusedNodeId === id}
    >
      {nt.path && (
        <div style={{ fontSize: 10, color: 'var(--fg-muted)' }}>
          📄 {nt.path}
          {nt.language && (
            <span style={{ marginLeft: 6, color }}>{nt.language}</span>
          )}
          <span style={{ marginLeft: 8 }}>
            {(nt.body ?? '').split('\n').length} lines
          </span>
        </div>
      )}
      <CodeBody code={nt.body ?? ''} language={nt.language} maxHeight={360} />
    </BaseCard>
  );
}
