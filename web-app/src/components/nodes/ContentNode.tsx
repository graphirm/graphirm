import { useMemo, useState } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { GraphNode } from '../../types/graph';
import { useFocusedNodeId } from '../../context/FocusContext';
import { useZoom } from '../../context/ZoomContext';
import { estimateExpandedPlainReserveHeight } from '../../layout/pretextDimensions';
import { BaseCard } from './BaseCard';
import { CodeBody } from './CodeBody';

export function ContentNode({ id, data: rawData, selected }: NodeProps) {
  const [expanded, setExpanded] = useState(false);
  const focusedNodeId = useFocusedNodeId();
  const { isLODEnabled } = useZoom();
  const data = rawData as unknown as GraphNode;
  const nt = data.node_type;
  if (nt.type !== 'Content') return null;

  const color = 'var(--node-content)';
  const label = nt.path ?? nt.content_type;
  const bodyPreview = (nt.body ?? '').slice(0, 60) + ((nt.body ?? '').length > 60 ? '…' : '');
  const preview = `${label}: ${bodyPreview}`;

  const expandedBodyStyle = useMemo(() => {
    if (isLODEnabled || !expanded) return undefined;
    try {
      const text = [nt.path, nt.body ?? ''].filter(Boolean).join('\n');
      const minH = estimateExpandedPlainReserveHeight(text, 360, 88);
      return { minHeight: minH, maxHeight: 520, overflowY: 'auto' as const };
    } catch {
      return undefined;
    }
  }, [expanded, isLODEnabled, nt.path, nt.body]);

  return (
    <BaseCard
      color={color}
      typeLabel={nt.content_type}
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
