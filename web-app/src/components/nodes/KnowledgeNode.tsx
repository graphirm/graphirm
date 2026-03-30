import { useMemo, useState } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { GraphNode } from '../../types/graph';
import { useFocusedNodeId } from '../../context/FocusContext';
import { useZoom } from '../../context/ZoomContext';
import { estimateExpandedPlainReserveHeight } from '../../layout/pretextDimensions';
import { BaseCard } from './BaseCard';
import styles from '../../styles/nodes.module.css';

export function KnowledgeNode({ id, data: rawData, selected }: NodeProps) {
  const [expanded, setExpanded] = useState(false);
  const focusedNodeId = useFocusedNodeId();
  const { isLODEnabled } = useZoom();
  const data = rawData as unknown as GraphNode;
  const nt = data.node_type;
  if (nt.type !== 'Knowledge') return null;

  const color = 'var(--node-knowledge)';
  const preview = `${nt.entity} (${nt.entity_type})`;

  const expandedBodyStyle = useMemo(() => {
    if (isLODEnabled || !expanded) return undefined;
    try {
      const text = `${nt.entity}\n${nt.entity_type}\n${nt.summary}`;
      const minH = estimateExpandedPlainReserveHeight(text, 280, 100);
      return { minHeight: minH, maxHeight: 480, overflowY: 'auto' as const };
    } catch {
      return undefined;
    }
  }, [expanded, isLODEnabled, nt.entity, nt.entity_type, nt.summary]);

  return (
    <BaseCard
      color={color}
      typeLabel={nt.entity_type}
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
      <div className={styles.body}>
        <strong style={{ color }}>{nt.entity}</strong>
        <div style={{ fontSize: 10, color: 'var(--fg-muted)', marginTop: 2 }}>{nt.entity_type}</div>
        <div style={{ marginTop: 6 }}>{nt.summary}</div>
        <div className={styles.confidenceBar} title={`Confidence: ${Math.round(nt.confidence * 100)}%`}>
          <div
            className={styles.confidenceFill}
            style={{ width: `${Math.round(nt.confidence * 100)}%` }}
          />
        </div>
        <div style={{ fontSize: 10, color: 'var(--fg-muted)', marginTop: 2, textAlign: 'right' }}>
          {Math.round(nt.confidence * 100)}% confidence
        </div>
      </div>
    </BaseCard>
  );
}
