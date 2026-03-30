import { useEffect, useMemo, useState } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { GraphNode } from '../../types/graph';
import { useSteer } from '../../context/SteerContext';
import { useFocusedNodeId } from '../../context/FocusContext';
import { useZoom } from '../../context/ZoomContext';
import { useCascadeCollapseGeneration } from '../../context/CascadeCollapseContext';
import { BaseCard } from './BaseCard';
import { MarkdownBody } from './MarkdownBody';
import { estimateInteractionExpandedReserveHeight } from '../../layout/pretextDimensions';
import styles from '../../styles/nodes.module.css';

const DESTRUCTIVE_TOOL_NAMES = new Set(['write', 'edit', 'bash']);

const ROLE_ICONS: Record<string, string> = {
  user: 'U',
  assistant: 'A',
  tool: 'T',
  system: 'S',
};

const ROLE_COLORS: Record<string, string> = {
  user: 'var(--accent)',
  assistant: 'var(--node-agent)',
  tool: 'var(--node-content)',
  system: 'var(--fg-muted)',
};

export function InteractionNode({ id, data: rawData, selected }: NodeProps) {
  const [expanded, setExpanded] = useState(false);
  const [localExpanded, setLocalExpanded] = useState(false);
  const onSteer = useSteer();
  const focusedNodeId = useFocusedNodeId();
  const { isLODEnabled } = useZoom();
  const cascadeCollapseGeneration = useCascadeCollapseGeneration();
  const data = rawData as unknown as GraphNode & { compact?: boolean };
  const nt = data.node_type;
  if (nt.type !== 'Interaction') return null;

  const toolNameRaw = (data.metadata as Record<string, unknown>)?.tool_name as string | undefined;
  const toolNameNorm = toolNameRaw?.toLowerCase();
  const isDestructiveTool =
    nt.role === 'tool' && toolNameNorm != null && DESTRUCTIVE_TOOL_NAMES.has(toolNameNorm);

  const color = isDestructiveTool ? 'var(--warning)' : (ROLE_COLORS[nt.role] ?? 'var(--node-agent)');
  const roleLabel = nt.role === 'assistant' ? 'agent' : nt.role;

  useEffect(() => {
    setLocalExpanded(false);
  }, [cascadeCollapseGeneration]);

  const stripMarkdown = (text: string): string => {
    return text
      .replace(/\*\*(.*?)\*\*/g, '$1')
      .replace(/\*(.*?)\*/g, '$1')
      .replace(/__(.*?)__/g, '$1')
      .replace(/_(.*?)_/g, '$1')
      .replace(/`([^`]+)`/g, '$1')
      .replace(/\[([^\]]*)\]\([^)]+\)/g, '$1')
      .replace(/^#+\s*/gm, '')
      .replace(/^-\s+/gm, '')
      .replace(/^>\s+/gm, '')
      .replace(/\n+/g, ' ')
      .trim();
  };

  const preview = stripMarkdown(nt.content ?? '').slice(0, 80) + ((nt.content ?? '').length > 80 ? '…' : '');

  const expandedBodyStyle = useMemo(() => {
    if (isLODEnabled || !expanded) return undefined;
    try {
      const minH = estimateInteractionExpandedReserveHeight(nt.content ?? '');
      return {
        minHeight: minH,
        maxHeight: 480,
        overflowY: 'auto' as const,
      };
    } catch {
      return undefined;
    }
  }, [expanded, isLODEnabled, nt.content]);

  // Compact mode: cascade intermediate cards in timeline layout.
  // localExpanded lets the user click to expand in-place.
  const isCompact = data.compact === true && !localExpanded;
  if (isCompact) {
    const label = toolNameRaw ?? (stripMarkdown(nt.content ?? '').slice(0, 60) || roleLabel);
    const icon = ROLE_ICONS[nt.role] ?? '?';
    const focused = focusedNodeId === id;
    return (
      <div
        className={`${styles.compactCard}${focused ? ` ${styles.focused}` : ''}`}
        onClick={() => setLocalExpanded(true)}
        title={stripMarkdown(nt.content ?? '')}
        style={{ '--compact-color': color } as React.CSSProperties}
      >
        <span className={styles.roleIcon}>{icon}</span>
        <span className={styles.compactLabel}>{label}</span>
      </div>
    );
  }

  return (
    <BaseCard
      color={color}
      typeLabel={roleLabel}
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
      {localExpanded && (
        <button
          className={styles.steerBtn}
          onClick={() => setLocalExpanded(false)}
          style={{ marginBottom: 4 }}
        >
          ↑ Collapse
        </button>
      )}
      <MarkdownBody content={nt.content ?? ''} maxHeight={320} />
      {nt.token_count != null && (
        <div style={{ fontSize: 10, color: 'var(--fg-muted)', textAlign: 'right' }}>
          {nt.token_count} tokens
        </div>
      )}
      {onSteer && (
        <button className={styles.steerBtn} onClick={() => onSteer(data.id)}>
          ↩ Steer from here
        </button>
      )}
    </BaseCard>
  );
}
