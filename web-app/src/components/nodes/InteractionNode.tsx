import { useEffect, useMemo, useState } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { GraphNode } from '../../types/graph';
import { useSteer } from '../../context/SteerContext';
import { useFocusedNodeId } from '../../context/FocusContext';
import { useZoom } from '../../context/ZoomContext';
import { useCascadeCollapseGeneration } from '../../context/CascadeCollapseContext';
import { useGraphCanvasActions } from '../../context/GraphCanvasActionsContext';
import { api } from '../../api/client';
import { BaseCard } from './BaseCard';
import { MarkdownBody } from './MarkdownBody';
import { RichInteractionPreview } from './RichPreview';
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
  const data = rawData as unknown as GraphNode & { compact?: boolean; precedingInteractionId?: string };
  const nt = data.node_type;
  if (nt.type !== 'Interaction') return null;

  const graphActions = useGraphCanvasActions();
  const [editText, setEditText] = useState(nt.content ?? '');
  const isEditingUser = nt.role === 'user' && graphActions?.editingUserNodeId === id;

  useEffect(() => {
    if (isEditingUser) setEditText(nt.content ?? '');
  }, [isEditingUser, nt.content]);

  const meta = data.metadata as Record<string, unknown> | undefined;
  const userEdited = nt.role === 'user' && meta?.edited === true;

  const toolNameRaw = (data.metadata as Record<string, unknown>)?.tool_name as string | undefined;
  const toolNameNorm = toolNameRaw?.toLowerCase();
  const isDestructiveTool =
    nt.role === 'tool' && toolNameNorm != null && DESTRUCTIVE_TOOL_NAMES.has(toolNameNorm);

  const color = isDestructiveTool ? 'var(--warning)' : (ROLE_COLORS[nt.role] ?? 'var(--node-agent)');
  const roleLabel = nt.role === 'assistant' ? 'agent' : nt.role;
  const typeLabelDisplay = userEdited ? `${roleLabel} · edited` : roleLabel;

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

  const previewTitle = useMemo(() => {
    const t = stripMarkdown(nt.content ?? '');
    return t.length > 320 ? `${t.slice(0, 320)}…` : t;
  }, [nt.content]);

  const handleSaveUserEdit = async () => {
    const trimmed = editText.trim();
    if (!trimmed || !graphActions) return;
    try {
      await api.markInteractionEdited(id, nt.content ?? '');
      graphActions.sendFromGraph(trimmed, data.precedingInteractionId);
      graphActions.setEditingUserNodeId(null);
    } catch (e) {
      console.error(e);
    }
  };

  if (isEditingUser) {
    return (
      <div
        className={styles.card}
        style={{
          borderLeft: `3px solid ${color}`,
          background: `color-mix(in srgb, ${color} 12%, var(--surface-2))`,
          padding: 'var(--space-2)',
          minWidth: 220,
          maxWidth: 320,
        }}
      >
        <div style={{ fontSize: 11, color: 'var(--fg-muted)', marginBottom: 6 }}>
          Edit user message
        </div>
        <textarea
          className={styles.annotationInput}
          value={editText}
          onChange={(e) => setEditText(e.target.value)}
          rows={6}
          style={{ width: '100%', boxSizing: 'border-box' }}
          autoFocus
          onKeyDown={(e) => {
            if (e.key === 'Escape') {
              e.stopPropagation();
              graphActions?.setEditingUserNodeId(null);
            }
          }}
        />
        <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
          <button type="button" className={styles.steerBtn} onClick={() => void handleSaveUserEdit()}>
            Re-send
          </button>
          <button
            type="button"
            className={styles.steerBtn}
            onClick={() => graphActions?.setEditingUserNodeId(null)}
          >
            Cancel
          </button>
        </div>
      </div>
    );
  }

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
        title={userEdited ? `${stripMarkdown(nt.content ?? '')} (edited)` : stripMarkdown(nt.content ?? '')}
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
      typeLabel={typeLabelDisplay}
      timestamp={data.created_at}
      preview={preview}
      previewNode={<RichInteractionPreview content={nt.content ?? ''} maxChars={80} />}
      previewTitle={previewTitle}
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
