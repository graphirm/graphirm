import type { CSSProperties, ReactNode } from 'react';
import { Handle, NodeResizer, Position } from '@xyflow/react';
import styles from '../../styles/nodes.module.css';

interface BaseCardProps {
  color: string;
  typeLabel: string;
  timestamp?: string;
  preview: string;
  selected?: boolean;
  expanded: boolean;
  onToggleExpand: () => void;
  children?: ReactNode;
  focused?: boolean;
  /** Pretext-reserved height / overflow for expanded body (accordion-style). */
  expandedBodyStyle?: CSSProperties;
}

function formatTimestamp(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch {
    return '';
  }
}

export function BaseCard({
  color,
  typeLabel,
  timestamp,
  preview,
  selected,
  expanded,
  onToggleExpand,
  children,
  focused = false,
  expandedBodyStyle,
}: BaseCardProps) {
  return (
    <div
      className={[
        styles.card,
        selected ? styles.selected : '',
        expanded ? styles.expanded : '',
        focused ? styles.focused : '',
      ].join(' ')}
      style={{
        borderLeft: `3px solid ${color}`,
        background: `color-mix(in srgb, ${color} 12%, var(--surface-2))`,
      }}
    >
      {/* NodeResizer only activates when selected + expanded */}
      {expanded && (
        <NodeResizer
          minWidth={200}
          minHeight={100}
          isVisible={!!selected}
          lineStyle={{ borderColor: color }}
          handleStyle={{ background: color, borderColor: color }}
        />
      )}

      <Handle type="target" position={Position.Left} style={{ opacity: 0.5 }} />
      <Handle type="source" position={Position.Right} style={{ opacity: 0.5 }} />

      <div className={styles.header}>
        <span
          className={styles.typeBadge}
          style={{ background: color + '33', color }}
        >
          {typeLabel}
        </span>
        {timestamp && (
          <span className={styles.timestamp}>{formatTimestamp(timestamp)}</span>
        )}
        <button
          className={styles.expandToggle}
          onClick={onToggleExpand}
          title={expanded ? 'Collapse' : 'Expand'}
        >
          {expanded ? '▲' : '▼'}
        </button>
      </div>

      {!expanded && (
        <div className={styles.preview}>{preview}</div>
      )}

      {expanded && (
        <div className={styles.expandedBody} style={expandedBodyStyle}>
          {children}
        </div>
      )}
    </div>
  );
}
