import { useState } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { GraphNode } from '../../types/graph';
import { useSteer } from '../../context/SteerContext';
import { BaseCard } from './BaseCard';
import { MarkdownBody } from './MarkdownBody';
import styles from '../../styles/nodes.module.css';

export function InteractionNode({ data: rawData, selected }: NodeProps) {
  const [expanded, setExpanded] = useState(false);
  const onSteer = useSteer();
  const data = rawData as unknown as GraphNode;
  const nt = data.node_type;
  if (nt.type !== 'Interaction') return null;

  const isUser = nt.role === 'user';
  const color = isUser ? 'var(--accent)' : 'var(--node-agent)';
  const roleLabel = nt.role === 'assistant' ? 'agent' : nt.role;
  
  // Strip markdown for collapsed preview
  const stripMarkdown = (text: string): string => {
    return text
      .replace(/\*\*(.*?)\*\*/g, '$1') // bold
      .replace(/\*(.*?)\*/g, '$1')      // italic
      .replace(/__(.*?)__/g, '$1')      // bold
      .replace(/_(.*?)_/g, '$1')        // italic
      .replace(/`([^`]+)`/g, '$1')      // inline code
      .replace(/\[([^\]]*)\]\([^)]+\)/g, '$1') // links
      .replace(/^#+\s*/gm, '')          // headers
      .replace(/^-\s+/gm, '')           // lists
      .replace(/^>\s+/gm, '')           // blockquotes
      .replace(/\n+/g, ' ')             // newlines to spaces
      .trim();
  };
  
  const preview = stripMarkdown(nt.content ?? '').slice(0, 80) + ((nt.content ?? '').length > 80 ? '…' : '');

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
