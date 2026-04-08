import type { SegmentPart } from '../types/graph';
import { MarkdownBody } from './nodes/MarkdownBody';
import hljs from './nodes/hljs-core';
import styles from '../styles/chat.module.css';

function badgeClassForType(t: string): string {
  const key = `seg_${t}` as keyof typeof styles;
  return typeof styles[key] === 'string' ? (styles[key] as string) : '';
}

export function SegmentCard({ segment }: { segment: SegmentPart }) {
  const typeClass = badgeClassForType(segment.type);

  if (segment.type === 'code') {
    const html = hljs.highlightAuto(segment.content).value;
    return (
      <div className={`${styles.segmentCard} ${styles.seg_code} ${typeClass}`}>
        <span className={styles.segBadge}>{segment.type}</span>
        <pre className={styles.segmentPre}>
          {/* eslint-disable-next-line react/no-danger */}
          <code dangerouslySetInnerHTML={{ __html: html }} />
        </pre>
      </div>
    );
  }

  return (
    <div className={`${styles.segmentCard} ${typeClass}`}>
      <span className={styles.segBadge}>{segment.type}</span>
      <MarkdownBody content={segment.content} maxHeight={200} />
    </div>
  );
}
