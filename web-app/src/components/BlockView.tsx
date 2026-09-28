import { MarkdownBody } from './nodes/MarkdownBody';
import hljs from './nodes/hljs-core';
import styles from '../styles/chat.module.css';

export interface BlockViewProps {
  kicker: string;
  content: string;
  state: 'pending' | 'streaming' | 'done';
}

/** Typed reply block: mono kicker, a check when closed, a cursor while streaming. */
export function BlockView({ kicker, content, state }: BlockViewProps) {
  return (
    <div className={styles.block}>
      <div className={styles.blockKicker}>
        <span>{kicker}</span>
        {state === 'done' && (
          <span className={styles.blockCheck} aria-label="done">
            ✓
          </span>
        )}
        {state === 'streaming' && <span className={styles.blockCursor} aria-hidden="true" />}
      </div>
      <div className={styles.blockBody}>
        {kicker === 'code' ? (
          <pre className={styles.segmentPre}>
            {/* eslint-disable-next-line react/no-danger */}
            <code dangerouslySetInnerHTML={{ __html: hljs.highlightAuto(content).value }} />
          </pre>
        ) : (
          <MarkdownBody content={content} maxHeight={200} />
        )}
      </div>
    </div>
  );
}
