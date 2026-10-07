import { useColumnChrome } from '../layout/shellChrome';
import { MarkdownBody } from './nodes/MarkdownBody';
import hljs from './nodes/hljs-core';
import styles from '../styles/chat.module.css';

export interface BlockViewProps {
  kicker: string;
  content: string;
  state: 'pending' | 'streaming' | 'done';
  n?: number;
  title?: string;
  items?: string[];
}

/** Typed reply block: mono kicker, a check when closed, a cursor while streaming. */
export function BlockView({ kicker, content, state, n, title, items }: BlockViewProps) {
  const column = useColumnChrome();
  const label = n != null ? `${n}. ${kicker}` : kicker;
  const lines = items?.filter(item => item.trim().length > 0) ?? [];
  return (
    <div className={styles.block}>
      <div className={styles.blockKicker}>
        <span>{label}</span>
        {state === 'done' && (
          <span className={styles.blockCheck} aria-label="done">
            ✓
          </span>
        )}
        {state === 'streaming' && <span className={styles.blockCursor} aria-hidden="true" />}
      </div>
      <div className={styles.blockBody}>
        {lines.length > 0 ? (
          <>
            {title ? <div className={styles.blockTitle}>{title}</div> : null}
            <ol className={styles.blockList}>
              {lines.map((item, index) => (
                <li key={`${index}-${item}`}>{item}</li>
              ))}
            </ol>
          </>
        ) : kicker === 'code' ? (
          <pre className={styles.segmentPre}>
            {/* eslint-disable-next-line react/no-danger */}
            <code dangerouslySetInnerHTML={{ __html: hljs.highlightAuto(content).value }} />
          </pre>
        ) : (
          <MarkdownBody content={content} maxHeight={200} bounded={!column} />
        )}
      </div>
    </div>
  );
}
