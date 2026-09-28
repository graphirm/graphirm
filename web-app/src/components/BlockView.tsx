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
      <div className={styles.blockBody}>{content}</div>
    </div>
  );
}
