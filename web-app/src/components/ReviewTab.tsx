import { buildReviewItems, type ReviewItem } from '../chat/reviewItems';
import styles from './ReviewTab.module.css';

export interface ReviewTabProps {
  sessions: { id: string; status?: string; name?: string }[];
  pendingApproval: { node_id: string; tool_name: string } | null;
  tasks: { id: string; status?: string; executor?: string; title?: string }[];
  onOpenChat: () => void;
}

const KIND_LABEL: Record<ReviewItem['kind'], string> = {
  approval: 'Approval',
  failed: 'Failed',
  paused: 'Paused',
  pi: 'Pi',
};

export function ReviewTab({ sessions, pendingApproval, tasks, onOpenChat }: ReviewTabProps) {
  const items = buildReviewItems({
    pending: pendingApproval,
    sessions,
    tasks,
  });

  return (
    <div className={styles.root}>
      {items.length === 0 ? <p className={styles.empty}>Nothing to review.</p> : null}
      <ul className={styles.list}>
        {items.map((item) => {
          const body = (
            <>
              <span className={styles.kind}>{KIND_LABEL[item.kind]}</span>
              <span className={styles.label}>{item.label}</span>
            </>
          );
          return (
            <li key={`${item.kind}:${item.id}`} className={styles.row}>
              {item.kind === 'approval' ? (
                <button type="button" className={styles.hit} onClick={onOpenChat}>
                  {body}
                </button>
              ) : (
                <div className={styles.hit}>{body}</div>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
