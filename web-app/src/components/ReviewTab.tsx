import { buildReviewItems, type ReviewItem } from '../chat/reviewItems';
import styles from './ReviewTab.module.css';

export interface ReviewTabProps {
  sessions: { id: string; status?: string; name?: string }[];
  pendingApproval: {
    node_id: string;
    tool_name: string;
    is_pause?: boolean;
    session_id?: string;
  } | null;
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
  const sessionName = pendingApproval?.session_id
    ? sessions.find((session) => session.id === pendingApproval.session_id)?.name
    : undefined;
  const items = buildReviewItems({
    pending: pendingApproval
      ? {
          node_id: pendingApproval.node_id,
          tool_name: pendingApproval.tool_name,
          is_pause: pendingApproval.is_pause,
          session_name: sessionName,
        }
      : null,
    sessions,
    tasks,
  });

  return (
    <div className={styles.root}>
      {items.length === 0 ? (
        <div className={styles.empty}>
          <p className={styles.kicker}>Review</p>
          <p className={styles.emptyTitle}>Nothing needs you.</p>
          <p className={styles.emptyBody}>
            Approvals, failed or paused sessions, and running Pi work show up here.
          </p>
        </div>
      ) : null}
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
