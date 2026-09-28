import type { StepRow } from '../chat/steps';
import styles from '../styles/chat.module.css';

/** Collapsed by default. One row lists every tool call in the turn. */
export function StepsRow({ steps }: { steps: StepRow[] }) {
  if (steps.length === 0) return null;
  return (
    <details className={styles.steps}>
      <summary className={styles.stepsSummary}>STEPS · {steps.length}</summary>
      <ol className={styles.stepList}>
        {steps.map((step) => (
          <li key={step.callId} className={styles.stepItem}>
            <span className={styles.stepTitle}>{step.title}</span>
            {step.detail ? <span className={styles.stepDetail}>{step.detail}</span> : null}
          </li>
        ))}
      </ol>
    </details>
  );
}
