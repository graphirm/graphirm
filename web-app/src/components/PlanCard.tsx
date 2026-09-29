import { useState } from 'react';
import { runSteerText, type PlanStep } from '../chat/planCard';
import styles from '../styles/chat.module.css';

export interface PlanCardProps {
  steps: PlanStep[];
  onResume: () => void | Promise<void>;
  onSend: (content: string) => void;
}

export function PlanCard({ steps, onResume, onSend }: PlanCardProps) {
  const [enabledById, setEnabledById] = useState<Record<string, boolean>>({});

  if (steps.length === 0) return null;

  const withEnabled = steps.map((step) => ({
    ...step,
    enabled: enabledById[step.id] !== false,
  }));
  const enabledCount = withEnabled.filter((step) => step.enabled).length;

  const toggle = (id: string) => {
    setEnabledById((prev) => ({ ...prev, [id]: prev[id] === false }));
  };

  const run = async () => {
    if (enabledCount === 0) return;
    try {
      await onResume();
    } catch {
      return;
    }
    onSend(runSteerText(withEnabled));
  };

  return (
    <section className={styles.planCard} aria-label="Plan">
      <ul className={styles.planList}>
        {withEnabled.map((step) => (
          <li key={step.id} className={styles.planRow}>
            <label className={styles.planLabel}>
              <input
                type="checkbox"
                checked={step.enabled}
                onChange={() => toggle(step.id)}
              />
              <span className={styles.planTitle}>{step.title}</span>
            </label>
            {step.note !== '' && <p className={styles.planNote}>{step.note}</p>}
            {(step.risky || step.executor === 'pi') && (
              <div className={styles.planMeta}>
                {step.risky && <span className={styles.planRisky}>risky</span>}
                {step.executor === 'pi' && <span className={styles.planExecutor}>pi</span>}
              </div>
            )}
          </li>
        ))}
      </ul>
      <button
        type="button"
        className={styles.planRun}
        disabled={enabledCount === 0}
        onClick={() => {
          void run();
        }}
      >
        Run {enabledCount} steps
      </button>
    </section>
  );
}
