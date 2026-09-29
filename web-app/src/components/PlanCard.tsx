import { useRef, useState } from 'react';
import { runSteerText, type PlanStep } from '../chat/planCard';
import styles from '../styles/chat.module.css';

export interface PlanCardProps {
  steps: PlanStep[];
  /** Missing ids stay enabled. Owned by DecisionShell so tab changes do not reset it. */
  enabledById: Record<string, boolean>;
  onToggle: (id: string) => void;
  onResume: () => void | Promise<void>;
  onSend: (content: string) => void;
  isThinking: boolean;
}

export function PlanCard({
  steps,
  enabledById,
  onToggle,
  onResume,
  onSend,
  isThinking,
}: PlanCardProps) {
  const running = useRef(false);
  const [locked, setLocked] = useState(false);

  if (steps.length === 0) return null;

  const withEnabled = steps.map((step) => ({
    ...step,
    enabled: enabledById[step.id] !== false,
  }));
  const enabledCount = withEnabled.filter((step) => step.enabled).length;
  const runDisabled = enabledCount === 0 || locked || isThinking;

  const run = async () => {
    if (running.current || isThinking || enabledCount === 0) return;
    running.current = true;
    setLocked(true);
    try {
      await onResume();
    } catch {
      return;
    } finally {
      running.current = false;
      setLocked(false);
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
                onChange={() => onToggle(step.id)}
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
        disabled={runDisabled}
        onClick={() => {
          void run();
        }}
      >
        Run {enabledCount} steps
      </button>
    </section>
  );
}
