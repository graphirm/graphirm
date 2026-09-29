import { useEffect, useRef, useState } from 'react';
import { api } from '../api/client';
import styles from '../styles/chat.module.css';

const NO_ROUTING_DETAIL = 'No routing detail';

interface JevChipProps {
  label: string;
  onOpen: () => void;
}

/** Button whose label is the formatted routing chip. */
export function JevChip({ label, onOpen }: JevChipProps) {
  return (
    <button type="button" className={styles.jevChip} onClick={onOpen}>
      {label}
    </button>
  );
}

interface JevSheetProps {
  interactionId: string;
  reason?: string;
  onClose: () => void;
}

/** Routing detail sheet. Feedback posts `wrong` or `keep` for this assistant message. */
export function JevSheet({ interactionId, reason, onClose }: JevSheetProps) {
  const [recorded, setRecorded] = useState<'wrong' | 'keep' | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inFlight = useRef(false);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      event.preventDefault();
      onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  async function send(verdict: 'wrong' | 'keep') {
    if (recorded || inFlight.current) return;
    inFlight.current = true;
    setError(null);
    try {
      await api.postRoutingFeedback(interactionId, verdict);
      setRecorded(verdict);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      inFlight.current = false;
    }
  }

  const detail = typeof reason === 'string' && reason.length > 0 ? reason : NO_ROUTING_DETAIL;
  const locked = recorded !== null;

  return (
    <div className={styles.jevSheetLayer}>
      <div className={styles.jevSheetBackdrop} onClick={onClose} />
      <dialog open className={styles.jevSheet} aria-label="Routing">
        <p className={styles.jevSheetReason}>{detail}</p>
        {recorded && <p className={styles.jevSheetReason}>{recorded}</p>}
        {error && <p className={styles.jevSheetReason}>{error}</p>}
        <div className={styles.jevSheetActions}>
          <button type="button" disabled={locked} onClick={() => void send('wrong')}>
            Wrong pick
          </button>
          <button type="button" disabled={locked} onClick={() => void send('keep')}>
            Keep it
          </button>
        </div>
      </dialog>
    </div>
  );
}
