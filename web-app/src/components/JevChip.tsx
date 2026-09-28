import { useEffect } from 'react';
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
  reason?: string;
  onClose: () => void;
}

/** Routing detail sheet. Feedback actions stay disabled and do not call the network. */
export function JevSheet({ reason, onClose }: JevSheetProps) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      event.preventDefault();
      onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const detail = typeof reason === 'string' && reason.length > 0 ? reason : NO_ROUTING_DETAIL;

  return (
    <div className={styles.jevSheetLayer}>
      <div className={styles.jevSheetBackdrop} onClick={onClose} />
      <dialog open className={styles.jevSheet} aria-label="Routing">
        <p className={styles.jevSheetReason}>{detail}</p>
        <div className={styles.jevSheetActions}>
          <button type="button" disabled>
            Wrong pick
          </button>
          <button type="button" disabled>
            Keep it
          </button>
        </div>
      </dialog>
    </div>
  );
}
