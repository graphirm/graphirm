import { useState, useRef, useEffect } from 'react';
import type { GraphNode } from '../../types/graph';
import { usePopoverActions } from '../../context/PopoverContext';
import styles from './NodePopover.module.css';

interface NodePopoverProps {
  node: GraphNode;
  position: { x: number; y: number };
  onClose: () => void;
}

export function NodePopover({ node, position, onClose }: NodePopoverProps) {
  const actions = usePopoverActions();
  const popoverRef = useRef<HTMLDivElement>(null);
  const [editingSummary, setEditingSummary] = useState(false);
  const [summaryText, setSummaryText] = useState('');

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (popoverRef.current && !popoverRef.current.contains(e.target as Node)) {
        onClose();
      }
    }
    function handleEscape(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        onClose();
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleEscape);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [onClose]);

  const handleSteer = () => {
    if (actions?.onSteer) {
      actions.onSteer(node.id);
      onClose();
    }
  };

  const handleRate = (rating: number) => {
    if (actions?.sessionId && actions.onRateTurn) {
      actions.onRateTurn(node.id, rating).then(() => onClose());
    }
  };

  const handleTaskStatus = (status: 'completed' | 'failed') => {
    if (actions?.sessionId && actions.onUpdateTaskStatus) {
      actions.onUpdateTaskStatus(node.id, status).then(() => onClose());
    }
  };

  const handleTogglePin = (pinned: boolean) => {
    if (actions?.sessionId && actions.onTogglePin) {
      actions.onTogglePin(node.id, pinned).then(() => onClose());
    }
  };

  const handleEditSummary = () => {
    if (!editingSummary) {
      setSummaryText(node.node_type.type === 'Knowledge' ? node.node_type.summary : '');
      setEditingSummary(true);
      return;
    }
    if (actions?.sessionId && actions.onEditSummary && summaryText.trim()) {
      actions.onEditSummary(node.id, summaryText.trim()).then(() => {
        setEditingSummary(false);
        onClose();
      });
    }
  };

  const handleCopyToClipboard = async () => {
    const content = node.node_type.type === 'Content' ? node.node_type.body : '';
    if (content) {
      await navigator.clipboard.writeText(content);
      onClose();
    }
  };

  const nodeType = node.node_type.type;

  return (
    <div
      ref={popoverRef}
      className={styles.popover}
      style={{
        left: position.x,
        top: position.y + 100, // position below node
      }}
    >
      {nodeType === 'Interaction' && (
        <>
          {node.node_type.role === 'user' ? (
            <button className={styles.popoverBtn} onClick={handleSteer}>
              ↩ Steer from here
            </button>
          ) : (
            <>
              <button className={styles.popoverBtn} onClick={handleSteer}>
                ↩ Reply from here
              </button>
              <div className={styles.ratingSection}>
                <div className={styles.ratingLabel}>Rate this turn:</div>
                <div className={styles.stars}>
                  {[1, 2, 3, 4, 5].map((rating) => (
                    <button
                      key={rating}
                      className={styles.starBtn}
                      onClick={() => handleRate(rating)}
                      title={`Rate ${rating} star${rating !== 1 ? 's' : ''}`}
                    >
                      ★
                    </button>
                  ))}
                </div>
              </div>
            </>
          )}
        </>
      )}

      {nodeType === 'Content' && (
        <>
          <button className={styles.popoverBtn} onClick={handleSteer}>
            ↩ Steer from here
          </button>
          <button className={styles.popoverBtn} onClick={handleCopyToClipboard}>
            📋 Copy to clipboard
          </button>
        </>
      )}

      {nodeType === 'Task' && (
        <>
          <button
            className={styles.popoverBtn}
            onClick={() => handleTaskStatus('completed')}
          >
            ✅ Mark completed
          </button>
          <button
            className={styles.popoverBtn}
            onClick={() => handleTaskStatus('failed')}
          >
            ❌ Mark failed
          </button>
        </>
      )}

      {nodeType === 'Knowledge' && (
        <>
          <button
            className={styles.popoverBtn}
            onClick={() => handleTogglePin(!node.metadata?.pinned)}
          >
            {node.metadata?.pinned ? '📌 Unpin' : '📌 Pin'}
          </button>
          {editingSummary ? (
            <div className={styles.editSection}>
              <textarea
                className={styles.summaryInput}
                value={summaryText}
                onChange={(e) => setSummaryText(e.target.value)}
                placeholder="Edit summary..."
                rows={3}
                autoFocus
              />
              <div className={styles.editActions}>
                <button
                  className={styles.popoverBtn}
                  onClick={handleEditSummary}
                  disabled={!summaryText.trim()}
                >
                  Save
                </button>
                <button
                  className={styles.popoverBtn}
                  onClick={() => setEditingSummary(false)}
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <button className={styles.popoverBtn} onClick={handleEditSummary}>
              ✏️ Edit summary
            </button>
          )}
        </>
      )}

      <button className={styles.popoverBtn} onClick={onClose}>
        Close
      </button>
    </div>
  );
}