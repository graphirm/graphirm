import { useEffect, useRef, useState } from 'react';
import type { NodeProps } from '@xyflow/react';
import { Handle, Position } from '@xyflow/react';
import styles from './PromptNode.module.css';

export interface PromptNodeData {
  contextRoot: string | null;
  onSend: (text: string, contextRoot: string | null) => void;
  onCancel: () => void;
}

export function PromptNode({ data }: NodeProps) {
  const d = data as unknown as PromptNodeData;
  const [text, setText] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  const handleSend = () => {
    const t = text.trim();
    if (!t) return;
    d.onSend(t, d.contextRoot);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    e.stopPropagation();
    if (e.key === 'Escape') {
      e.preventDefault();
      d.onCancel();
    }
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className={styles.promptNode}>
      <Handle type="target" position={Position.Left} id="context" style={{ opacity: 0.85 }} />
      <div className={styles.header}>
        <span className={styles.label}>Prompt</span>
        {d.contextRoot ? (
          <span className={styles.contextBadge} title={`Context: ${d.contextRoot}`}>
            ↩ context set
          </span>
        ) : null}
      </div>
      <textarea
        ref={textareaRef}
        className={styles.textarea}
        value={text}
        onChange={e => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Message… Ctrl+Enter or ⌘+Enter to send"
        rows={3}
        aria-label="Prompt message"
      />
      <div className={styles.actions}>
        <button type="button" className={styles.cancelBtn} onClick={d.onCancel}>
          Cancel
        </button>
        <button
          type="button"
          className={styles.sendBtn}
          onClick={handleSend}
          disabled={!text.trim()}
        >
          Send
        </button>
      </div>
    </div>
  );
}
