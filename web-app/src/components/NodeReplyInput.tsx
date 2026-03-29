import { useCallback, useEffect, useRef, useState } from 'react';
import styles from './NodeReplyInput.module.css';

interface NodeReplyInputProps {
  nodeId: string;
  position: { x: number; y: number };
  onSend: (content: string) => void;
  onCancel: () => void;
  isThinking: boolean;
}

export function NodeReplyInput({ nodeId, position, onSend, onCancel, isThinking }: NodeReplyInputProps) {
  const [inputValue, setInputValue] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Focus input when component mounts
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Handle Escape key to cancel
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        onCancel();
      }
    }
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onCancel]);

  const handleSubmit = useCallback(() => {
    const trimmed = inputValue.trim();
    if (trimmed) {
      onSend(trimmed);
      setInputValue('');
      onCancel(); // Close the input after sending
    }
  }, [inputValue, onSend, onCancel]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }, [handleSubmit]);

  return (
    <div 
      className={styles.wrapper}
      style={{
        left: position.x,
        top: position.y + 80, // Position below the node (assuming node height ~80px)
      }}
    >
      <div className={styles.inputBar}>
        <textarea
          ref={inputRef}
          className={styles.input}
          placeholder={`Reply to node ${nodeId.slice(0, 8)}...`}
          value={inputValue}
          onChange={e => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isThinking}
          rows={1}
        />
        <button
          className={styles.sendBtn}
          onClick={handleSubmit}
          disabled={isThinking || !inputValue.trim()}
        >
          Send
        </button>
        <button
          className={styles.cancelBtn}
          onClick={onCancel}
          disabled={isThinking}
        >
          Cancel
        </button>
        {isThinking && (
          <span className={styles.thinkingBadge}>Thinking...</span>
        )}
      </div>
    </div>
  );
}