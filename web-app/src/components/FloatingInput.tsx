import { useCallback, useEffect, useRef, useState } from 'react';
import styles from './FloatingInput.module.css';

interface FloatingInputProps {
  onSend: (content: string) => void;
  isThinking: boolean;
  chatCollapsed: boolean;
}

export function FloatingInput({ onSend, isThinking, chatCollapsed }: FloatingInputProps) {
  const [visible, setVisible] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  // Toggle expanded state
  const toggleVisible = useCallback(() => {
    setVisible(v => !v);
  }, []);

  // Hide to hint state
  const collapse = useCallback(() => {
    setVisible(false);
    setInputValue('');
  }, []);

  // Handle keyboard shortcuts
  useEffect(() => {
    if (!chatCollapsed) return;
    function handleKeyDown(e: KeyboardEvent) {
      // If Escape is pressed and visible, collapse
      if (e.key === 'Escape' && visible) {
        collapse();
        return;
      }

      // If / or Enter is pressed and no input is focused, expand
      const isInputElement = 
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        (e.target instanceof HTMLElement && e.target.isContentEditable);

      if (!isInputElement && (e.key === '/' || e.key === 'Enter')) {
        // Prevent default for Enter to avoid form submission
        if (e.key === 'Enter') {
          e.preventDefault();
        }
        setVisible(true);
      }
    }

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [chatCollapsed, visible, collapse]);

  // Focus input when visible changes to true
  useEffect(() => {
    if (visible) inputRef.current?.focus();
  }, [visible]);

  // If chat is not collapsed, render nothing
  if (!chatCollapsed) {
    return null;
  }

  // Hint state (collapsed bar)
  if (!visible) {
    return (
      <div className={styles.wrapper}>
        <div 
          className={styles.hint}
          onClick={toggleVisible}
        >
          Press / or Enter to send
        </div>
      </div>
    );
  }

  // Expanded input state
  const handleSubmit = useCallback(() => {
    const trimmed = inputValue.trim();
    if (trimmed) {
      onSend(trimmed);
      setInputValue('');
      setVisible(false);
    }
  }, [inputValue, onSend]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }, [handleSubmit]);

  return (
    <div className={styles.wrapper}>
      <div className={styles.inputBar}>
        <input
          ref={inputRef}
          type="text"
          className={styles.input}
          placeholder="Send a message..."
          value={inputValue}
          onChange={e => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isThinking}
        />
        <button
          className={styles.sendBtn}
          onClick={handleSubmit}
          disabled={isThinking || !inputValue.trim()}
        >
          Send
        </button>
        {isThinking && (
          <span className={styles.thinkingBadge}>Thinking...</span>
        )}
      </div>
    </div>
  );
}
