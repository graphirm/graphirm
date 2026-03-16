import { useEffect } from 'react';

interface ShortcutHandlers {
  onFitView: () => void;
  onToggleLayout: () => void;
  onNewSession: () => void;
  onFocusChat: () => void;
}

/**
 * Global keyboard shortcuts for the whiteboard:
 * - F  — fit view (zoom-to-fit all nodes)
 * - L  — cycle layout mode (dagre → timeline → free)
 * - N  — new session
 * - /  — focus chat input
 * - Escape — blur active element / close expansions
 */
export function useKeyboardShortcuts({
  onFitView,
  onToggleLayout,
  onNewSession,
  onFocusChat,
}: ShortcutHandlers) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Ignore when user is typing in an input, textarea, or contenteditable.
      const tag = (e.target as HTMLElement)?.tagName.toLowerCase();
      if (tag === 'input' || tag === 'textarea' || (e.target as HTMLElement)?.isContentEditable) {
        if (e.key === 'Escape') {
          (e.target as HTMLElement).blur();
        }
        return;
      }

      if (e.metaKey || e.ctrlKey || e.altKey) return;

      switch (e.key.toLowerCase()) {
        case 'f':
          e.preventDefault();
          onFitView();
          break;
        case 'l':
          e.preventDefault();
          onToggleLayout();
          break;
        case 'n':
          e.preventDefault();
          onNewSession();
          break;
        case '/':
          e.preventDefault();
          onFocusChat();
          break;
        default:
          break;
      }
    };

    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onFitView, onToggleLayout, onNewSession, onFocusChat]);
}
