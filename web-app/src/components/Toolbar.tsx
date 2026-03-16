import type { LayoutMode } from '../hooks/useGraphData';
import styles from './Toolbar.module.css';

interface ToolbarProps {
  layoutMode: LayoutMode;
  onLayoutChange: (mode: LayoutMode) => void;
  onAddAnnotation: () => void;
}

const LAYOUT_LABELS: Record<LayoutMode, string> = {
  dagre: 'DAG',
  timeline: 'Timeline',
  free: 'Free',
};

const MODES: LayoutMode[] = ['dagre', 'timeline', 'free'];

export function Toolbar({ layoutMode, onLayoutChange, onAddAnnotation }: ToolbarProps) {
  return (
    <div className={styles.toolbar}>
      <span className={styles.title}>Graph</span>
      <div className={styles.layoutGroup}>
        {MODES.map(mode => (
          <button
            key={mode}
            className={[styles.layoutBtn, layoutMode === mode ? styles.active : ''].join(' ')}
            onClick={() => onLayoutChange(mode)}
          >
            {LAYOUT_LABELS[mode]}
          </button>
        ))}
      </div>
      <button className="secondary" style={{ fontSize: 11, padding: '3px 8px' }} onClick={onAddAnnotation}>
        + Note
      </button>
    </div>
  );
}
