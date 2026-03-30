import { type RefObject } from 'react';
import type { LayoutMode, NodeFilter } from '../hooks/useGraphData';
import { useTheme } from '../hooks/useTheme';
import styles from './Toolbar.module.css';

const NODE_TYPE_OPTIONS = ['Interaction', 'Agent', 'Content', 'Task', 'Knowledge'] as const;

interface ToolbarProps {
  layoutMode: LayoutMode;
  onLayoutChange: (mode: LayoutMode) => void;
  onAddAnnotation: () => void;
  /** Timeline only: collapse in-place expanded cascade cards on the canvas. */
  onCollapseTimelineCascades?: () => void;
  filter: NodeFilter;
  onFilterChange: (f: NodeFilter) => void;
  matchCount: number;
  totalCount: number;
  searchInputRef?: RefObject<HTMLInputElement | null>;
}

const LAYOUT_LABELS: Record<LayoutMode, string> = {
  dagre: 'DAG',
  timeline: 'Timeline',
  masonry: 'Masonry',
  free: 'Free',
};

const MODES: LayoutMode[] = ['dagre', 'timeline', 'masonry', 'free'];

export function Toolbar({
  layoutMode,
  onLayoutChange,
  onAddAnnotation,
  onCollapseTimelineCascades,
  filter,
  onFilterChange,
  matchCount,
  totalCount,
  searchInputRef,
}: ToolbarProps) {
  const { theme, toggle: toggleTheme } = useTheme();
  const isFiltering = filter.query.trim() !== '' || filter.types.size > 0;

  function toggleType(t: string) {
    const next = new Set(filter.types);
    if (next.has(t)) next.delete(t);
    else next.add(t);
    onFilterChange({ ...filter, types: next });
  }

  return (
    <div className={styles.toolbar}>
      <span className={styles.title}>Graph</span>

      <input
        ref={searchInputRef}
        className={styles.searchInput}
        type="text"
        placeholder="Search nodes…"
        value={filter.query}
        onChange={e => onFilterChange({ ...filter, query: e.target.value })}
      />

      <div className={styles.typeGroup}>
        {NODE_TYPE_OPTIONS.map(t => (
          <button
            key={t}
            className={[styles.typeBtn, filter.types.has(t) ? styles.typeActive : ''].join(' ')}
            onClick={() => toggleType(t)}
            title={`Show only ${t} nodes`}
          >
            {t[0]}
          </button>
        ))}
      </div>

      {isFiltering && (
        <span className={styles.matchCount}>
          {matchCount}/{totalCount}
        </span>
      )}

      {isFiltering && (
        <button
          className={styles.clearBtn}
          onClick={() => onFilterChange({ query: '', types: new Set() })}
          title="Clear filter"
        >
          ✕
        </button>
      )}

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

      {onCollapseTimelineCascades && (
        <button
          type="button"
          className="secondary"
          style={{ fontSize: 11, padding: '3px 8px' }}
          onClick={onCollapseTimelineCascades}
          title="Collapse all expanded cascade cards (timeline)"
        >
          ⊖ Collapse all
        </button>
      )}

      <button className="secondary" style={{ fontSize: 11, padding: '3px 8px' }} onClick={onAddAnnotation}>
        + Note
      </button>

      <button
        onClick={toggleTheme}
        className={styles.toolbarBtn}
        title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
        style={{ marginLeft: 'auto' }}
      >
        {theme === 'dark' ? '☀' : '◉'}
      </button>
    </div>
  );
}
