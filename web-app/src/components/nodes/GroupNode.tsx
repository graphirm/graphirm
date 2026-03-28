import { useState } from 'react';
import type { NodeProps } from '@xyflow/react';
import { useZoom } from '../../context/ZoomContext';
import styles from './GroupNode.module.css';

interface GroupData {
  label: string;
  color: string;
  collapsed: boolean;
  onToggle: () => void;
}

export function GroupNode({ data: rawData }: NodeProps) {
  const data = rawData as unknown as GroupData;
  const { isLODEnabled } = useZoom();
  const [collapsed, setCollapsed] = useState(data.collapsed ?? false);

  const toggle = () => {
    if (!isLODEnabled) {
      setCollapsed(c => !c);
    }
    data.onToggle?.();
  };

  return (
    <div
      className={[styles.group, isLODEnabled ? styles.groupLOD : ''].join(' ')}
      style={{ borderColor: data.color + '55' }}
    >
      <div className={styles.groupHeader} style={{ color: data.color }}>
        <span className={styles.groupLabel}>{data.label}</span>
        {!isLODEnabled && (
          <button className={styles.collapseBtn} onClick={toggle}>
            {collapsed ? '▶' : '▼'}
          </button>
        )}
      </div>
    </div>
  );
}
