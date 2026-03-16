import { useState } from 'react';
import type { NodeProps } from '@xyflow/react';
import styles from './GroupNode.module.css';

interface GroupData {
  label: string;
  color: string;
  collapsed: boolean;
  onToggle: () => void;
}

export function GroupNode({ data: rawData }: NodeProps) {
  const data = rawData as unknown as GroupData;
  const [collapsed, setCollapsed] = useState(data.collapsed ?? false);

  const toggle = () => {
    setCollapsed(c => !c);
    data.onToggle?.();
  };

  return (
    <div
      className={styles.group}
      style={{ borderColor: data.color + '55' }}
    >
      <div className={styles.groupHeader} style={{ color: data.color }}>
        <span className={styles.groupLabel}>{data.label}</span>
        <button className={styles.collapseBtn} onClick={toggle}>
          {collapsed ? '▶' : '▼'}
        </button>
      </div>
    </div>
  );
}
