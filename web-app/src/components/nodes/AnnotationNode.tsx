import { useCallback, useState } from 'react';
import type { NodeProps } from '@xyflow/react';
import { Handle, Position } from '@xyflow/react';
import { useZoom } from '../../context/ZoomContext';
import styles from '../../styles/nodes.module.css';

interface AnnotationData {
  text?: string;
  onTextChange?: (text: string) => void;
}

export function AnnotationNode({ data: rawData }: NodeProps) {
  const data = rawData as unknown as AnnotationData;
  const { isLODEnabled } = useZoom();
  const [text, setText] = useState(data.text ?? '');

  const handleChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value);
    data.onTextChange?.(e.target.value);
  }, [data]);

  return (
    <div className={[styles.annotationCard, isLODEnabled ? styles.annotationCardLOD : ''].join(' ')}>
      <Handle type="target" position={Position.Left} style={{ opacity: 0.4 }} />
      <Handle type="source" position={Position.Right} style={{ opacity: 0.4 }} />
      <textarea
        className={styles.annotationInput}
        value={text}
        onChange={handleChange}
        placeholder="Type annotation…"
        rows={isLODEnabled ? 1 : 2}
      />
    </div>
  );
}
