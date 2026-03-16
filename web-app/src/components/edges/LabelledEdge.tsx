import { BaseEdge, EdgeLabelRenderer, getBezierPath, getSmoothStepPath } from '@xyflow/react';
import type { EdgeProps } from '@xyflow/react';
import type { EdgeType } from '../../types/graph';

const EDGE_COLORS: Record<string, string> = {
  responds_to: '#ffffff44',
  reads: '#3b82f688',
  modifies: '#f9731688',
  produces: '#4ade8088',
  depends_on: '#a855f788',
  spawned_by: '#ec489988',
  contains: '#4ade8044',
  summarizes: '#ce93d844',
  delegates_to: '#fbbf2488',
  follows_up: '#ffffff33',
  steers: '#0e639c88',
  relates_to: '#ffffff22',
  derived_from: '#ce93d866',
  approved_by: '#16a34a88',
  rejected_by: '#dc262688',
};

const STROKE_WIDTH: Record<string, number> = {
  depends_on: 2.5,
  produces: 2.5,
};

// Hierarchical edges use smooth-step; cross-cutting use bezier.
const SMOOTH_STEP_TYPES: EdgeType[] = ['responds_to', 'produces', 'contains', 'spawned_by', 'follows_up'];

function toDisplayLabel(edgeType: string): string {
  return edgeType.replace(/_/g, ' ');
}

export function LabelledEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  label,
  markerEnd,
}: EdgeProps & { data?: { edge_type?: EdgeType } }) {
  const edgeType = (data?.edge_type ?? label ?? '') as EdgeType;
  const color = EDGE_COLORS[edgeType] ?? '#ffffff22';
  const strokeWidth = STROKE_WIDTH[edgeType] ?? 1.5;
  const useSmooth = SMOOTH_STEP_TYPES.includes(edgeType);

  const pathParams = {
    sourceX, sourceY, targetX, targetY,
    sourcePosition, targetPosition,
  };

  const [edgePath, labelX, labelY] = useSmooth
    ? getSmoothStepPath(pathParams)
    : getBezierPath(pathParams);

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        style={{ stroke: color, strokeWidth, opacity: 0.85 }}
      />
      {edgeType && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
              fontSize: 9,
              color: '#888',
              pointerEvents: 'none',
              whiteSpace: 'nowrap',
            }}
            className="nodrag nopan"
          >
            {toDisplayLabel(edgeType)}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}
