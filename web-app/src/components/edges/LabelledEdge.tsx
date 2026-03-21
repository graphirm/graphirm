import { BaseEdge, EdgeLabelRenderer, getBezierPath, getSmoothStepPath } from '@xyflow/react';
import type { EdgeProps } from '@xyflow/react';
import type { EdgeType } from '../../types/graph';

let _colorCache: Record<string, string> = {};
let _cacheTheme = '';

function getEdgeColor(edgeType: string): string {
  const currentTheme = document.documentElement.getAttribute('data-theme') ?? 'dark';
  if (currentTheme !== _cacheTheme) {
    _colorCache = {};
    _cacheTheme = currentTheme;
  }
  if (!_colorCache[edgeType]) {
    const varName = `--edge-${edgeType.replace(/_/g, '-')}`;
    const val = getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
    _colorCache[edgeType] = val || 'rgba(255,255,255,0.13)';
  }
  return _colorCache[edgeType];
}

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
  const color = getEdgeColor(edgeType);
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
