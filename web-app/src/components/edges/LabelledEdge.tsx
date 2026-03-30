import { BaseEdge, EdgeLabelRenderer, getBezierPath, getSmoothStepPath, useViewport } from '@xyflow/react';
import type { EdgeProps } from '@xyflow/react';
import type { EdgeType } from '../../types/graph';

const LABEL_ZOOM_THRESHOLD = 0.6;

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
  const { zoom } = useViewport();

  const pathParams = {
    sourceX, sourceY, targetX, targetY,
    sourcePosition, targetPosition,
  };

  const [edgePath, labelX, labelY] = useSmooth
    ? getSmoothStepPath(pathParams)
    : getBezierPath(pathParams);

  const showLabel = edgeType && zoom >= LABEL_ZOOM_THRESHOLD;

  // Nudge label slightly off the stroke (perpendicular) for readability — cheap
  // substitute until Pretext line-aware anchors exist on node cards.
  const dx = targetX - sourceX;
  const dy = targetY - sourceY;
  const len = Math.hypot(dx, dy) || 1;
  const labelNudge = 11;
  const ox = (-dy / len) * labelNudge;
  const oy = (dx / len) * labelNudge;

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        style={{ stroke: color, strokeWidth, opacity: 0.85 }}
      />
      {showLabel && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX + ox}px, ${labelY + oy}px)`,
              fontSize: 9,
              fontWeight: 600,
              letterSpacing: '0.02em',
              color: 'var(--fg-muted, #888)',
              pointerEvents: 'none',
              whiteSpace: 'nowrap',
              padding: '2px 6px',
              borderRadius: 4,
              background: 'color-mix(in srgb, var(--surface-2, #1f1f1f) 88%, transparent)',
              border: '1px solid color-mix(in srgb, var(--border, #2e2e2e) 70%, transparent)',
              boxShadow: '0 1px 2px rgba(0,0,0,0.25)',
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
