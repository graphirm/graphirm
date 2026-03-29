import type { Node, Edge } from '@xyflow/react';
import type { GraphNode } from '../types/graph';

// Width estimates for collision avoidance — must match or exceed CSS --card-max-width (280px).
const NODE_WIDTHS: Record<string, number> = {
  Interaction: 280,
  Agent: 280,
  Content: 280,
  Knowledge: 280,
  Task: 280,
};
const DEFAULT_NODE_WIDTH = 280;
const BAND_GAP = 32;

// Y positions per type band — spaced 140px apart to avoid tall cards overlapping.
export const TYPE_Y: Record<string, number> = {
  Agent: 0,
  Task: 140,
  Interaction: 280,
  Content: 420,
  Knowledge: 560,
};

export const TYPE_LABELS: Record<string, string> = {
  Agent: 'Agent',
  Task: 'Task',
  Interaction: 'Interaction',
  Content: 'Content',
  Knowledge: 'Knowledge',
};

export function applyTimelineLayout(
  nodes: Node[],
  rawNodes: GraphNode[],
  _edges: Edge[],
  canvasWidth: number,
): Node[] {
  if (nodes.length === 0) return nodes;

  const times = rawNodes
    .map(n => new Date(n.created_at).getTime())
    .filter(t => !isNaN(t));

  if (times.length === 0) return nodes;

  const tMin = Math.min(...times);
  const tMax = Math.max(...times);
  const tRange = tMax - tMin || 1;
  const padding = 80;

  const rawById = new Map(rawNodes.map(n => [n.id, n]));

  // Pass 1: compute natural X/Y from timestamps.
  const positioned = nodes.map((node) => {
    const raw = rawById.get(node.id);
    if (!raw) return node;

    const t = new Date(raw.created_at).getTime();
    const x = isNaN(t)
      ? padding
      : padding + ((t - tMin) / tRange) * (canvasWidth - padding * 2);

    const y = TYPE_Y[raw.node_type.type] ?? 280;

    return { ...node, position: { x, y } };
  });

  // Pass 2: stagger overlapping nodes within each type band.
  // Sort each band by natural X and nudge right when nodes would overlap.
  const byBand = new Map<string, Node[]>();
  for (const node of positioned) {
    const raw = rawById.get(node.id);
    const band = raw?.node_type.type ?? 'Interaction';
    if (!byBand.has(band)) byBand.set(band, []);
    byBand.get(band)!.push(node);
  }

  const staggered = new Map(positioned.map(n => [n.id, n]));
  for (const [band, bandNodes] of byBand) {
    const nodeWidth = NODE_WIDTHS[band] ?? DEFAULT_NODE_WIDTH;
    const sorted = [...bandNodes].sort((a, b) => a.position.x - b.position.x);
    let nextAllowedX = -Infinity;
    for (const node of sorted) {
      const x = Math.max(node.position.x, nextAllowedX);
      nextAllowedX = x + nodeWidth + BAND_GAP;
      staggered.set(node.id, { ...node, position: { ...node.position, x } });
    }
  }

  return positioned.map(n => staggered.get(n.id) ?? n);
}
